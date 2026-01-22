# app/streamlit_app.py
import json
import logging
import os
import requests
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import weaviate
import weaviate.classes as wvc

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langsmith import Client, traceable
from pydantic import BaseModel, Field

from utils.jira_client import JiraIssue, JiraClient, extract_jira_keys_from_text, ATLASSIAN_INSTANCE_URL
from utils.jira_ticket_processing import JiraIssueLLMProcessor, LogAnalysisOutput
from utils.weaviate_utils import JIRA_COLLECTION_NAME


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatbot_debug.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
EMBEDDING_MODEL_NAME = "nomic-embed-text-v2-moe"  # nomic-embed-text?
VLM_MODEL_NAME = "qwen2.5vl:7b"
LLM_MODEL_NAME = "llama3"
JIRA_BASE_URL = os.environ.get("ATLASSIAN_INSTANCE_URL", "").replace("/rest/api/3", "").rstrip("/")
MAX_NUM_SIMILAR_TICKETS_FROM_RAG = 5
NUM_MOST_RELEVANT_TICKETS_TO_RETURN = 5
MAX_MSG_HISTORY = 10
NUM_EMBEDDING_RETRIES = 2
MAX_EMBEDDINGS_INPUT_CHARS = 4000


EMBEDDER = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url="http://localhost:11434")


class IntentClassification(Enum):
    FOLLOW_UP_ON_CURRENT_TICKET = "follow_up_on_current_ticket"
    ANALYZE_NEW_TICKET = "analyze_new_ticket"
    UNRELATED_CHAT = "unrelated_chat"
    MORE_THAN_ONE_KEY = "more_than_one_key"


class IntentOutput(BaseModel):
    """Classification of user intent."""
    intent: IntentClassification = Field(description="The classified intent of the user message")


class SimilarTicketInfo(BaseModel):
    """Information about a similar Jira ticket."""
    key: str = Field(description="Jira ticket key")
    summary: str = Field(description="Summary of the Jira ticket")
    similarity_reason: str = Field(description="Reason why this ticket is similar to the current ticket")
    status: str = Field(description="Current status of the Jira ticket")
    title: str = Field(description="Title of the Jira ticket")
    score: float = Field(description="Similarity score")
    issue_type: str = Field(description="Type of the Jira issue")


class FinalAnalysisOutput(BaseModel):
    """Final analysis output structure."""
    ticket_summary: str = Field(description="Concise summary of the Jira ticket")
    key_issues: List[str] = Field(description="List of key issues based on the current ticket and patterns from similar tickets")
    root_causes: List[str] = Field(description="List of likely root causes based on the current ticket and patterns from similar tickets")
    similar_tickets: List[SimilarTicketInfo] = Field(description="List of similar tickets with details")
    suggested_solutions: List[str] = Field(description="List of suggested solutions or mitigation steps supported by evidence from similar tickets")
    important_notes: List[str] = Field(description="List of important notes, warnings, or risks")


class JiraChatBot:
    def __init__(
            self,
            model: str = LLM_MODEL_NAME,
            max_history: int = MAX_MSG_HISTORY,
    ):
        self.chat = ChatOllama(model=model, base_url="http://localhost:11434", verbose=True, temperature=0.1)
        self.max_history = max_history
        self.system_prompt = """You are a helpful Jira ticket solver assistant.
        Be concise and factual.
        Do not hallucinate. If something is unknown, say "unknown".
        """
        self.history = [SystemMessage(content=self.system_prompt)]
        self.embedder = EMBEDDER

        # ---- Connect to Weaviate ----
        self.db_client = weaviate.connect_to_local()

        # ---- Connect to Langsmith ----
        self.langsmith_client = Client()

    # ------------------------
    # Internal helpers
    # ------------------------
    def _truncate_history(self):
        """Keep system prompt + last N messages"""
        system_msg = self.history[0]
        self.history = [system_msg] + self.history[-self.max_history:]

    def _add_user(self, text: str):
        self.history.append(HumanMessage(content=text))
        self._truncate_history()

    def _add_assistant(self, text: str):
        self.history.append(AIMessage(content=text))
        self._truncate_history()

    # ------------------------
    # Public API
    # ------------------------
    def run(self) -> None:
        # ---- Streamlit UI ----
        st.set_page_config(page_title="Jira Ticket Assistant", layout="wide")
        st.title("🎫 Jira Ticket Assistant")
        st.sidebar.markdown("### 📖 How to Use")
        st.sidebar.markdown("""
        1. Enter a Jira ticket key (e.g., `GC-123`) or paste a URL
        2. Get automatic analysis with similar tickets
        3. Continue chatting to ask follow-up questions
        4. Analyze a new ticket anytime by entering another key
        """)

        self._initialize_session_state()

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_input := st.chat_input(
                "Enter a Jira ticket key/URL (e.g., GC-123 or https://.../browse/GC-123) or ask a question..."):

            # Add user message to chat history
            self._add_user(user_input)

            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                # Create placeholders
                status_placeholder = st.empty()
                response_placeholder = st.empty()

                # Show status immediately BEFORE any LLM calls
                status_placeholder.markdown("🤔 **Thinking...**")

                intent: IntentClassification = self._classify_intent(user_input)

                if intent == IntentClassification.ANALYZE_NEW_TICKET:
                    potential_keys = extract_jira_keys_from_text(user_input)
                    response = self._fetch_and_analyze_ticket(potential_keys[0], status_placeholder)
                else:
                    response = self._continue_conversation(user_input, intent)

                # Clear the status message and show response
                status_placeholder.empty()
                response_placeholder.markdown(response)

            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Add response to chat history
            self._add_assistant(response)

            st.rerun()

        # Display current ticket info in sidebar
        if st.session_state.current_ticket:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎫 Current Ticket")
            ticket = st.session_state.current_ticket
            st.sidebar.markdown(f"**{ticket['key']}**")
            st.sidebar.markdown(f"Status: `{ticket['status']}`")
            if JIRA_BASE_URL:
                st.sidebar.markdown(f"[View in Jira →]({JIRA_BASE_URL}/browse/{ticket['key']})")

    def close(self) -> None:
        """Close connections properly."""
        if self.db_client:
            self.db_client.close()

    def reset(self) -> None:
        """Clear conversation history"""
        self.history = [SystemMessage(content=self.system_prompt)]

    # ------------------------
    # Internal Functions
    # ------------------------
    @staticmethod
    def _initialize_session_state():
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_ticket" not in st.session_state:
            st.session_state.current_ticket = None
        if "similar_tickets" not in st.session_state:
            st.session_state.similar_tickets = []

    def _fetch_and_analyze_ticket(self, jira_key: str, status_placeholder=None) -> str:
        """Fetch ticket from Jira and find similar tickets."""
        # Update status
        if status_placeholder:
            status_placeholder.markdown("📥 **Analyzing ticket...**")

        try:
            jira_issue: JiraIssue = JiraClient().fetch_issue_by_key(jira_key)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return f"❌ Jira issue with key '{jira_key}' not found. Try another key."
            else:
                return f"❌ Failed to fetch Jira issue: {e}"
        except Exception as e:
            return f"❌ Failed to fetch Jira issue: {e}"

        issue_summary, logs_analysis = JiraIssueLLMProcessor(LLM_MODEL_NAME, VLM_MODEL_NAME).process_issue(
            jira_issue, status_placeholder)
        # Store current ticket
        st.session_state.current_ticket = {
            "key": jira_issue.key,
            "summary": issue_summary,
            "description": jira_issue.description or "",
            "status": jira_issue.status,
            "priority": jira_issue.priority,
            "created": jira_issue.created,
            "labels": jira_issue.labels,
        }

        # Search for similar tickets
        if status_placeholder:
            status_placeholder.markdown("🔍 **Finding similar tickets...**")

        query_text = (issue_summary + "\n\n" + jira_issue.description)[:MAX_EMBEDDINGS_INPUT_CHARS]

        # Generate embedding using Ollama
        query_embedding = self.embedder.embed_documents([query_text])[0]

        # Query RAG for similar tickets
        jira_collection = self.db_client.collections.get(JIRA_COLLECTION_NAME)
        similar_tickets_from_rag = jira_collection.query.near_vector(
            near_vector=query_embedding,
            limit=MAX_NUM_SIMILAR_TICKETS_FROM_RAG,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )

        self._extract_similar_tickets_from_rag_result(similar_tickets_from_rag, jira_issue.key)

        # Generate initial analysis
        if status_placeholder:
            status_placeholder.markdown("✨ **Finalizing the response...**")

        return self._generate_analysis(logs_analysis)

    def _extract_similar_tickets_from_rag_result(self, rag_result: Any, current_issue_key: str) -> str:
        """
        Process and store similar tickets (exclude the current ticket itself).
        rerank the rag results and returns the NUM_MOST_RELEVANT_TICKETS_TO_RETURN most relevant tickets
        """
        similar_tickets = []
        for obj in rag_result.objects:
            ticket_key = obj.properties.get("issue_key", "")
            # Skip if this is the same ticket we're analyzing
            if ticket_key == current_issue_key:
                continue

            similar_tickets.append({
                "key": ticket_key,
                "summary": obj.properties.get("summary", ""),
                "status": obj.properties.get("status", ""),
                "resolution": obj.properties.get("resolution", "Unresolved"),
                "content": obj.properties.get("clean_description", ""),
                "title": obj.properties.get("title", ""),
                "score": 1 - obj.metadata.distance if obj.metadata.distance else 0.0,
                "issue_type": obj.properties.get("issue_type", ""),
                "labels": obj.properties.get("labels", []),
            })

        st.session_state.similar_tickets = similar_tickets

        return self._rerank_similar_tickets()

    @traceable
    def _rerank_similar_tickets(self) -> str:
        """Rerank similar tickets using LLM for better relevance."""
        # TODO: Add reranking logic

        # current_ticket = st.session_state.current_ticket
        # similar_tickets = st.session_state.similar_tickets
        #
        # if not similar_tickets:
        #     return ""
        #
        # rerank_system_prompt = self._create_rerank_system_prompt()
        # rerank_input_prompt = self._create_rerank_input_prompt(current_ticket, similar_tickets)
        #
        # llm = ChatOllama(model=LLM_MODEL_NAME,
        #                  base_url="http://localhost:11434",
        #                  verbose=True,
        #                  temperature=0.1)
        #
        # messages = [
        #     SystemMessage(content=rerank_system_prompt),
        #     HumanMessage(content=rerank_input_prompt)
        # ]
        #
        # x = llm.invoke(messages).content.strip()
        # print("Rerank LLM output:", x)

        return ""

    @staticmethod
    def _create_rerank_system_prompt() -> str:
        rerank_system_prompt = """You are a JSON-only response bot.
    
        OUTPUT: Return ONLY a valid JSON array. No text before. No text after. No explanations.
        
        Example:
        [{"key": "KAN-45", "summary": "...", "labels": [], "status": "Done", "resolution": "Fixed", "relevance_reason": "..."}]
        
        TASK: Rank similar tickets by relevance to the current ticket. Return top 5 only.
        
        RULES:
        - Compare: summary, description, labels, issue type, status
        - Include: key, summary, labels, status, resolution, relevance_reason
        - Output ONLY the JSON array"""

        return rerank_system_prompt

    @staticmethod
    def _create_rerank_input_prompt(current_ticket: Dict, similar_tickets: List) -> str:
        rerank_input_prompt = f"""
        Current ticket:
        {current_ticket["summary"] + "\n" + current_ticket["description"]}
        
        List of similar tickets to rank:
        {json.dumps(similar_tickets, indent=2)}

        RESPOND WITH ONLY A JSON ARRAY OF THE TOP 5 MOST RELEVANT TICKETS. NO OTHER TEXT."""

        return rerank_input_prompt

    @traceable
    def _generate_analysis(self, logs_analysis: List[LogAnalysisOutput]) -> str:
        """Generate final ticket analysis with LLM using similar tickets."""
        final_analysis_system_prompt = self._create_final_analysis_system_prompt()
        final_analysis_input_prompt = self._create_final_analysis_input_prompt()
        llm = ChatOllama(model=LLM_MODEL_NAME, base_url="http://localhost:11434", temperature=0.1)
        structured_llm = llm.with_structured_output(FinalAnalysisOutput)
        messages = [
            SystemMessage(content=final_analysis_system_prompt),
            HumanMessage(content=final_analysis_input_prompt)
        ]
        final_analysis = structured_llm.invoke(messages)
        if isinstance(final_analysis, dict):
            final_analysis = FinalAnalysisOutput(**final_analysis)

        final_analysis_text = self._parse_final_analysis_output(final_analysis, logs_analysis)

        return final_analysis_text

    @staticmethod
    def _create_final_analysis_system_prompt() -> str:
        final_analysis_system_prompt = """
        You are an expert Jira analyst assisting software engineers.

        Your task is to analyze a Jira ticket together with its top 5 most similar past tickets and produce a 
        structured, factual analysis.
        
        Rules:
        - Base all conclusions ONLY on the information provided.
        - Do NOT hallucinate missing details.
        - If a root cause or solution is not clearly supported, state "unknown".
        - Be concise, technical, and professional.
        - Do not invent Jira ticket keys, statuses, or links.
        - Use Markdown formatting exactly as requested.
        - When explaining similarity, be specific (shared error message, component, symptom, environment, etc.).
        
        Output format must strictly follow the sections and headings provided.
        """

        return final_analysis_system_prompt

    @staticmethod
    def _create_final_analysis_input_prompt() -> str:
        # Format similar tickets with explicit details for comparison
        similar_tickets_details = []
        for idx, ticket in enumerate(st.session_state.similar_tickets, 1):
            similar_tickets_details.append(f"""
        Ticket {idx}:
        - Key: {ticket['key']}
        - Title: {ticket['title']}
        - Summary: {ticket['summary'][:2400]}
        - Status: {ticket['status']}
        - Resolution: {ticket.get('resolution', 'Unresolved')}
        - Labels: {ticket.get('labels', [])}
        - Issue Type: {ticket.get('issue_type', 'Unknown')}
        """)
        similar_tickets_formatted = "\n".join(similar_tickets_details).strip() or "NONE"
        current_ticket = st.session_state.current_ticket
        similat_tickets_num = 0 if similar_tickets_formatted == "NONE" else len(st.session_state.similar_tickets)

        final_analysis_input_prompt = f"""
        Current Jira ticket information:
        - CURRENT TICKET: {current_ticket['key']}
        - Summary: {current_ticket['summary'][:2400]}...
        - Description: {current_ticket['description'][:1600]}...
        - Status: {current_ticket['status']} 
        - Priority: {current_ticket['priority']}
        - Labels: {current_ticket['labels']}
        
        Top 5 most similar Jira tickets (already reranked by relevance):
        {similar_tickets_formatted}
        
        HARD RULES (must follow):
        - You MUST return `similar_tickets` with EXACTLY {similat_tickets_num} items.
        - You MUST include EVERY provided similar ticket key exactly once.
        - Do NOT drop items. Do NOT merge items. Do NOT invent additional items.
        - If the "Top similar Jira tickets" section is "NONE":
          - Return an empty list for similar_tickets: []
          - Do NOT mention any other Jira ticket keys in any section
          - Any similarity_reason/solutions based on similar tickets must be "unknown"
        
        IMPORTANT: For each similar ticket, provide a SPECIFIC similarity_reason explaining 
        what EXACTLY makes it similar (e.g., "Same API endpoint failure", "Identical NPE in UserService.java", 
        "Both involve timeout on database connection"). Do NOT use generic phrases like "shared error patterns".
        If no clear similarity exists, say "Low similarity - retrieved by vector search only".
        
        Jira base URL:
        {JIRA_BASE_URL}
        
        """

        return final_analysis_input_prompt

    @staticmethod
    def _parse_final_analysis_output(final_analysis: FinalAnalysisOutput,
                                     logs_analysis: List[LogAnalysisOutput]) -> str:
        """Format the final analysis output into a readable string."""

        def sanitize_text(text: str) -> str:
            """Remove unwanted code fences and clean up text."""
            if not text:
                return ""
            text = text.replace("```python", "").replace("```", "")
            return text.strip()

        ticket_summary = sanitize_text(final_analysis.ticket_summary)
        issues_formatted = "\n".join(f"    - {sanitize_text(issue)}" for issue in final_analysis.key_issues)
        root_causes_formatted = "\n".join(f"    - {sanitize_text(rc)}" for rc in final_analysis.root_causes)

        error_lines = []
        for log_analysis in logs_analysis:
            for error in log_analysis.errors:
                error_text = sanitize_text(error.error_lines[:500])
                error_block = f"    - Log File: {log_analysis.log_filename}"
                if error.source_code_filename:
                    error_block += f"  \n      File in code: {error.source_code_filename}"
                error_block += f"  \n      Error: `{error_text}`"
                error_block += f"  \n      Context: {sanitize_text(error.context)}"
                error_lines.append(error_block)
        errors_in_logs_formatted = "\n\n".join(error_lines) if error_lines else "    - No errors found in logs."

        similar_ticket_lines = []
        for similar_ticket in final_analysis.similar_tickets:
            ticket_key = similar_ticket.key
            jira_url = f"{JIRA_BASE_URL}/browse/{ticket_key}"
            similar_ticket_lines.append(
                f"* [{ticket_key}]({jira_url}): {sanitize_text(similar_ticket.title)}\n"
                f"   - Similarity Reason: {sanitize_text(similar_ticket.similarity_reason)}\n"
                f"   - Status: {similar_ticket.status}"
            )
        similar_tickets_formatted = "\n".join(
            similar_ticket_lines) if similar_ticket_lines else "No similar tickets found."

        suggested_solutions_formatted = "\n".join(f"* {sanitize_text(s)}" for s in final_analysis.suggested_solutions)
        important_notes_formatted = "\n".join(f"* {sanitize_text(n)}" for n in final_analysis.important_notes)

        # NO leading whitespace - start each line at column 0
        analysis_text = (
            f"## 📋 Ticket Summary\n"
            f"{ticket_summary}\n\n"
            f"## 🔍 Key Issues & Root Causes\n"
            f"* Main issues:\n"
            f"{issues_formatted}\n"
            f"* Likely root causes:\n"
            f"{root_causes_formatted}\n"
            f"* Errors found in logs:\n"
            f"{errors_in_logs_formatted}\n\n"
            f"## 🎯 Top Similar Tickets\n"
            f"{similar_tickets_formatted}\n\n"
            f"## 💡 Suggested Solutions\n"
            f"{suggested_solutions_formatted}\n\n"
            f"## ⚠️ Important Notes\n"
            f"{important_notes_formatted}"
        )

        return analysis_text

    @traceable
    def _classify_intent(self, user_msg: str) -> IntentClassification:
        """Classify user intent into one of: follow_up_on_current_ticket, analyze_new_ticket, unrelated_chat"""
        current_ticket_key = st.session_state.current_ticket['key'] if st.session_state.current_ticket else None
        user_msg_lower = user_msg.lower()
        potential_keys = extract_jira_keys_from_text(user_msg_lower)

        # ------------------------
        # No current ticket
        # ------------------------
        if not current_ticket_key:
            if potential_keys:
                if len(potential_keys) == 1:
                    return IntentClassification.ANALYZE_NEW_TICKET
                else:
                    return IntentClassification.MORE_THAN_ONE_KEY
            return IntentClassification.UNRELATED_CHAT

        # ------------------------
        # Current ticket exists but no key found in user input
        # ------------------------
        if not potential_keys:
            # Ambiguous → ask LLM
            llm_result = self._classify_intent_with_llm(user_msg)

            if llm_result == IntentClassification.UNRELATED_CHAT.value:
                return IntentClassification.UNRELATED_CHAT
            else:
                return IntentClassification.FOLLOW_UP_ON_CURRENT_TICKET

        # ------------------------
        # Current ticket exists and key found in user input
        # ------------------------
        if len(potential_keys) == 1:
            if potential_keys[0] == current_ticket_key:
                return IntentClassification.FOLLOW_UP_ON_CURRENT_TICKET
            similar_tickets_keys = {t["key"] for t in st.session_state.similar_tickets}
            if potential_keys[0] in similar_tickets_keys:
                return IntentClassification.FOLLOW_UP_ON_CURRENT_TICKET
            return IntentClassification.ANALYZE_NEW_TICKET
        else:
            return IntentClassification.MORE_THAN_ONE_KEY

    @traceable
    def _classify_intent_with_llm(self, user_msg: str) -> str:
        intent_system_prompt = self._create_intent_classification_system_prompt()
        intent_input_prompt = self._create_intent_input_prompt(user_msg)
        structured_llm = self.chat.with_structured_output(IntentOutput)
        messages = [
            SystemMessage(content=intent_system_prompt),
            HumanMessage(content=intent_input_prompt)
        ]

        return structured_llm.invoke(messages).intent.value

    @staticmethod
    def _create_intent_classification_system_prompt() -> str:
        intent_system_prompt = """You are an intent classification assistant for a Jira chatbot.
        Classify user messages into one of three intents:
        1. follow_up_on_current_ticket - User is asking about the current ticket being discussed.
        2. analyze_new_ticket - User is providing a new Jira ticket key or URL to analyze.
        3. unrelated_chat - User is engaging in unrelated conversation.

        Use the following rules:
        - If the message contains a new Jira ticket key (different from current), classify as analyze_new_ticket.
        - If the message references "this ticket" or "the ticket" and there is a current ticket, classify as 
        follow_up_on_current_ticket.
        - If neither of the above, classify as unrelated_chat.

        Always respond with ONLY the intent name."""

        return intent_system_prompt

    @staticmethod
    def _create_intent_input_prompt(user_msg: str) -> str:
        intent_input_prompt = f"""
        Current ticket key: {st.session_state.current_ticket['key'] or "None"}
        Current ticket summary: {st.session_state.current_ticket['summary'][:200]}
        User message: "{user_msg}"

        Classify the intent as one of:
        1. follow_up_on_current_ticket
        2. analyze_new_ticket
        3. unrelated_chat

        Respond with ONLY the intent name."""

        return intent_input_prompt

    @traceable
    def _continue_conversation(self, user_message: str, intent: IntentClassification) -> str:
        """Handle follow-up conversation with context awareness."""
        response = ""

        if intent == IntentClassification.FOLLOW_UP_ON_CURRENT_TICKET:
            follow_up_system_prompt = self._create_follow_up_system_prompt()
            follow_up_input_prompt = self._create_follow_up_input_prompt(user_message)
            messages = [
                SystemMessage(content=follow_up_system_prompt),
                *self.history[1:],  # Exclude original system prompt
                HumanMessage(content=follow_up_input_prompt)
            ]
            response = self.chat.invoke(messages).content

        if intent == IntentClassification.UNRELATED_CHAT:
            messages = [
                *self.history,
                HumanMessage(content=user_message)
            ]
            response = self.chat.invoke(messages).content

        if intent == IntentClassification.MORE_THAN_ONE_KEY:
            issue_keys = extract_jira_keys_from_text(user_message)
            tickets_formatted = "\n".join(
                f"- {key} ({ATLASSIAN_INSTANCE_URL}/browse/{key})"
                for key in issue_keys
            )

            response = f"""I see that you mentioned multiple tickets.

            Which one would you like me to analyze?

            {tickets_formatted}

            👉 Please reply with a single ticket key.
            """

        return response

    @staticmethod
    def _create_follow_up_system_prompt() -> str:
        follow_up_system_prompt = """You are a Jira support assistant.
        You are continuing an existing discussion about the SAME Jira ticket.
        Use the provided context about the current ticket and recent conversation history to answer user questions.
        
        Rules:
        - Do NOT re-summarize the ticket unless explicitly asked
        - Focus only on the user's latest question
        - Use prior context and similar tickets if helpful
        - Be concise and actionable
        - Do not hallucinate. If something is unknown, say "unknown".
        """

        return follow_up_system_prompt

    @staticmethod
    def _create_follow_up_input_prompt(user_msg: str) -> str:
        ticket = st.session_state.current_ticket

        follow_up_input_prompt = f"""
        Current ticket info: 
        Ticket Key: {ticket['key']}
        Summary: {ticket['summary'][:2400]}
        Description: {ticket['description'][:500]}
        Status: {ticket['status']}
        Priority: {ticket['priority']}
        Created: {ticket.get('created', 'N/A')}
        
        User message: "{user_msg}"

        Using the context above, provide a concise and accurate response to the user's question about the current ticket.
        """

        return follow_up_input_prompt


if __name__ == "__main__":
    jira_chatbot = JiraChatBot()
    try:
        jira_chatbot.run()
    finally:
        jira_chatbot.close()
