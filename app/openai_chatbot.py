# app/openai_chatbot.py
import json
import logging
import os
import requests
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_LLM_DEPLOYMENT,
    AZURE_OPENAI_TEMPERATURE,
)
from core.ticket_analyzer import TicketAnalysis, TicketAnalyzer
from utils.jira_client import extract_jira_keys_from_text, ATLASSIAN_INSTANCE_URL
from utils.llm_logger import log_llm_call, log_run_summary


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('openai_chatbot_debug.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
JIRA_BASE_URL = os.environ.get("ATLASSIAN_INSTANCE_URL", "").replace("/rest/api/3", "").rstrip("/")

# Chatbot-specific constants
MAX_MSG_HISTORY = 10


class IntentClassification(Enum):
    FOLLOW_UP_ON_CURRENT_TICKET = "follow_up_on_current_ticket"
    ANALYZE_NEW_TICKET = "analyze_new_ticket"
    UNRELATED_CHAT = "unrelated_chat"
    MORE_THAN_ONE_KEY = "more_than_one_key"


class IntentOutput(BaseModel):
    """Classification of user intent."""
    intent: IntentClassification = Field(description="The classified intent of the user message")


class OpenAIJiraChatBot:
    """
    Streamlit chatbot using Azure OpenAI for LLM and embeddings.
    Uses JiraCollectionOpenAI Weaviate collection for vector similarity search.
    """

    def __init__(
            self,
            deployment: str = AZURE_OPENAI_LLM_DEPLOYMENT,
            max_history: int = MAX_MSG_HISTORY,
    ):
        self.chat = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=AZURE_OPENAI_TEMPERATURE,
        )
        self.max_history = max_history
        self.system_prompt = """You are a Jira ticket analysis assistant that helps engineers understand and troubleshoot issues.

Your ONLY capabilities:
1. Analyze Jira tickets when users provide a ticket key (e.g., GC-123)
2. Find similar past tickets using RAG-based vector search
3. Identify patterns, errors, and potential solutions based on similar tickets
4. Answer follow-up questions about analyzed tickets

What you CANNOT do:
- Create new Jira tickets
- Update ticket status, priority, assignee, or any fields
- Add comments, labels, or modify tickets in any way
- Access Jira APIs for write operations

When greeting users or responding to unrelated chat:
- Briefly introduce yourself
- Ask them to provide a Jira ticket key to analyze
- Do NOT list capabilities you don't have

Be concise, factual, and technical. If something is unknown, say "unknown".
"""
        self.history = [SystemMessage(content=self.system_prompt)]

        # All ticket analysis is delegated to the shared, stateless analyzer.
        self.analyzer = TicketAnalyzer()

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
        st.set_page_config(page_title="Jira Ticket Assistant (OpenAI)", layout="wide")
        st.title("🎫 Jira Ticket Assistant (OpenAI)")
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
            st.sidebar.markdown(f"**{ticket['key']}**: {ticket.get('title', '')}")
            st.sidebar.markdown(f"Status: `{ticket['status']}`")
            if JIRA_BASE_URL:
                st.sidebar.markdown(f"[View in Jira →]({JIRA_BASE_URL}/browse/{ticket['key']})")

    def close(self) -> None:
        """Close connections properly."""
        if self.analyzer:
            self.analyzer.close()

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
        """Analyze a ticket via the shared analyzer and render it for Streamlit."""
        on_progress = status_placeholder.markdown if status_placeholder else None

        try:
            analysis = self.analyzer.analyze(jira_key, on_progress=on_progress)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return f"❌ Jira issue with key '{jira_key}' not found. Try another key."
            else:
                return f"❌ Failed to fetch Jira issue: {e}"
        except Exception as e:
            return f"❌ Failed to fetch Jira issue: {e}"

        # Keep the conversational context the follow-up prompts rely on.
        st.session_state.current_ticket = {
            "key": analysis.ticket_key,
            "title": analysis.ticket_title,
            "summary": analysis.processed_summary,
            "description": analysis.description,
            "status": analysis.ticket_status,
            "priority": analysis.ticket_priority,
            "created": analysis.ticket_created,
            "labels": analysis.ticket_labels,
        }
        st.session_state.similar_tickets = analysis.similar_tickets

        analysis_text = self._parse_final_analysis_output(analysis)
        log_run_summary()
        return analysis_text

    @staticmethod
    def _parse_final_analysis_output(analysis: TicketAnalysis) -> str:
        """Format a TicketAnalysis into markdown for the Streamlit renderer."""

        def sanitize_text(text: str) -> str:
            """Remove unwanted code fences and escape markdown/LaTeX special chars."""
            if not text:
                return ""
            text = text.replace("```python", "").replace("```", "")
            # Escape characters that Streamlit markdown interprets unexpectedly
            for ch in ("$", "<", ">", "{", "}", "~"):
                text = text.replace(ch, f"\\{ch}")
            return text.strip()

        ticket_summary = sanitize_text(analysis.ticket_summary)
        issues_formatted = "\n".join(f"    - {sanitize_text(issue)}" for issue in analysis.key_issues)
        root_causes_formatted = "\n".join(f"    - {sanitize_text(rc)}" for rc in analysis.root_causes)

        error_lines = []
        for highlight in analysis.error_log_highlights:
            exception_text = sanitize_text(highlight.exception_line) if highlight.exception_line else None

            # Determine whether to show the error_lines code block
            raw_error = highlight.error_lines.strip() if highlight.error_lines else ""
            show_code_block = bool(raw_error)
            if raw_error.lower() == "undefined":
                show_code_block = False
            elif exception_text and (
                raw_error.startswith("Traceback")
                or raw_error.startswith('File "')
            ):
                # Redundant truncated traceback when we already have the exception
                show_code_block = False

            # Build metadata as indented sub-items under "Errors found in logs:"
            error_block = f"    - **Log File:** {highlight.log_filename}"
            if highlight.source_code_filename:
                error_block += f"\n      **File in code:** `{sanitize_text(highlight.source_code_filename)}`"
            error_block += f"\n      **Context:** {highlight.context}"
            if exception_text:
                error_block += f"\n      **Exception:** `{exception_text}`"
            if show_code_block:
                error_text = sanitize_text(raw_error[:500])
                # Code block at column 0 to prevent Streamlit bleeding
                error_block += f"\n\n```\n{error_text}\n```"

            error_lines.append(error_block)
        errors_in_logs_formatted = "\n".join(error_lines) if error_lines else "    - No errors found in logs."

        similar_ticket_lines = []
        for similar_ticket in analysis.similar_tickets:
            ticket_key = similar_ticket.key
            jira_url = f"{JIRA_BASE_URL}/browse/{ticket_key}"
            similar_ticket_lines.append(
                f"* [{ticket_key}]({jira_url}): {sanitize_text(similar_ticket.title)}\n"
                f"   - Similarity Reason: {sanitize_text(similar_ticket.similarity_reason)}\n"
                f"   - Status: {similar_ticket.status}"
            )
        similar_tickets_formatted = "\n".join(
            similar_ticket_lines) if similar_ticket_lines else "No similar tickets found."

        def strip_list_marker(text: str) -> str:
            stripped = text.lstrip()
            for marker in ("- ", "* ", "• ", "◦ "):
                if stripped.startswith(marker):
                    return stripped[len(marker):]
            return stripped

        filtered_solutions = [
            strip_list_marker(s) for s in analysis.suggested_solutions
            if s.strip().lower() not in ("unknown", "")
        ]
        suggested_solutions_formatted = "\n".join(
            f"* {sanitize_text(s)}" for s in filtered_solutions
        ) if filtered_solutions else "No suggested solutions."
        important_notes_formatted = "\n".join(f"* {sanitize_text(n)}" for n in analysis.important_notes)

        # Ticket presentation header
        ticket_key = analysis.ticket_key
        ticket_title = analysis.ticket_title
        jira_url = f"{JIRA_BASE_URL}/browse/{ticket_key}"

        # NO leading whitespace - start each line at column 0
        analysis_text = (
            f"***[{ticket_key}]({jira_url}) — {ticket_title}**\n\n"
            f"---\n\n"
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
                    intent = IntentClassification.ANALYZE_NEW_TICKET
                else:
                    intent = IntentClassification.MORE_THAN_ONE_KEY
            else:
                intent = IntentClassification.UNRELATED_CHAT
            log_llm_call("intent_classification (rule-based)", "N/A",
                         f"user_msg={user_msg}, current_ticket=None, keys={potential_keys}",
                         intent.value, duration_ms=0)
            return intent

        # ------------------------
        # Current ticket exists but no key found in user input
        # ------------------------
        if not potential_keys:
            # Ambiguous -> ask LLM
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
                intent = IntentClassification.FOLLOW_UP_ON_CURRENT_TICKET
            elif potential_keys[0] in {t.key for t in st.session_state.similar_tickets}:
                intent = IntentClassification.FOLLOW_UP_ON_CURRENT_TICKET
            else:
                intent = IntentClassification.ANALYZE_NEW_TICKET
        else:
            intent = IntentClassification.MORE_THAN_ONE_KEY
        log_llm_call("intent_classification (rule-based)", "N/A",
                     f"user_msg={user_msg}, current_ticket={current_ticket_key}, keys={potential_keys}",
                     intent.value, duration_ms=0)
        return intent

    def _classify_intent_with_llm(self, user_msg: str) -> str:
        intent_system_prompt = self._create_intent_classification_system_prompt()
        intent_input_prompt = self._create_intent_input_prompt(user_msg)
        structured_llm = self.chat.with_structured_output(IntentOutput)
        messages = [
            SystemMessage(content=intent_system_prompt),
            HumanMessage(content=intent_input_prompt)
        ]

        start = time.time()
        result = structured_llm.invoke(messages)
        duration_ms = int((time.time() - start) * 1000)
        log_llm_call("intent_classification", AZURE_OPENAI_LLM_DEPLOYMENT, messages, result, duration_ms)
        return result.intent.value

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
            start = time.time()
            result = self.chat.invoke(messages)
            duration_ms = int((time.time() - start) * 1000)
            log_llm_call("follow_up_conversation", AZURE_OPENAI_LLM_DEPLOYMENT, messages, result.content, duration_ms)
            response = result.content

        if intent == IntentClassification.UNRELATED_CHAT:
            messages = [
                *self.history,
                HumanMessage(content=user_message)
            ]
            start = time.time()
            result = self.chat.invoke(messages)
            duration_ms = int((time.time() - start) * 1000)
            log_llm_call("unrelated_chat", AZURE_OPENAI_LLM_DEPLOYMENT, messages, result.content, duration_ms)
            response = result.content

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
    jira_chatbot = OpenAIJiraChatBot()
    try:
        jira_chatbot.run()
    finally:
        jira_chatbot.close()
