# app/chatbot_agent.py
import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Custom imports
from utils.jira_client import JiraIssue, JiraClient

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


class JiraAgent:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            streaming=False,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0,
        )

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Initialize prompt
        self.prompt = self._create_prompt()

        # Initialize tools
        self.tools = self._create_tools()

        # Initialize agent
        self.agent = self._create_agent()

    def _create_agent(self):
        """Create the conversational agent"""
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="force",
            agent_kwargs={
                "prefix": self.prompt,
                "format_instructions": """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [get_ticket_details, search_similar_tickets]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
- Only put brief thoughts in the 'Thought:' section
- Put the complete structured analysis in the 'Final Answer:' section
- Never put markdown formatting in the 'Thought:' section""",
            },
        )
        return agent

    def chat(self, message: str) -> str:
        """Send a message to the agent and get a response"""
        try:
            # For ticket analysis, use the agent with verbose logging
            if self._is_ticket_request(message):
                # Try agent first for the detailed logs you want
                try:
                    response = self.agent.invoke({"input": message})
                    return response.get("output", "No response received")
                except Exception as agent_error:
                    print(f"⚠️ Agent failed: {agent_error}")
                    print("🔄 Falling back to direct analysis...")
                    return self._analyze_ticket(message)
            else:
                # Use regular agent for other requests
                response = self.agent.invoke({"input": message})
                return response.get("output", "No response received")
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def _is_ticket_request(self, message: str) -> bool:
        """Check if the message is requesting ticket analysis"""
        ticket_patterns = ["KAN-", "analyze", "ticket", "jira", "atlassian.net"]
        return any(pattern.lower() in message.lower() for pattern in ticket_patterns)

    def _analyze_ticket(self, message: str) -> str:
        """Analyze a ticket using both tools in sequence"""
        try:
            # Extract ticket key
            ticket_key = self._extract_ticket_key(message)
            if not ticket_key:
                return "Could not identify ticket key. Please provide a valid Jira ticket key (e.g., KAN-9)."

            # Step 1: Get ticket details
            ticket_details = self.tools[1].func(
                ticket_key
            )  # get_ticket_details is second tool

            # Step 2: Search similar tickets using summary
            summary_match = (
                ticket_details.split("Summary: ")[1].split("\n")[0]
                if "Summary: " in ticket_details
                else ticket_key
            )

            similar_tickets = self.tools[0].func(
                summary_match
            )  # search_similar_tickets is first tool

            # Step 3: Generate structured response
            result = self._generate_structured_response(ticket_details, similar_tickets)

            return result

        except Exception as e:
            return f"Error analyzing ticket: {str(e)}"

    def _extract_ticket_key(self, message: str) -> str:
        """Extract ticket key from message"""
        import re

        # Look for patterns like KAN-9, ABC-123, etc.
        match = re.search(r"([A-Z]+-\d+)", message.upper())
        return match.group(1) if match else None

    def _generate_structured_response(
        self, ticket_details: str, similar_tickets: str
    ) -> str:
        """Generate structured response using LLM"""
        prompt = f"""
Based on the following ticket details and similar tickets, provide a structured analysis:

TICKET DETAILS:
{ticket_details}

SIMILAR TICKETS:
{similar_tickets}

Please provide your response in this EXACT format:

## 📋 Summary
[Write a one-paragraph summary of the ticket in max 60 words]

## 🔍 Root Cause Analysis
- [Bullet point explaining likely root cause 1]
- [Bullet point explaining likely root cause 2]
- [Add more bullet points as needed]

## 🔗 Similar Tickets (Top 3)
1. **[TICKET-KEY](https://royho10.atlassian.net/browse/TICKET-KEY)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])
2. **[TICKET-KEY](https://royho10.atlassian.net/browse/TICKET-KEY)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])
3. **[TICKET-KEY](https://royho10.atlassian.net/browse/TICKET-KEY)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])

## 💡 Suggested Solution
[Provide a concrete suggested fix. Include code snippets if appropriate. Prioritize solutions from resolved tickets.]

## 🎯 Confidence Level
[low/medium/high]
"""

        response = self.llm.invoke(prompt)
        return response.content

    def _create_prompt(self):
        return """You are a senior engineer and Jira AI Assistant. 

CRITICAL: When analyzing a Jira ticket (like KAN-5, KAN-9, etc.), you MUST follow ALL these steps in order:

MANDATORY WORKFLOW FOR TICKET ANALYSIS:
1. FIRST: Use get_ticket_details tool to get ticket information
2. SECOND: Use search_similar_tickets tool with the ticket summary as query
3. THIRD: Provide analysis in the EXACT structured format below

NEVER skip step 2 (search_similar_tickets) - it is mandatory for every ticket analysis.

Your Final Answer MUST use this EXACT structured format:

## 📋 Summary
[Write a one-paragraph summary of the ticket in max 60 words]

## 🔍 Root Cause Analysis
- [Bullet point explaining likely root cause 1]
- [Bullet point explaining likely root cause 2]
- [Add more bullet points as needed]

## 🔗 Similar Tickets (Top 3)
1. **[TICKET-KEY](jira-url)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])
2. **[TICKET-KEY](jira-url)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])
3. **[TICKET-KEY](jira-url)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])

## 💡 Suggested Solution
[Provide a concrete suggested fix. Include code snippets if appropriate. Prioritize solutions from resolved tickets.]

## 🎯 Confidence Level
[low/medium/high]

For general questions, provide helpful and clear responses. Always be professional and helpful."""

    def _create_tools(self):
        """Create the tools for the agent"""

        # Initialize vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="jira_tickets",
        )

        # Tool 1: Search similar tickets
        def search_similar_tickets(query: str) -> str:
            """Search for similar Jira tickets based on a query"""
            try:
                results = vectorstore.similarity_search(query, k=5)
                if not results:
                    return "No similar tickets found."

                output = "Similar tickets found:\n"
                for i, doc in enumerate(results, 1):
                    output += f"{i}. {doc.page_content[:200]}...\n"
                    output += f"   Metadata: {doc.metadata}\n\n"
                return output
            except Exception as e:
                return f"Error searching tickets: {str(e)}"

        # Tool 2: Get ticket details
        def get_ticket_details(ticket_key: str) -> str:
            """Get detailed information about a specific Jira ticket from its key"""
            try:
                # Clean the ticket key - remove any extra whitespace, newlines, or markdown
                clean_key = (
                    ticket_key.strip()
                    .replace("\n", "")
                    .replace("`", "")
                    .replace("```", "")
                )

                jira_client = JiraClient()
                issue: JiraIssue = jira_client.fetch_issue_by_key(clean_key)

                if not issue:
                    return "Could not retrieve ticket details."

                return f"""Ticket Details:
Key: {issue.key}
Summary: {issue.summary}
Description: {str(issue.description)[:500] if issue.description else "No description"}...
Status: {issue.status.name}
Labels: {", ".join(issue.labels) if issue.labels else "None"}
Resolution: {issue.resolution.name if issue.resolution else "Unresolved"}
Comments: {len(issue.comments)} comments available"""
            except Exception as e:
                return f"Error getting ticket details: {str(e)}"

        # Create LangChain tools
        tools = [
            Tool(
                name="search_similar_tickets",
                description="MANDATORY for ticket analysis: Search for similar Jira tickets in a vector db based on a query. ALWAYS use this after get_ticket_details for any ticket analysis.",
                func=search_similar_tickets,
            ),
            Tool(
                name="get_ticket_details",
                description="Get detailed information about a specific Jira ticket using its key. ALWAYS use search_similar_tickets after this tool.",
                func=get_ticket_details,
            ),
        ]

        return tools


# Basic Streamlit setup
st.set_page_config(page_title="Jira AI Agent", layout="wide")
st.title("🤖 Jira AI Agent")


# Initialize the agent (with caching to avoid recreating on each run)
@st.cache_resource
def get_agent():
    return JiraAgent()


# Get the agent instance
agent = get_agent()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Jira tickets or paste a ticket URL..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with helpful information
with st.sidebar:
    st.header("💡 How to use")
    st.markdown("""
    **Try these commands:**
    - Paste a Jira ticket URL for analysis
    - Ask: "Find similar tickets about login issues"
    - Ask: "What are common database problems?"
    - Ask: "Analyze ticket ABC-123"
    
    **Features:**
    - 🔍 Smart ticket search using AI
    - 📊 Structured analysis reports
    - 💾 Conversation memory
    - 🎯 Confidence scoring
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
