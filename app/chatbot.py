# app/streamlit_app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from utils.jira_client import JiraIssue, JiraClient
from urllib.parse import urlparse
import re

load_dotenv()
CHROMA_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY
)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="jira_tickets",
)

llm = ChatOpenAI(
    model="gpt-4o-mini", streaming=False, openai_api_key=OPENAI_API_KEY, temperature=0.0
)

st.set_page_config(page_title="Jira Similarity Assistant", layout="wide")
st.title("Jira Similarity Assistant (MVP)")


def extract_key_from_url(url_or_key: str) -> str:
    # Try to extract JIRA key like PROJ-123 from url or return input if it looks like key
    url_or_key = url_or_key.strip()
    # If looks like key
    if re.match(r"^[A-Z]+-\d+$", url_or_key):
        return url_or_key
    # Try parse URL
    try:
        parsed = urlparse(url_or_key)
        if parsed.path:
            m = re.search(r"/browse/([A-Z]+-\d+)", parsed.path)
            if m:
                return m.group(1)
    except Exception:
        pass
    return url_or_key


def build_prompt_based_on_similar_tickets(ticket, similar_list):
    sim_text = ""
    for i, s in enumerate(similar_list[:3], start=1):  # Limited to 3 as requested
        md = s["metadata"]
        sim_text += f"{i}. {md.get('ticket_key')} | status={md.get('status')} | resolution={md.get('resolution')} | summary={md.get('summary')}\n   snippet: {s['page_content'][:300]}...\n\n"

    # Get Jira base URL for links
    jira_base_url = (
        os.environ.get("ATLASSIAN_INSTANCE_URL", "").rstrip("/rest/api/3").rstrip("/")
    )

    prompt = f"""
You are a senior engineer helping to triage a Jira ticket.

Ticket:
Summary: {ticket["summary"]}
Description: {ticket.get("description", "")}

Top similar ticket chunks (metadata and snippet):
{sim_text}

Return your response in this EXACT format:

## 📋 Summary
[Write a one-paragraph summary of the incoming ticket in max 60 words]

## 🔍 Root Cause Analysis
- [Bullet point explaining likely root cause 1]
- [Bullet point explaining likely root cause 2]
- [Add more bullet points as needed]

## 🔗 Similar Tickets (Top 3)
1. **[TICKET-KEY]({jira_base_url}/browse/TICKET-KEY)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])
2. **[TICKET-KEY]({jira_base_url}/browse/TICKET-KEY)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])  
3. **[TICKET-KEY]({jira_base_url}/browse/TICKET-KEY)** - [Explain why it's similar] (Status: [status], Resolution: [resolution or "Unresolved"])

## 💡 Suggested Solution
[Provide a concrete suggested fix. If appropriate include a small code snippet using ```code blocks```. Prioritize solutions that appeared in tickets with resolution not null (i.e., solved).]

## 🎯 Confidence Level
[low/medium/high]

Use the ticket information from the similar tickets above. Replace [TICKET-KEY] with actual ticket keys from the similar tickets data and create proper markdown links.
"""
    return prompt


st.sidebar.markdown("### Usage")
st.sidebar.markdown("Paste a Jira ticket URL or key, then press Analyze.")

# Initialize session state for analysis completion
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False

# Only show form if analysis hasn't been completed
if not st.session_state.analysis_completed:
    with st.form("ticket_form"):
        url_or_key = st.text_input(
            "Jira ticket URL or key (e.g., PROJ-123 or https://.../browse/PROJ-123)"
        )
        submitted = st.form_submit_button("Analyze")
else:
    st.info("Analysis completed. Refresh the page to analyze another ticket.")
    submitted = False
    url_or_key = None
if submitted and url_or_key:
    key = extract_key_from_url(url_or_key)
    try:
        issue: JiraIssue = JiraClient().fetch_issue_by_key(key)
    except Exception as e:
        st.error(f"Failed to fetch Jira issue: {e}")
        st.stop()

    ticket = {
        "key": issue.key,
        "summary": issue.summary,
        "description": issue.description,
    }

    # Build query text and do vector search
    # Handle case where description might be a dict (rich text) or None
    description_text = ""
    if ticket["description"]:
        if isinstance(ticket["description"], dict):
            # If it's a dict, try to extract text content
            description_text = str(ticket["description"])
        else:
            description_text = ticket["description"]

    query_text = ticket["summary"] + "\n\n" + description_text
    # search returns Document objects; use vectorstore.similarity_search_with_score for scores
    results = vectorstore.similarity_search_with_score(query_text, k=10)

    # Convert results to simpler dicts and prioritize solved ones
    # results returns (Document, score)
    similar_issues = []
    for doc, score in results:
        similar_issues.append(
            {"page_content": doc.page_content, "score": score, "metadata": doc.metadata}
        )

    # Prioritize solved/resolved tickets first
    similar_issues_sorted = sorted(
        similar_issues,
        key=lambda x: (0 if x["metadata"].get("resolution") else 1, x["score"]),
    )

    prompt = build_prompt_based_on_similar_tickets(ticket, similar_issues_sorted[:10])
    # call LLM
    with st.spinner("Generating answer from LLM..."):
        resp = llm.invoke(prompt)  # Use invoke instead of generate with dict format
        # extract text
        text = resp.content

    # Display structured analysis
    st.subheader(f"Analysis for {ticket['key']}")

    # Add Jira ticket link
    jira_base_url = (
        os.environ.get("ATLASSIAN_INSTANCE_URL", "").rstrip("/rest/api/3").rstrip("/")
    )
    if jira_base_url:
        ticket_url = f"{jira_base_url}/browse/{ticket['key']}"
        st.markdown(f"🔗 [View ticket in Jira]({ticket_url})")

    # Display the LLM's structured response (now includes pre-formatted links)
    st.markdown(text)

    # Mark analysis as completed to disable further input
    st.session_state.analysis_completed = True
