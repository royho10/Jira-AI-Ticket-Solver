"""Stateless Jira ticket Analysis.

Owns the full Analysis pipeline: fetch the ticket via the Jira service account,
process its attachments (VLM/logs), summarize, search Weaviate for similar
tickets, rerank them, and assemble a structured result.

No UI framework and no request state -- callers (the Streamlit app, the Remote
MCP Server) provide their own presentation layer.
"""
import logging
import time
from typing import Any, Callable, List, Optional

import weaviate.classes as wvc
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_LLM_DEPLOYMENT,
    AZURE_OPENAI_TEMPERATURE,
    JIRA_COLLECTION_NAME,
    MAX_EMBEDDINGS_INPUT_CHARS,
    MAX_SIMILAR_TICKETS_AFTER_RERANK,
    RERANK_SCORE_THRESHOLD,
)
from utils.jira_client import JiraClient, JiraIssue
from utils.llm_logger import log_llm_call
from utils.openai_jira_ticket_processing import LogAnalysisOutput, OpenAIJiraIssueLLMProcessor
from utils.weaviate_client import connect_to_weaviate

logger = logging.getLogger(__name__)

MAX_CANDIDATES_FROM_RAG = 20
# Half the embedding budget goes to the description, the rest to the summary.
MAX_EMBEDDINGS_DESCRIPTION_INPUT_CHARS = MAX_EMBEDDINGS_INPUT_CHARS / 2


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------
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


class RerankTicketScore(BaseModel):
    """Score for a single ticket from LLM reranking."""
    key: str = Field(description="Jira ticket key")
    relevance_score: int = Field(description="Relevance score 0-10")
    reason: str = Field(description="Brief reason for the score")


class RerankOutput(BaseModel):
    """Output from LLM reranking of candidate tickets."""
    scored_tickets: List[RerankTicketScore]


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------
class SimilarTicket(BaseModel):
    """A historical ticket retrieved by vector search and kept after reranking."""
    key: str
    title: str = ""
    summary: str = ""
    status: str = ""
    resolution: str = "Unresolved"
    issue_type: str = ""
    labels: List[str] = Field(default_factory=list)
    vector_score: float = 0.0
    relevance_score: Optional[int] = None
    relevance_reason: Optional[str] = None
    similarity_reason: Optional[str] = None


class ErrorLogHighlight(BaseModel):
    """A single error extracted from an attached log file."""
    log_filename: str
    context: str = ""
    error_lines: Optional[str] = None
    exception_line: Optional[str] = None
    source_code_filename: Optional[str] = None


class TicketAnalysis(BaseModel):
    """The complete structured result of analyzing one Jira ticket."""
    ticket_key: str
    ticket_title: str = ""
    ticket_status: Optional[str] = None
    ticket_priority: Optional[str] = None
    ticket_created: Optional[str] = None
    ticket_labels: List[str] = Field(default_factory=list)
    description: str = ""

    # Output of the attachment/comment processing pipeline (includes image insights)
    processed_summary: str = ""

    # Final LLM analysis
    ticket_summary: str = ""
    key_issues: List[str] = Field(default_factory=list)
    root_causes: List[str] = Field(default_factory=list)
    suggested_solutions: List[str] = Field(default_factory=list)
    important_notes: List[str] = Field(default_factory=list)

    error_log_highlights: List[ErrorLogHighlight] = Field(default_factory=list)
    similar_tickets: List[SimilarTicket] = Field(default_factory=list)


class _StatusCallbackAdapter:
    """Adapts a plain `on_progress(str)` callable to the `.markdown(str)` shape
    the processing pipeline expects, so the processor stays unmodified."""

    def __init__(self, on_progress: Callable[[str], None]):
        self._on_progress = on_progress

    def markdown(self, message: str) -> None:
        self._on_progress(message)


class TicketAnalyzer:
    """Runs the Analysis for a single ticket. Stateless across calls."""

    def __init__(
        self,
        jira_client: Optional[JiraClient] = None,
        processor: Optional[OpenAIJiraIssueLLMProcessor] = None,
        db_client: Any = None,
        embedder: Any = None,
        chat: Any = None,
        collection_name: str = JIRA_COLLECTION_NAME,
    ):
        self.jira_client = jira_client or JiraClient()
        self.processor = processor or OpenAIJiraIssueLLMProcessor(
            AZURE_OPENAI_LLM_DEPLOYMENT, AZURE_OPENAI_LLM_DEPLOYMENT
        )
        self.db_client = db_client or connect_to_weaviate()
        self.embedder = embedder or AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
        )
        self.chat = chat or AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_LLM_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=AZURE_OPENAI_TEMPERATURE,
        )
        self.collection_name = collection_name

    def close(self) -> None:
        if self.db_client:
            self.db_client.close()

    # ------------------------
    # Public API
    # ------------------------
    def analyze(
        self,
        ticket_key: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> TicketAnalysis:
        """Fetch, process and analyze a ticket, returning a structured result.

        Raises whatever the Jira client raises when the ticket cannot be fetched
        -- callers decide how to present that failure.
        """
        def progress(message: str) -> None:
            if on_progress:
                on_progress(message)

        progress("📥 **Analyzing ticket...**")
        jira_issue: JiraIssue = self.jira_client.fetch_issue_by_key(ticket_key)

        status_callback = _StatusCallbackAdapter(on_progress) if on_progress else None
        processed_summary, logs_analysis = self.processor.process_issue(jira_issue, status_callback)

        progress("🔍 **Finding similar tickets...**")
        candidates = self._find_similar_tickets(jira_issue, processed_summary)
        similar_tickets = self._rerank(jira_issue, processed_summary, candidates)

        progress("✨ **Finalizing the response...**")
        return self._build_analysis(jira_issue, processed_summary, logs_analysis, similar_tickets)

    # ------------------------
    # Pipeline steps
    # ------------------------
    def _build_embedding_query(self, jira_issue: JiraIssue, processed_summary: str) -> str:
        """Description-priority embedding text -- matches the indexer construction."""
        desc_text = (jira_issue.description or "")[:int(MAX_EMBEDDINGS_DESCRIPTION_INPUT_CHARS)]
        remaining = MAX_EMBEDDINGS_INPUT_CHARS - len(desc_text) - 2  # 2 for "\n\n"
        summary_part = processed_summary[:remaining]
        return desc_text + "\n\n" + summary_part

    def _find_similar_tickets(self, jira_issue: JiraIssue, processed_summary: str) -> List[SimilarTicket]:
        """Vector-search Weaviate, excluding the ticket being analyzed.

        A vector search failure degrades to an analysis without similar tickets
        rather than failing the whole request.
        """
        try:
            query_embedding = self.embedder.embed_documents(
                [self._build_embedding_query(jira_issue, processed_summary)]
            )[0]
            collection = self.db_client.collections.get(self.collection_name)
            rag_result = collection.query.near_vector(
                near_vector=query_embedding,
                limit=MAX_CANDIDATES_FROM_RAG,
                return_metadata=wvc.query.MetadataQuery(distance=True),
            )
        except Exception as e:
            logger.error(f"Similar ticket search failed, continuing without it: {e}")
            return []

        candidates = []
        for obj in rag_result.objects:
            ticket_key = obj.properties.get("issue_key", "")
            if ticket_key == jira_issue.key:
                continue
            distance = obj.metadata.distance
            candidates.append(SimilarTicket(
                key=ticket_key,
                title=obj.properties.get("title", ""),
                summary=obj.properties.get("summary", ""),
                status=obj.properties.get("status", ""),
                resolution=obj.properties.get("resolution") or "Unresolved",
                issue_type=obj.properties.get("issue_type", ""),
                labels=obj.properties.get("labels") or [],
                vector_score=1 - distance if distance else 0.0,
            ))
        return candidates

    def _rerank(
        self,
        jira_issue: JiraIssue,
        processed_summary: str,
        candidates: List[SimilarTicket],
    ) -> List[SimilarTicket]:
        """Score candidates with an LLM, drop the weak ones, keep the top N.

        A reranking failure degrades to the original vector-search order.
        """
        if not candidates:
            return []

        candidates_text = "\n".join(
            f"- Key: {c.key}, Title: {c.title}, Summary: {c.summary[:500]}"
            for c in candidates
        )
        rerank_prompt = f"""You are a Jira ticket similarity scorer. Given a current ticket and a list of candidate tickets,
score each candidate on a 0-10 relevance scale based on:
- Same error patterns or error messages
- Same component/service affected
- Same symptoms described
- Same root cause

Score guide: 0 = completely unrelated, 5 = moderately similar, 10 = nearly identical issue.

Current ticket:
- Key: {jira_issue.key}
- Summary: {processed_summary[:1000]}
- Description: {(jira_issue.description or "")[:1000]}

Candidate tickets:
{candidates_text}

Score each candidate. Return ALL candidates with their scores."""

        messages = [
            SystemMessage(content="You are an expert at comparing Jira tickets for similarity."),
            HumanMessage(content=rerank_prompt),
        ]
        try:
            structured_llm = self.chat.with_structured_output(RerankOutput)
            start = time.time()
            rerank_result = structured_llm.invoke(messages)
            duration_ms = int((time.time() - start) * 1000)
            log_llm_call("reranking", AZURE_OPENAI_LLM_DEPLOYMENT, messages, rerank_result, duration_ms)
            if isinstance(rerank_result, dict):
                rerank_result = RerankOutput(**rerank_result)
        except Exception as e:
            logger.error(f"Reranking failed, keeping original order: {e}")
            return candidates[:MAX_SIMILAR_TICKETS_AFTER_RERANK]

        score_map = {s.key: s for s in rerank_result.scored_tickets}
        reranked = []
        for candidate in candidates:
            scored = score_map.get(candidate.key)
            if scored and scored.relevance_score >= RERANK_SCORE_THRESHOLD:
                candidate.relevance_score = scored.relevance_score
                candidate.relevance_reason = scored.reason
                reranked.append(candidate)

        reranked.sort(key=lambda t: t.relevance_score or 0, reverse=True)
        return reranked[:MAX_SIMILAR_TICKETS_AFTER_RERANK]

    def _build_analysis(
        self,
        jira_issue: JiraIssue,
        processed_summary: str,
        logs_analysis: List[LogAnalysisOutput],
        similar_tickets: List[SimilarTicket],
    ) -> TicketAnalysis:
        final_analysis = self._generate_final_analysis(jira_issue, processed_summary, similar_tickets)

        # The reranked list is the source of truth for *which* tickets are
        # similar; the LLM only supplies the per-ticket similarity reason. This
        # keeps hallucinated keys out of the result entirely.
        reasons = {s.key: s.similarity_reason for s in final_analysis.similar_tickets}
        for ticket in similar_tickets:
            ticket.similarity_reason = reasons.get(ticket.key)

        return TicketAnalysis(
            ticket_key=jira_issue.key,
            ticket_title=jira_issue.summary or "",
            ticket_status=jira_issue.status,
            ticket_priority=jira_issue.priority,
            ticket_created=jira_issue.created,
            ticket_labels=jira_issue.labels or [],
            description=jira_issue.description or "",
            processed_summary=processed_summary,
            ticket_summary=final_analysis.ticket_summary,
            key_issues=final_analysis.key_issues,
            root_causes=final_analysis.root_causes,
            suggested_solutions=final_analysis.suggested_solutions,
            important_notes=final_analysis.important_notes,
            error_log_highlights=self._extract_error_highlights(logs_analysis),
            similar_tickets=similar_tickets,
        )

    @staticmethod
    def _extract_error_highlights(logs_analysis: List[LogAnalysisOutput]) -> List[ErrorLogHighlight]:
        highlights = []
        for log_analysis in logs_analysis or []:
            for error in log_analysis.errors:
                highlights.append(ErrorLogHighlight(
                    log_filename=log_analysis.log_filename,
                    context=error.context or "",
                    error_lines=error.error_lines,
                    exception_line=error.exception_line,
                    source_code_filename=error.source_code_filename,
                ))
        return highlights

    def _generate_final_analysis(
        self,
        jira_issue: JiraIssue,
        processed_summary: str,
        similar_tickets: List[SimilarTicket],
    ) -> FinalAnalysisOutput:
        messages = [
            SystemMessage(content=self._final_analysis_system_prompt()),
            HumanMessage(content=self._final_analysis_input_prompt(
                jira_issue, processed_summary, similar_tickets
            )),
        ]
        structured_llm = self.chat.with_structured_output(FinalAnalysisOutput)
        start = time.time()
        final_analysis = structured_llm.invoke(messages)
        duration_ms = int((time.time() - start) * 1000)
        log_llm_call("final_analysis", AZURE_OPENAI_LLM_DEPLOYMENT, messages, final_analysis, duration_ms)
        if isinstance(final_analysis, dict):
            final_analysis = FinalAnalysisOutput(**final_analysis)
        return final_analysis

    @staticmethod
    def _final_analysis_system_prompt() -> str:
        return """
        You are an expert Jira analyst assisting software engineers.

        Your task is to analyze a Jira ticket together with its most similar past tickets (if any) and produce a
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

    @staticmethod
    def _final_analysis_input_prompt(
        jira_issue: JiraIssue,
        processed_summary: str,
        similar_tickets: List[SimilarTicket],
    ) -> str:
        similar_tickets_details = []
        for idx, ticket in enumerate(similar_tickets, 1):
            similar_tickets_details.append(f"""
        Ticket {idx}:
        - Key: {ticket.key}
        - Title: {ticket.title}
        - Summary: {ticket.summary[:2400]}
        - Status: {ticket.status}
        - Resolution: {ticket.resolution}
        - Labels: {ticket.labels}
        - Issue Type: {ticket.issue_type}
        """)
        similar_tickets_formatted = "\n".join(similar_tickets_details).strip() or "NONE"

        return f"""
        Current Jira ticket information:
        - CURRENT TICKET: {jira_issue.key}
        - Summary: {processed_summary[:2400]}...
        - Description: {(jira_issue.description or "")[:1600]}...
        - Status: {jira_issue.status}
        - Priority: {jira_issue.priority}
        - Labels: {jira_issue.labels}

        Similar Jira tickets (reranked by relevance, {len(similar_tickets)} total):
        {similar_tickets_formatted}

        HARD RULES (must follow):
        - Return `similar_tickets` with exactly the tickets provided above. If none provided, return an empty list.
        - You MUST include EVERY provided similar ticket key exactly once.
        - Do NOT drop items. Do NOT merge items. Do NOT invent additional items.
        - If the "Similar Jira tickets" section is "NONE":
          - Return an empty list for similar_tickets: []
          - Do NOT mention any other Jira ticket keys in any section
          - Any similarity_reason/solutions based on similar tickets must be "unknown"

        IMPORTANT: For each similar ticket, provide a SPECIFIC similarity_reason explaining
        what EXACTLY makes it similar (e.g., "Same API endpoint failure", "Identical NPE in UserService.java",
        "Both involve timeout on database connection"). Do NOT use generic phrases like "shared error patterns".
        If no clear similarity exists, say "Low similarity - retrieved by vector search only".
        """
