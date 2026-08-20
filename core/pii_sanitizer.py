"""LLM-based PII redaction for analysis results.

Every response leaving the Remote MCP Server must pass through this module: the
Analysis is built from raw Jira data that can name customers, hosts and tenants,
and once that text reaches a user's Claude context it cannot be recalled.

The pass fails closed. If the LLM call errors, or if it fails to return a
redacted counterpart for every submitted segment, `SanitizationError` is raised
rather than returning partially-sanitized text.
"""
import logging
import time
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_LLM_DEPLOYMENT,
    AZURE_OPENAI_TEMPERATURE,
)
from core.ticket_analyzer import TicketAnalysis
from utils.llm_logger import log_llm_call

logger = logging.getLogger(__name__)


class SanitizationError(RuntimeError):
    """The redaction pass could not be completed, so nothing may be returned.

    Messages must never quote the text being sanitized -- this exception is
    logged, and the unsanitized text is what we are trying to keep out of logs.
    """


class PiiCategory(str, Enum):
    """A category of customer-identifying information to redact."""
    PERSON_NAME = "person_name"
    EMAIL_ADDRESS = "email_address"
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    TENANT_ID = "tenant_id"
    PHONE_NUMBER = "phone_number"
    CREDENTIAL = "credential"


PII_CATEGORY_INSTRUCTIONS = {
    PiiCategory.PERSON_NAME: "names of individuals (customer contacts, end users, engineers)",
    PiiCategory.EMAIL_ADDRESS: "email addresses",
    PiiCategory.IP_ADDRESS: "IPv4 and IPv6 addresses, except non-routable ones such as 127.0.0.1, 0.0.0.0 and localhost",
    PiiCategory.DOMAIN: "customer hostnames and domain names, except public vendor domains (github.com, microsoft.com, atlassian.net)",
    PiiCategory.TENANT_ID: "tenant, account, org and customer identifiers, including UUIDs used as such",
    PiiCategory.PHONE_NUMBER: "phone numbers",
    PiiCategory.CREDENTIAL: "API keys, tokens, passwords and connection strings",
}

DEFAULT_PII_CATEGORIES: Sequence[PiiCategory] = tuple(PII_CATEGORY_INSTRUCTIONS)

# Fields carrying free text written by humans or produced by the LLM. Attachment
# filenames count: customers name their log uploads after their own hosts and
# organizations.
SANITIZED_TEXT_FIELDS = ("ticket_title", "description", "processed_summary", "ticket_summary")
SANITIZED_LIST_FIELDS = ("ticket_labels", "key_issues", "root_causes", "suggested_solutions", "important_notes")
SANITIZED_HIGHLIGHT_FIELDS = ("log_filename", "source_code_filename", "context", "error_lines", "exception_line")
SANITIZED_SIMILAR_TICKET_FIELDS = ("title", "summary", "relevance_reason", "similarity_reason")

# Fields deliberately withheld from the pass: identifiers, enum-like values and
# timestamps, which carry no customer prose and must survive character-for-
# character for the result to be usable. Named explicitly so that a new free-text
# field on the models fails the coverage test in tests/unit/test_pii_sanitizer.py
# instead of silently shipping unredacted.
STRUCTURAL_TEXT_FIELDS = ("ticket_key", "ticket_status", "ticket_priority", "ticket_created")
STRUCTURAL_HIGHLIGHT_FIELDS = ()
STRUCTURAL_SIMILAR_TICKET_FIELDS = ("key", "status", "resolution", "issue_type")

SEGMENT_PREFIX = "[["


class RedactedSegment(BaseModel):
    """One submitted segment, returned with its PII replaced."""
    id: int = Field(description="The id of the segment, exactly as submitted")
    text: str = Field(description="The segment text with all PII replaced by placeholders")


class SanitizationOutput(BaseModel):
    """The redacted counterpart of every submitted segment."""
    redacted_segments: List[RedactedSegment] = Field(description="Every submitted segment, redacted")


class _Segment:
    """A piece of free text plus the write-back that puts its redaction home."""

    def __init__(self, text: str, apply: Callable[[str], None]):
        self.text = text
        self.apply = apply


class PiiSanitizer:
    """Redacts customer-identifying information from a `TicketAnalysis`."""

    def __init__(
        self,
        chat: Any = None,
        categories: Optional[Sequence[PiiCategory]] = None,
    ):
        resolved_categories = tuple(DEFAULT_PII_CATEGORIES if categories is None else categories)
        if not resolved_categories:
            raise ValueError(
                "At least one PII category is required -- an empty scope would turn "
                "the mandatory redaction pass into a no-op."
            )
        self.categories = resolved_categories
        self.chat = chat or AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_LLM_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=AZURE_OPENAI_TEMPERATURE,
        )

    def sanitize(self, analysis: TicketAnalysis) -> TicketAnalysis:
        """Return a copy of `analysis` with PII redacted.

        Raises `SanitizationError` if redaction could not be completed.
        """
        sanitized = analysis.model_copy(deep=True)
        segments = self._collect_segments(sanitized)
        if not segments:
            return sanitized

        redactions = self._request_redactions(segments)
        for index, segment in enumerate(segments):
            segment.apply(redactions[index])
        return sanitized

    # ------------------------
    # Collecting free text
    # ------------------------
    @staticmethod
    def _collect_segments(analysis: TicketAnalysis) -> List[_Segment]:
        segments: List[_Segment] = []

        def add_attribute(owner: Any, name: str) -> None:
            value = getattr(owner, name, None)
            if isinstance(value, str) and value.strip():
                segments.append(_Segment(
                    value,
                    lambda text, o=owner, n=name: setattr(o, n, text),
                ))

        def add_list(owner: Any, name: str) -> None:
            values = getattr(owner, name, None) or []
            for index, value in enumerate(values):
                if isinstance(value, str) and value.strip():
                    segments.append(_Segment(
                        value,
                        lambda text, lst=values, i=index: lst.__setitem__(i, text),
                    ))

        for name in SANITIZED_TEXT_FIELDS:
            add_attribute(analysis, name)
        for name in SANITIZED_LIST_FIELDS:
            add_list(analysis, name)
        for highlight in analysis.error_log_highlights:
            for name in SANITIZED_HIGHLIGHT_FIELDS:
                add_attribute(highlight, name)
        for ticket in analysis.similar_tickets:
            for name in SANITIZED_SIMILAR_TICKET_FIELDS:
                add_attribute(ticket, name)
            add_list(ticket, "labels")
        return segments

    # ------------------------
    # The redaction pass
    # ------------------------
    def _request_redactions(self, segments: List[_Segment]) -> List[str]:
        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=self._input_prompt(segments)),
        ]
        try:
            structured_llm = self.chat.with_structured_output(SanitizationOutput)
            start = time.time()
            output = structured_llm.invoke(messages)
            duration_ms = int((time.time() - start) * 1000)
            log_llm_call("pii_sanitization", AZURE_OPENAI_LLM_DEPLOYMENT, messages, output, duration_ms)
            if isinstance(output, dict):
                output = SanitizationOutput(**output)
        except Exception as e:
            # Deliberately logs only the exception type: the payload we failed to
            # sanitize must not reach the logs.
            logger.error(f"PII sanitization failed ({type(e).__name__}), refusing to return the analysis")
            raise SanitizationError("PII sanitization pass failed") from e

        by_id = {segment.id: segment.text for segment in output.redacted_segments}
        missing = [index for index in range(len(segments)) if index not in by_id]
        if missing:
            raise SanitizationError(
                f"PII sanitization returned {len(by_id)} of {len(segments)} segments; "
                f"{len(missing)} would have been returned unredacted"
            )
        return [by_id[index] for index in range(len(segments))]

    def _system_prompt(self) -> str:
        return """
        You are a data redaction service protecting customer privacy.

        You receive numbered text segments taken from a Jira ticket analysis and
        return every segment with customer-identifying information replaced by a
        placeholder.

        Rules:
        - Return EVERY segment you are given, with the SAME id. Never drop, merge,
          reorder or invent segments.
        - Replace each piece of identifying information with an upper-case
          placeholder naming its category, e.g. [REDACTED_EMAIL_ADDRESS].
        - Preserve all other text EXACTLY as written, character for character.
          Do not summarize, rephrase, translate, reformat or fix anything.
        - Keep technical content intact: exception types, stack frames, class and
          method names, file paths, Jira ticket keys, log levels, error messages,
          version numbers, HTTP status codes and code snippets are NOT PII.
        - A filename or path is technical, but an identifying part *inside* one
          still gets replaced: `agent-acme-prod01.log` becomes
          `agent-[REDACTED_DOMAIN].log`, while `PolicyCompiler.java` is untouched.
        - If a segment contains nothing to redact, return it unchanged.
        """

    def _input_prompt(self, segments: List[_Segment]) -> str:
        # Built line by line rather than as an indented block: segment markers
        # must sit at the start of their line for the format to be unambiguous.
        lines = ["Redact ONLY these categories of information:"]
        lines.extend(
            f"- {category.value}: {PII_CATEGORY_INSTRUCTIONS[category]}"
            for category in self.categories
        )
        lines += [
            "",
            "Anything not listed above is not in scope -- leave it exactly as it is.",
            "",
            f"Below are {len(segments)} segments. Each begins with its id in double square",
            "brackets at the start of a line; the segment text runs until the next id or",
            "the end of the input, and may span multiple lines.",
            "",
            f"Return all {len(segments)} segments, ids 0 to {len(segments) - 1}.",
            "",
            "Segments:",
        ]
        lines.extend(
            f"{SEGMENT_PREFIX}{index}]] {segment.text}"
            for index, segment in enumerate(segments)
        )
        return "\n".join(lines)
