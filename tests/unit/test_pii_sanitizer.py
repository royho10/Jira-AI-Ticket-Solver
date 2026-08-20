"""Tests for the PII Sanitizer.

Every response leaving the Remote MCP Server passes through this module, so a
silent failure here leaks customer data into the user's Claude context. These
tests pin the mechanics (what gets sent, what comes back, what is preserved) and
the fail-closed behaviour. Redaction *quality* is covered by the llm_eval tests
in tests/eval/test_pii_sanitization.py.
"""
from unittest.mock import MagicMock

import pytest

from core.pii_sanitizer import (
    DEFAULT_PII_CATEGORIES,
    PiiCategory,
    PiiSanitizer,
    SanitizationError,
)
from core.ticket_analyzer import ErrorLogHighlight, SimilarTicket, TicketAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_analysis(**overrides) -> TicketAnalysis:
    """A fully populated analysis, one unique marker per sanitizable field."""
    data = dict(
        ticket_key="GC-100",
        ticket_title="TITLE_MARKER",
        ticket_status="Open",
        ticket_priority="High",
        ticket_created="2024-01-01T00:00:00.000+0000",
        ticket_labels=["LABEL_MARKER"],
        description="DESCRIPTION_MARKER",
        processed_summary="PROCESSED_MARKER",
        ticket_summary="SUMMARY_MARKER",
        key_issues=["ISSUE_MARKER"],
        root_causes=["ROOT_CAUSE_MARKER"],
        suggested_solutions=["SOLUTION_MARKER"],
        important_notes=["NOTE_MARKER"],
        error_log_highlights=[ErrorLogHighlight(
            log_filename="LOG_FILENAME_MARKER.log",
            context="CONTEXT_MARKER",
            error_lines="ERROR_LINES_MARKER",
            exception_line="EXCEPTION_MARKER",
            source_code_filename="SOURCE_FILENAME_MARKER.java",
        )],
        similar_tickets=[SimilarTicket(
            key="GC-50",
            title="SIMILAR_TITLE_MARKER",
            summary="SIMILAR_SUMMARY_MARKER",
            status="Done",
            resolution="Fixed",
            issue_type="Bug",
            labels=["SIMILAR_LABEL_MARKER"],
            vector_score=0.8,
            relevance_score=9,
            relevance_reason="RELEVANCE_REASON_MARKER",
            similarity_reason="SIMILARITY_REASON_MARKER",
        )],
    )
    data.update(overrides)
    return TicketAnalysis(**data)


ALL_MARKERS = [
    "TITLE_MARKER", "LABEL_MARKER", "DESCRIPTION_MARKER", "PROCESSED_MARKER",
    "SUMMARY_MARKER", "ISSUE_MARKER", "ROOT_CAUSE_MARKER", "SOLUTION_MARKER",
    "NOTE_MARKER", "LOG_FILENAME_MARKER", "SOURCE_FILENAME_MARKER",
    "CONTEXT_MARKER", "ERROR_LINES_MARKER", "EXCEPTION_MARKER",
    "SIMILAR_TITLE_MARKER", "SIMILAR_SUMMARY_MARKER", "SIMILAR_LABEL_MARKER",
    "RELEVANCE_REASON_MARKER", "SIMILARITY_REASON_MARKER",
]


def make_chat(transform=None, drop_ids=(), extra_ids=()):
    """A chat model that redacts by applying `transform` to each segment."""
    from core.pii_sanitizer import RedactedSegment, SanitizationOutput

    if transform is None:
        def transform(text):
            return text

    captured = {}

    def invoke(messages):
        captured["messages"] = messages
        prompt = "\n".join(str(m.content) for m in messages)
        captured["prompt"] = prompt
        segments = [
            RedactedSegment(id=segment.id, text=transform(segment.text))
            for segment in _parse_segments(prompt)
            if segment.id not in drop_ids
        ]
        segments.extend(RedactedSegment(id=i, text="ignored") for i in extra_ids)
        return SanitizationOutput(redacted_segments=segments)

    def _parse_segments(prompt):
        from core.pii_sanitizer import RedactedSegment as Segment
        found = []
        for line in prompt.splitlines():
            if line.startswith("[[") and "]] " in line:
                raw_id, text = line[2:].split("]] ", 1)
                if raw_id.isdigit():
                    found.append(Segment(id=int(raw_id), text=text))
        return found

    structured = MagicMock()
    structured.invoke.side_effect = invoke
    chat = MagicMock()
    chat.with_structured_output.return_value = structured
    chat.captured = captured
    return chat


def prompt_of(chat) -> str:
    return chat.captured["prompt"]


# ---------------------------------------------------------------------------
# What the LLM is asked to redact
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_every_free_text_field_is_submitted_for_redaction():
    chat = make_chat()
    PiiSanitizer(chat=chat).sanitize(make_analysis())

    prompt = prompt_of(chat)
    for marker in ALL_MARKERS:
        assert marker in prompt, f"{marker} was never submitted for redaction"


@pytest.mark.deterministic
def test_structural_fields_are_not_submitted_for_redaction():
    """Ticket keys and statuses are structural, not customer data -- sending them
    invites the LLM to mangle identifiers we rely on."""
    chat = make_chat()
    PiiSanitizer(chat=chat).sanitize(make_analysis())

    prompt = prompt_of(chat)
    for structural in ("GC-100", "GC-50", "Open", "High", "Fixed"):
        assert structural not in prompt


@pytest.mark.deterministic
def test_attachment_filenames_are_submitted_for_redaction():
    """Customers name their log uploads after their own hosts and orgs, and the
    filename is printed in the MCP response."""
    chat = make_chat()
    PiiSanitizer(chat=chat).sanitize(make_analysis())

    prompt = prompt_of(chat)
    assert "LOG_FILENAME_MARKER.log" in prompt
    assert "SOURCE_FILENAME_MARKER.java" in prompt


@pytest.mark.deterministic
def test_every_string_field_on_the_models_is_classified():
    """A new free-text field must be added to the sanitized lists, not forgotten.

    Every string-bearing field on the three result models has to appear in either
    the sanitized tuples or the structural tuples; an unclassified field would
    otherwise ship to the user's Claude context unredacted, silently.
    """
    from core.pii_sanitizer import (
        SANITIZED_HIGHLIGHT_FIELDS,
        SANITIZED_LIST_FIELDS,
        SANITIZED_SIMILAR_TICKET_FIELDS,
        SANITIZED_TEXT_FIELDS,
        STRUCTURAL_HIGHLIGHT_FIELDS,
        STRUCTURAL_SIMILAR_TICKET_FIELDS,
        STRUCTURAL_TEXT_FIELDS,
    )

    def string_fields(model):
        found = set()
        for name, field in model.model_fields.items():
            annotation = str(field.annotation)
            if "str" in annotation:
                found.add(name)
        return found

    cases = [
        (TicketAnalysis, set(SANITIZED_TEXT_FIELDS) | set(SANITIZED_LIST_FIELDS) | set(STRUCTURAL_TEXT_FIELDS)),
        (ErrorLogHighlight, set(SANITIZED_HIGHLIGHT_FIELDS) | set(STRUCTURAL_HIGHLIGHT_FIELDS)),
        (SimilarTicket, set(SANITIZED_SIMILAR_TICKET_FIELDS) | {"labels"} | set(STRUCTURAL_SIMILAR_TICKET_FIELDS)),
    ]
    for model, classified in cases:
        unclassified = string_fields(model) - classified
        assert not unclassified, (
            f"{model.__name__} has unclassified string fields {sorted(unclassified)} -- "
            "add them to the sanitized or the structural tuple in core/pii_sanitizer.py"
        )


@pytest.mark.deterministic
def test_empty_and_blank_fields_are_not_submitted():
    chat = make_chat()
    PiiSanitizer(chat=chat).sanitize(make_analysis(description="", ticket_summary="   "))

    prompt = prompt_of(chat)
    segment_count = sum(1 for line in prompt.splitlines() if line.startswith("[["))
    assert segment_count == len(ALL_MARKERS) - 2


@pytest.mark.deterministic
def test_analysis_with_no_free_text_skips_the_llm_entirely():
    chat = make_chat()
    analysis = TicketAnalysis(ticket_key="GC-1")

    result = PiiSanitizer(chat=chat).sanitize(analysis)

    chat.with_structured_output.assert_not_called()
    assert result.ticket_key == "GC-1"


# ---------------------------------------------------------------------------
# Writing the redacted text back
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_redacted_text_lands_in_the_field_it_came_from():
    chat = make_chat(transform=lambda text: f"CLEAN::{text}")

    result = PiiSanitizer(chat=chat).sanitize(make_analysis())

    assert result.ticket_title == "CLEAN::TITLE_MARKER"
    assert result.description == "CLEAN::DESCRIPTION_MARKER"
    assert result.processed_summary == "CLEAN::PROCESSED_MARKER"
    assert result.ticket_summary == "CLEAN::SUMMARY_MARKER"
    assert result.ticket_labels == ["CLEAN::LABEL_MARKER"]
    assert result.key_issues == ["CLEAN::ISSUE_MARKER"]
    assert result.root_causes == ["CLEAN::ROOT_CAUSE_MARKER"]
    assert result.suggested_solutions == ["CLEAN::SOLUTION_MARKER"]
    assert result.important_notes == ["CLEAN::NOTE_MARKER"]

    highlight = result.error_log_highlights[0]
    assert highlight.context == "CLEAN::CONTEXT_MARKER"
    assert highlight.error_lines == "CLEAN::ERROR_LINES_MARKER"
    assert highlight.exception_line == "CLEAN::EXCEPTION_MARKER"

    similar = result.similar_tickets[0]
    assert similar.title == "CLEAN::SIMILAR_TITLE_MARKER"
    assert similar.summary == "CLEAN::SIMILAR_SUMMARY_MARKER"
    assert similar.labels == ["CLEAN::SIMILAR_LABEL_MARKER"]
    assert similar.relevance_reason == "CLEAN::RELEVANCE_REASON_MARKER"
    assert similar.similarity_reason == "CLEAN::SIMILARITY_REASON_MARKER"


@pytest.mark.deterministic
def test_structural_fields_survive_sanitization():
    chat = make_chat(transform=lambda text: "[REDACTED]")

    result = PiiSanitizer(chat=chat).sanitize(make_analysis())

    assert result.ticket_key == "GC-100"
    assert result.ticket_status == "Open"
    assert result.ticket_priority == "High"
    assert result.ticket_created == "2024-01-01T00:00:00.000+0000"

    similar = result.similar_tickets[0]
    assert similar.key == "GC-50"
    assert similar.status == "Done"
    assert similar.resolution == "Fixed"
    assert similar.issue_type == "Bug"
    assert similar.vector_score == 0.8
    assert similar.relevance_score == 9


@pytest.mark.deterministic
def test_the_caller_s_analysis_is_left_untouched():
    """The sanitizer returns a copy so an unsanitized original can never be
    mistaken for the sanitized result."""
    original = make_analysis()
    chat = make_chat(transform=lambda text: "[REDACTED]")

    result = PiiSanitizer(chat=chat).sanitize(original)

    assert original.ticket_title == "TITLE_MARKER"
    assert original.key_issues == ["ISSUE_MARKER"]
    assert original.similar_tickets[0].labels == ["SIMILAR_LABEL_MARKER"]
    assert result is not original


@pytest.mark.deterministic
def test_analysis_with_no_pii_is_returned_verbatim():
    chat = make_chat()  # identity transform: the LLM found nothing to redact
    original = make_analysis()

    result = PiiSanitizer(chat=chat).sanitize(original)

    assert result.model_dump() == original.model_dump()


# ---------------------------------------------------------------------------
# Configurable scope
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_default_scope_covers_the_documented_pii_categories():
    for expected in (
        PiiCategory.PERSON_NAME,
        PiiCategory.EMAIL_ADDRESS,
        PiiCategory.IP_ADDRESS,
        PiiCategory.DOMAIN,
        PiiCategory.TENANT_ID,
    ):
        assert expected in DEFAULT_PII_CATEGORIES


@pytest.mark.deterministic
def test_configured_categories_are_the_only_ones_requested():
    chat = make_chat()
    sanitizer = PiiSanitizer(chat=chat, categories=[PiiCategory.EMAIL_ADDRESS])

    sanitizer.sanitize(make_analysis())

    prompt = prompt_of(chat)
    assert PiiCategory.EMAIL_ADDRESS.value in prompt
    assert PiiCategory.IP_ADDRESS.value not in prompt


@pytest.mark.deterministic
def test_an_empty_category_list_is_rejected():
    """An empty scope would silently turn the mandatory redaction pass into a
    no-op, which is exactly the failure this module exists to prevent."""
    with pytest.raises(ValueError):
        PiiSanitizer(chat=make_chat(), categories=[])


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_llm_failure_raises_rather_than_returning_unsanitized_text():
    chat = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("azure unavailable")
    chat.with_structured_output.return_value = structured

    with pytest.raises(SanitizationError):
        PiiSanitizer(chat=chat).sanitize(make_analysis())


@pytest.mark.deterministic
def test_a_segment_missing_from_the_response_raises():
    """A dropped segment would pass through unredacted, so the whole pass fails."""
    chat = make_chat(transform=lambda text: "[REDACTED]", drop_ids=(2,))

    with pytest.raises(SanitizationError):
        PiiSanitizer(chat=chat).sanitize(make_analysis())


@pytest.mark.deterministic
def test_unknown_segment_ids_in_the_response_are_ignored():
    chat = make_chat(transform=lambda text: f"CLEAN::{text}", extra_ids=(9999,))

    result = PiiSanitizer(chat=chat).sanitize(make_analysis())

    assert result.ticket_title == "CLEAN::TITLE_MARKER"


@pytest.mark.deterministic
def test_the_failure_message_never_repeats_the_unsanitized_text():
    chat = make_chat(transform=lambda text: "[REDACTED]", drop_ids=(1,))

    with pytest.raises(SanitizationError) as exc_info:
        PiiSanitizer(chat=chat).sanitize(make_analysis())

    message = str(exc_info.value)
    for marker in ALL_MARKERS:
        assert marker not in message
