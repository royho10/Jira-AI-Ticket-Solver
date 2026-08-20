"""Redaction quality of the PII Sanitizer against a real LLM.

The deterministic tests in tests/unit/test_pii_sanitizer.py pin the mechanics
(what is submitted, how it is written back, how it fails). These check the thing
that actually matters in production: curated customer data is gone from the
result, and the technical content an engineer needs is still there.
"""
import json

import pytest

from tests.conftest import dataset_range


def build_analysis(data: dict):
    from core.ticket_analyzer import ErrorLogHighlight, SimilarTicket, TicketAnalysis

    return TicketAnalysis(
        **{
            **data,
            "error_log_highlights": [
                ErrorLogHighlight(**h) for h in data.get("error_log_highlights", [])
            ],
            "similar_tickets": [
                SimilarTicket(**t) for t in data.get("similar_tickets", [])
            ],
        }
    )


@pytest.mark.llm_eval
@pytest.mark.parametrize("case_index", dataset_range("pii_sanitization"))
def test_curated_pii_is_redacted_and_technical_content_survives(
    pii_sanitization_dataset, llm_chat, case_index
):
    from core.pii_sanitizer import PiiSanitizer

    case = pii_sanitization_dataset[case_index]
    analysis = build_analysis(case["input"])

    result = PiiSanitizer(chat=llm_chat).sanitize(analysis)
    serialized = json.dumps(result.model_dump())

    for pii in case["expected"]["must_not_contain"]:
        assert pii not in serialized, f"[{case['name']}] leaked {pii!r}"

    for technical in case["expected"]["must_contain"]:
        assert technical in serialized, f"[{case['name']}] lost {technical!r}"


@pytest.mark.llm_eval
def test_analysis_without_pii_is_left_semantically_intact(
    pii_sanitization_dataset, llm_chat
):
    """The no-PII case must come back with no placeholders injected -- an
    over-eager sanitizer destroys the analysis just as surely as a leak."""
    from core.pii_sanitizer import PiiSanitizer

    case = next(c for c in pii_sanitization_dataset if not c["expected"]["must_not_contain"])
    analysis = build_analysis(case["input"])

    result = PiiSanitizer(chat=llm_chat).sanitize(analysis)

    assert "[REDACTED" not in json.dumps(result.model_dump())
