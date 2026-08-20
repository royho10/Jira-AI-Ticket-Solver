"""Tests for the extracted, stateless Ticket Analyzer.

The analyzer orchestrates the full Analysis pipeline (fetch -> attachment
processing -> summarization -> vector search -> reranking) and returns a
structured TicketAnalysis. All external services are mocked here.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.ticket_analyzer import TicketAnalysis, TicketAnalyzer
from utils.openai_jira_ticket_processing import ErrorInLog, LogAnalysisOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_rag_result(tickets):
    """Build a fake Weaviate near_vector result."""
    objects = []
    for t in tickets:
        objects.append(SimpleNamespace(
            properties={
                "issue_key": t.get("key", "OTHER-1"),
                "summary": t.get("summary", "Some summary"),
                "status": t.get("status", "Done"),
                "resolution": t.get("resolution", "Fixed"),
                "clean_description": t.get("description", "Some description"),
                "title": t.get("title", "Some title"),
                "issue_type": t.get("issue_type", "Bug"),
                "labels": t.get("labels", []),
            },
            metadata=SimpleNamespace(distance=t.get("distance", 0.2)),
        ))
    return SimpleNamespace(objects=objects)


def make_chat(rerank_scores=None, final_analysis=None):
    """Build a fake chat model that dispatches by requested structured output."""
    from core.ticket_analyzer import FinalAnalysisOutput, RerankOutput, RerankTicketScore, SimilarTicketInfo

    if rerank_scores is None:
        rerank_scores = []
    if final_analysis is None:
        final_analysis = {}

    rerank_output = RerankOutput(scored_tickets=[
        RerankTicketScore(key=k, relevance_score=score, reason=f"reason for {k}")
        for k, score in rerank_scores
    ])

    analysis_output = FinalAnalysisOutput(
        ticket_summary=final_analysis.get("ticket_summary", "A concise summary"),
        key_issues=final_analysis.get("key_issues", ["Issue one"]),
        root_causes=final_analysis.get("root_causes", ["Root cause one"]),
        similar_tickets=[
            SimilarTicketInfo(
                key=s["key"],
                summary=s.get("summary", ""),
                similarity_reason=s.get("similarity_reason", "Same stack trace"),
                status=s.get("status", "Done"),
                title=s.get("title", "Title"),
                score=s.get("score", 0.8),
                issue_type=s.get("issue_type", "Bug"),
            )
            for s in final_analysis.get("similar_tickets", [])
        ],
        suggested_solutions=final_analysis.get("suggested_solutions", ["Do the thing"]),
        important_notes=final_analysis.get("important_notes", ["Note one"]),
    )

    chat = MagicMock()

    def with_structured_output(model, **kwargs):
        structured = MagicMock()
        structured.invoke.return_value = rerank_output if model is RerankOutput else analysis_output
        return structured

    chat.with_structured_output.side_effect = with_structured_output
    return chat


def build_analyzer(jira_issue, processor_result, rag_tickets, chat):
    """Wire an analyzer with all external services mocked."""
    jira_client = MagicMock()
    jira_client.fetch_issue_by_key.return_value = jira_issue

    processor = MagicMock()
    processor.process_issue.return_value = processor_result

    collection = MagicMock()
    collection.query.near_vector.return_value = make_rag_result(rag_tickets)
    db_client = MagicMock()
    db_client.collections.get.return_value = collection

    embedder = MagicMock()
    embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    analyzer = TicketAnalyzer(
        jira_client=jira_client,
        processor=processor,
        db_client=db_client,
        embedder=embedder,
        chat=chat,
    )
    return analyzer, jira_client, processor, collection, embedder


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_analyzer_module_has_no_streamlit_dependency():
    """The analyzer must be callable from the MCP server, which has no Streamlit."""
    source = Path("core/ticket_analyzer.py").read_text()
    assert "streamlit" not in source
    assert "session_state" not in source


@pytest.mark.deterministic
def test_ticket_analysis_exposes_the_documented_fields():
    fields = TicketAnalysis.model_fields
    for expected in (
        "ticket_key",
        "ticket_summary",
        "root_causes",
        "error_log_highlights",
        "processed_summary",
        "similar_tickets",
    ):
        assert expected in fields


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_analyze_orchestrates_the_full_pipeline(make_jira_issue):
    jira_issue = make_jira_issue({
        "key": "GC-100",
        "summary": "Collector crashes on startup",
        "description": "The collector crashes with a NullPointerException",
        "status": "Open",
        "priority": "High",
        "labels": ["collector"],
    })
    chat = make_chat(
        rerank_scores=[("GC-50", 9)],
        final_analysis={"similar_tickets": [{"key": "GC-50", "similarity_reason": "Same NPE"}]},
    )
    analyzer, jira_client, processor, collection, embedder = build_analyzer(
        jira_issue,
        ("Processed summary text", []),
        [{"key": "GC-50", "title": "Older crash"}],
        chat,
    )

    result = analyzer.analyze("GC-100")

    jira_client.fetch_issue_by_key.assert_called_once_with("GC-100")
    processor.process_issue.assert_called_once()
    embedder.embed_documents.assert_called_once()
    collection.query.near_vector.assert_called_once()

    assert isinstance(result, TicketAnalysis)
    assert result.ticket_key == "GC-100"
    assert result.ticket_title == "Collector crashes on startup"
    assert result.ticket_status == "Open"
    assert result.processed_summary == "Processed summary text"
    assert result.ticket_summary == "A concise summary"
    assert [t.key for t in result.similar_tickets] == ["GC-50"]


@pytest.mark.deterministic
def test_embedding_query_prioritises_description_then_summary(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-1", "description": "DESCRIPTION_MARKER"})
    chat = make_chat()
    analyzer, _, _, _, embedder = build_analyzer(
        jira_issue, ("SUMMARY_MARKER", []), [], chat
    )

    analyzer.analyze("GC-1")

    embedded_text = embedder.embed_documents.call_args.args[0][0]
    assert "DESCRIPTION_MARKER" in embedded_text
    assert "SUMMARY_MARKER" in embedded_text
    assert embedded_text.index("DESCRIPTION_MARKER") < embedded_text.index("SUMMARY_MARKER")


@pytest.mark.deterministic
def test_current_ticket_is_excluded_from_similar_tickets(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-100"})
    chat = make_chat(rerank_scores=[("GC-100", 10), ("GC-50", 8)])
    analyzer, _, _, _, _ = build_analyzer(
        jira_issue,
        ("Summary", []),
        [{"key": "GC-100"}, {"key": "GC-50"}],
        chat,
    )

    result = analyzer.analyze("GC-100")

    assert "GC-100" not in [t.key for t in result.similar_tickets]
    assert [t.key for t in result.similar_tickets] == ["GC-50"]


@pytest.mark.deterministic
def test_low_scoring_candidates_are_filtered_out(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-100"})
    chat = make_chat(rerank_scores=[("GC-50", 9), ("GC-51", 2)])
    analyzer, _, _, _, _ = build_analyzer(
        jira_issue,
        ("Summary", []),
        [{"key": "GC-50"}, {"key": "GC-51"}],
        chat,
    )

    result = analyzer.analyze("GC-100")

    keys = [t.key for t in result.similar_tickets]
    assert keys == ["GC-50"]


@pytest.mark.deterministic
def test_similar_tickets_are_sorted_by_relevance_score(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-100"})
    chat = make_chat(rerank_scores=[("GC-50", 6), ("GC-51", 10), ("GC-52", 8)])
    analyzer, _, _, _, _ = build_analyzer(
        jira_issue,
        ("Summary", []),
        [{"key": "GC-50"}, {"key": "GC-51"}, {"key": "GC-52"}],
        chat,
    )

    result = analyzer.analyze("GC-100")

    assert [t.key for t in result.similar_tickets] == ["GC-51", "GC-52", "GC-50"]
    assert [t.relevance_score for t in result.similar_tickets] == [10, 8, 6]


# ---------------------------------------------------------------------------
# Attachment shape edge cases
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_ticket_with_no_attachments_produces_no_error_highlights(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-1", "attachments": []})
    chat = make_chat()
    analyzer, _, _, _, _ = build_analyzer(jira_issue, ("Summary only", []), [], chat)

    result = analyzer.analyze("GC-1")

    assert result.error_log_highlights == []
    assert result.similar_tickets == []


@pytest.mark.deterministic
def test_ticket_with_logs_only_surfaces_error_highlights(make_jira_issue):
    jira_issue = make_jira_issue({
        "key": "GC-2",
        "attachments": [{"filename": "collector.log", "mime_type": "text/plain"}],
    })
    log_analysis = LogAnalysisOutput(
        log_filename="collector.log",
        errors=[ErrorInLog(
            context="Collector failed to start",
            error_lines="ERROR NullPointerException at Collector.java:42",
            exception_line="java.lang.NullPointerException",
            source_code_filename="Collector.java",
        )],
    )
    chat = make_chat()
    analyzer, _, _, _, _ = build_analyzer(
        jira_issue, ("Summary with logs", [log_analysis]), [], chat
    )

    result = analyzer.analyze("GC-2")

    assert len(result.error_log_highlights) == 1
    highlight = result.error_log_highlights[0]
    assert highlight.log_filename == "collector.log"
    assert highlight.exception_line == "java.lang.NullPointerException"
    assert highlight.source_code_filename == "Collector.java"


@pytest.mark.deterministic
def test_ticket_with_images_keeps_attachment_insights_in_summary(make_jira_issue):
    jira_issue = make_jira_issue({
        "key": "GC-3",
        "attachments": [{"filename": "screenshot.png", "mime_type": "image/png"}],
    })
    chat = make_chat()
    analyzer, _, processor, _, _ = build_analyzer(
        jira_issue,
        ("Summary including image analysis: red error dialog visible", []),
        [],
        chat,
    )

    result = analyzer.analyze("GC-3")

    assert "image analysis" in result.processed_summary
    assert result.error_log_highlights == []
    # The image-bearing issue is handed to the processor untouched
    assert processor.process_issue.call_args.args[0] is jira_issue


@pytest.mark.deterministic
def test_no_similar_matches_still_returns_a_complete_analysis(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-4", "summary": "Isolated issue"})
    chat = make_chat()
    analyzer, _, _, collection, _ = build_analyzer(jira_issue, ("Summary", []), [], chat)

    result = analyzer.analyze("GC-4")

    assert result.similar_tickets == []
    assert result.ticket_key == "GC-4"
    assert result.ticket_summary
    assert result.key_issues


@pytest.mark.deterministic
def test_reranking_is_skipped_when_there_are_no_candidates(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-5"})
    chat = make_chat()
    analyzer, _, _, _, _ = build_analyzer(jira_issue, ("Summary", []), [], chat)

    analyzer.analyze("GC-5")

    requested_models = [c.args[0].__name__ for c in chat.with_structured_output.call_args_list]
    assert "RerankOutput" not in requested_models


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_vector_search_failure_degrades_to_analysis_without_similar_tickets(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-6"})
    chat = make_chat()
    analyzer, _, _, collection, _ = build_analyzer(jira_issue, ("Summary", []), [], chat)
    collection.query.near_vector.side_effect = RuntimeError("weaviate unreachable")

    result = analyzer.analyze("GC-6")

    assert result.similar_tickets == []
    assert result.ticket_summary


@pytest.mark.deterministic
def test_rerank_failure_degrades_to_vector_order(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-7"})
    from core.ticket_analyzer import RerankOutput

    chat = MagicMock()

    def with_structured_output(model, **kwargs):
        structured = MagicMock()
        if model is RerankOutput:
            structured.invoke.side_effect = RuntimeError("rerank failed")
        else:
            structured.invoke.return_value = make_chat().with_structured_output(model).invoke([])
        return structured

    chat.with_structured_output.side_effect = with_structured_output
    analyzer, _, _, _, _ = build_analyzer(
        jira_issue, ("Summary", []), [{"key": "GC-50"}, {"key": "GC-51"}], chat
    )

    result = analyzer.analyze("GC-7")

    assert [t.key for t in result.similar_tickets] == ["GC-50", "GC-51"]


@pytest.mark.deterministic
def test_progress_callback_receives_status_updates(make_jira_issue):
    jira_issue = make_jira_issue({"key": "GC-8"})
    chat = make_chat()
    analyzer, _, _, _, _ = build_analyzer(jira_issue, ("Summary", []), [], chat)

    messages = []
    analyzer.analyze("GC-8", on_progress=messages.append)

    assert messages
    assert all(isinstance(m, str) for m in messages)
