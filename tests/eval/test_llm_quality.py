"""
LLM-as-Judge evaluation tests.

These tests use a separate LLM call to judge the quality of outputs
from the main LLM calls. They validate subjective quality dimensions
like accuracy, faithfulness, and relevance using rubric-based scoring.

These are the most expensive tests - run them nightly or pre-release.
"""
import pytest

from tests.conftest import dataset_range
from utils.openai_jira_ticket_processing import (
    OpenAIJiraIssueLLMProcessor,
    LogAnalysisOutput,
    FinalIssueSummeryOutput,
)

PASS_THRESHOLD = 3.0  # Out of 5


# ===========================================================================
# Error Extraction Quality
# ===========================================================================
@pytest.mark.llm_eval
class TestErrorExtractionQuality:

    @pytest.mark.parametrize("case_idx", dataset_range("error_extraction"))
    def test_error_extraction_quality(
        self, llm_processor, error_extraction_dataset, llm_judge, case_idx
    ):
        """Judge the quality of error extraction output."""
        case = error_extraction_dataset[case_idx]
        if not case["expected"]["should_find_errors"]:
            pytest.skip("No errors expected in this case")

        result = llm_processor._extract_errors_from_description(case["input"]["description"])
        if result is None:
            pytest.fail(f"[{case['test_id']}] Expected errors but LLM returned None")

        judgment = llm_judge(
            input_text=f"Description:\n{case['input']['description']}",
            output_text=str(result),
            rubric=case["rubric"],
        )

        avg_score = judgment.get("average_score", 0)
        assert avg_score >= PASS_THRESHOLD, (
            f"[{case['test_id']}] Quality score {avg_score:.1f} < {PASS_THRESHOLD}\n"
            f"Judgment: {judgment.get('explanation', 'N/A')}\n"
            f"Scores: {judgment.get('scores', {})}"
        )


# ===========================================================================
# Log Parsing Quality
# ===========================================================================
@pytest.mark.llm_eval
class TestLogParsingQuality:

    @pytest.mark.parametrize("case_idx", dataset_range("log_parsing"))
    def test_log_parsing_quality(
        self, llm_processor, log_parsing_dataset, llm_judge, case_idx
    ):
        """Judge the quality of log parsing output."""
        case = log_parsing_dataset[case_idx]
        log_text = case["input"]["log_text"]
        filename = case["input"]["filename"]

        result = llm_processor._summarize_log((log_text, filename))
        if result is None:
            pytest.fail(f"[{case['test_id']}] LLM returned None")

        judgment = llm_judge(
            input_text=f"Log file ({filename}):\n{log_text}",
            output_text=str(result),
            rubric=case["rubric"],
        )

        avg_score = judgment.get("average_score", 0)
        assert avg_score >= PASS_THRESHOLD, (
            f"[{case['test_id']}] Quality score {avg_score:.1f} < {PASS_THRESHOLD}\n"
            f"Judgment: {judgment.get('explanation', 'N/A')}\n"
            f"Scores: {judgment.get('scores', {})}"
        )


# ===========================================================================
# Final Summary Quality
# ===========================================================================
@pytest.mark.llm_eval
class TestFinalSummaryQuality:

    @pytest.mark.parametrize("case_idx", dataset_range("final_summary"))
    def test_final_summary_quality(
        self, llm_processor, final_summary_dataset, make_jira_issue, llm_judge, case_idx
    ):
        """Judge the quality of final summary generation."""
        case = final_summary_dataset[case_idx]
        jira_issue = make_jira_issue(case["input"]["jira_issue"])

        result = llm_processor._create_final_issue_summary(
            jira_issue,
            case["input"]["comments_text"],
            case["input"]["summarized_attachments"],
            case["input"]["related_issues_text"],
        )

        input_text = (
            f"Ticket: {jira_issue.key} - {jira_issue.summary}\n"
            f"Description: {jira_issue.description}\n"
            f"Comments: {case['input']['comments_text']}\n"
            f"Attachments: {case['input']['summarized_attachments']}"
        )

        judgment = llm_judge(
            input_text=input_text,
            output_text=str(result),
            rubric=case["rubric"],
        )

        avg_score = judgment.get("average_score", 0)
        assert avg_score >= PASS_THRESHOLD, (
            f"[{case['test_id']}] Quality score {avg_score:.1f} < {PASS_THRESHOLD}\n"
            f"Judgment: {judgment.get('explanation', 'N/A')}\n"
            f"Scores: {judgment.get('scores', {})}"
        )


# ===========================================================================
# Reranking Quality
# ===========================================================================
@pytest.mark.llm_eval
class TestRerankingQuality:

    @pytest.mark.parametrize("case_idx", dataset_range("reranking"))
    def test_reranking_quality(
        self, llm_chat, reranking_dataset, llm_judge, case_idx
    ):
        """Judge the quality of reranking output."""
        from core.ticket_analyzer import RerankOutput
        from langchain_core.messages import SystemMessage, HumanMessage

        case = reranking_dataset[case_idx]
        current = case["input"]["current_ticket"]
        candidates = case["input"]["candidates"]

        candidate_details = "\n".join(
            f"- Key: {t['key']}, Title: {t['title']}, Summary: {t['summary']}"
            for t in candidates
        )

        rerank_prompt = f"""Score each candidate on 0-10 relevance to the current ticket.

Current ticket:
- Key: {current['key']}
- Summary: {current['summary']}
- Description: {current['description']}

Candidates:
{candidate_details}"""

        structured_llm = llm_chat.with_structured_output(RerankOutput)
        result = structured_llm.invoke([
            SystemMessage(content="You are an expert at comparing Jira tickets for similarity."),
            HumanMessage(content=rerank_prompt),
        ])

        judgment = llm_judge(
            input_text=f"Current: {current['summary']}\nCandidates:\n{candidate_details}",
            output_text=str(result),
            rubric=case["rubric"],
        )

        avg_score = judgment.get("average_score", 0)
        assert avg_score >= PASS_THRESHOLD, (
            f"[{case['test_id']}] Quality score {avg_score:.1f} < {PASS_THRESHOLD}\n"
            f"Judgment: {judgment.get('explanation', 'N/A')}\n"
            f"Scores: {judgment.get('scores', {})}"
        )


# ===========================================================================
# Final Analysis Quality
# ===========================================================================
@pytest.mark.llm_eval
class TestFinalAnalysisQuality:

    @pytest.mark.parametrize("case_idx", dataset_range("final_analysis"))
    def test_final_analysis_quality(
        self, llm_chat, final_analysis_dataset, llm_judge, case_idx
    ):
        """Judge the quality of final analysis generation."""
        from core.ticket_analyzer import FinalAnalysisOutput
        from langchain_core.messages import SystemMessage, HumanMessage

        case = final_analysis_dataset[case_idx]
        current = case["input"]["current_ticket"]
        similar_tickets = case["input"]["similar_tickets"]

        similar_details = "\n".join(
            f"- {t['key']}: {t['title']} ({t['summary'][:500]})"
            for t in similar_tickets
        ) or "NONE"

        input_prompt = f"""
        Current ticket: {current['key']} - {current['summary']}
        Description: {current['description']}
        Similar tickets: {similar_details}
        """

        structured_llm = llm_chat.with_structured_output(FinalAnalysisOutput)
        result = structured_llm.invoke([
            SystemMessage(content="You are an expert Jira analyst. Produce a structured analysis."),
            HumanMessage(content=input_prompt),
        ])

        judgment = llm_judge(
            input_text=input_prompt,
            output_text=str(result),
            rubric=case["rubric"],
        )

        avg_score = judgment.get("average_score", 0)
        assert avg_score >= PASS_THRESHOLD, (
            f"[{case['test_id']}] Quality score {avg_score:.1f} < {PASS_THRESHOLD}\n"
            f"Judgment: {judgment.get('explanation', 'N/A')}\n"
            f"Scores: {judgment.get('scores', {})}"
        )
