"""
Unit tests for the final_summary LLM call (_create_final_issue_summary).

Deterministic tests validate helper methods and output parsing.
LLM eval tests validate the real LLM output quality.
"""
import pytest
from unittest.mock import MagicMock

from tests.conftest import dataset_range
from utils.openai_jira_ticket_processing import (
    OpenAIJiraIssueLLMProcessor,
    FinalIssueSummeryOutput,
    LogAnalysisOutput,
    ErrorInLog,
)


# ===========================================================================
# Deterministic tests
# ===========================================================================
class TestFinalSummaryDeterministic:

    @pytest.mark.deterministic
    def test_final_issue_summary_output_schema(self):
        output = FinalIssueSummeryOutput(
            issue_summery="Test summary in one sentence.",
            main_issues=["Issue 1", "Issue 2"],
            likely_root_causes=["Cause 1"],
            comments="No relevant comments.",
        )
        assert len(output.main_issues) == 2
        assert isinstance(output.likely_root_causes, list)

    @pytest.mark.deterministic
    def test_parse_final_issue_summary_output_to_text(self):
        """Validate text formatting of final summary."""
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        summary = FinalIssueSummeryOutput(
            issue_summery="The API returns 500 errors.",
            main_issues=["Connection pool exhaustion"],
            likely_root_causes=["Too many concurrent connections"],
            comments="Team is investigating.",
        )
        log_summaries = [
            LogAnalysisOutput(
                log_filename="app.log",
                errors=[
                    ErrorInLog(
                        source_code_filename="pool.py",
                        error_lines="ERROR connection pool full",
                        exception_line="TimeoutError",
                        context="Pool exhausted",
                    )
                ],
            )
        ]
        text = processor._parse_final_issue_summary_output_to_text(summary, log_summaries)

        assert "Final Issue Summary" in text
        assert "Connection pool exhaustion" in text
        assert "Too many concurrent connections" in text
        assert "pool.py" in text
        assert "TimeoutError" in text
        assert "Team is investigating" in text

    @pytest.mark.deterministic
    def test_parse_final_summary_no_logs(self):
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        summary = FinalIssueSummeryOutput(
            issue_summery="Simple bug report.",
            main_issues=["UI glitch"],
            likely_root_causes=["unknown"],
            comments="No comments.",
        )
        text = processor._parse_final_issue_summary_output_to_text(summary, [])
        assert "No errors found in logs" in text

    @pytest.mark.deterministic
    def test_process_comments_truncation(self):
        """Comments should be truncated to MAX_WORDS_IN_COMMENTS."""
        from utils.jira_client import JiraComment

        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        long_comment = JiraComment(
            id="1",
            author_display_name="Test User",
            body=" ".join(["word"] * 500),  # 500 words
            created="2024-01-01T00:00:00.000+0000",
            updated="2024-01-01T00:00:00.000+0000",
        )
        result = processor._process_comments([long_comment])
        word_count = len(result.split())
        assert word_count <= 410  # MAX_WORDS_IN_COMMENTS (400) + author name words

    @pytest.mark.deterministic
    def test_process_comments_skips_automation(self):
        from utils.jira_client import JiraComment

        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        automation_comment = JiraComment(
            id="1",
            author_display_name="Automation for Jira",
            body="Automated transition",
            created="2024-01-01T00:00:00.000+0000",
            updated="2024-01-01T00:00:00.000+0000",
        )
        human_comment = JiraComment(
            id="2",
            author_display_name="Alice",
            body="I found the root cause.",
            created="2024-01-02T00:00:00.000+0000",
            updated="2024-01-02T00:00:00.000+0000",
        )
        result = processor._process_comments([automation_comment, human_comment])
        assert "Automation for Jira" not in result
        assert "Alice" in result

    @pytest.mark.deterministic
    def test_process_comments_empty(self):
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        result = processor._process_comments([])
        assert result == "No comments in the ticket."

    @pytest.mark.deterministic
    def test_process_related_issues(self):
        from utils.jira_client import JiraRelatedIssue

        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        related = [
            JiraRelatedIssue(key="TEST-10", relation_type="Duplicate", direction="outward"),
            JiraRelatedIssue(key="TEST-20", relation_type="Relates to", direction="inward"),
        ]
        result = processor._process_related_issues(related)
        assert "TEST-10" in result
        assert "TEST-20" in result
        assert "Duplicate" in result

    @pytest.mark.deterministic
    def test_process_related_issues_empty(self):
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        result = processor._process_related_issues([])
        assert "No related issues" in result


# ===========================================================================
# LLM eval tests
# ===========================================================================
class TestFinalSummaryLLMEval:

    @pytest.mark.llm_eval
    @pytest.mark.parametrize("case_idx", dataset_range("final_summary"))
    def test_final_summary(self, llm_processor, final_summary_dataset, make_jira_issue, case_idx):
        """Test final summary generation with real LLM."""
        case = final_summary_dataset[case_idx]
        jira_issue = make_jira_issue(case["input"]["jira_issue"])
        expected = case["expected"]

        result = llm_processor._create_final_issue_summary(
            jira_issue,
            case["input"]["comments_text"],
            case["input"]["summarized_attachments"],
            case["input"]["related_issues_text"],
        )

        assert isinstance(result, FinalIssueSummeryOutput), (
            f"[{case['test_id']}] Expected FinalIssueSummeryOutput, got {type(result)}"
        )

        # Summary length check
        sentences = [s.strip() for s in result.issue_summery.split(".") if s.strip()]
        assert len(sentences) <= expected["summary_max_sentences"] + 1, (
            f"[{case['test_id']}] Summary too long: {len(sentences)} sentences"
        )

        # Keyword checks on summary
        summary_lower = result.issue_summery.lower()
        for keyword in expected.get("summary_must_contain", []):
            assert keyword.lower() in summary_lower, (
                f"[{case['test_id']}] Expected '{keyword}' in summary: {result.issue_summery}"
            )

        # Main issues count
        assert len(result.main_issues) >= expected["min_main_issues"], (
            f"[{case['test_id']}] Expected >= {expected['min_main_issues']} main issues"
        )

        # Root causes count
        assert len(result.likely_root_causes) >= expected["min_root_causes"], (
            f"[{case['test_id']}] Expected >= {expected['min_root_causes']} root causes"
        )

        # Check main issues keywords
        all_issues_text = " ".join(result.main_issues).lower()
        for keyword in expected.get("main_issues_must_mention", []):
            assert keyword.lower() in all_issues_text, (
                f"[{case['test_id']}] Expected '{keyword}' in main_issues: {result.main_issues}"
            )
