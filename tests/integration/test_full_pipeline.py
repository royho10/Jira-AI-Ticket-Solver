"""
Integration tests for the full ticket processing pipeline.

These tests call the real LLM and validate the end-to-end flow:
  JiraIssue -> process_issue -> (summary_text, log_summaries)

They require Azure OpenAI env vars and mock only the JiraClient
(no real Jira API calls needed).
"""
import pytest
from unittest.mock import patch, MagicMock

from tests.conftest import dataset_range
from utils.openai_jira_ticket_processing import (
    OpenAIJiraIssueLLMProcessor,
    LogAnalysisOutput,
    FinalIssueSummeryOutput,
)


@pytest.mark.integration
class TestFullPipeline:

    @pytest.mark.parametrize("case_idx", dataset_range("integration"))
    def test_process_issue_end_to_end(
        self, integration_dataset, make_jira_issue, case_idx
    ):
        """Full pipeline: JiraIssue -> process_issue -> validate output."""
        case = integration_dataset[case_idx]
        jira_issue = make_jira_issue(case["input"]["jira_issue"])
        expected = case["expected"]

        processor = OpenAIJiraIssueLLMProcessor()

        # Mock JiraClient so we don't make real Jira API calls for attachments
        with patch.object(processor, "jira_client", MagicMock()):
            summary_text, log_summaries = processor.process_issue(jira_issue)

        # Validate output types
        if expected.get("final_output_is_valid_string"):
            assert isinstance(summary_text, str)
            assert len(summary_text) > 50, (
                f"[{case['test_id']}] Summary text too short: {len(summary_text)} chars"
            )

        if expected.get("log_summaries_is_list"):
            assert isinstance(log_summaries, list)

        # Check summary content keywords
        summary_lower = summary_text.lower()
        for keyword in expected.get("summary_must_contain", []):
            assert keyword.lower() in summary_lower, (
                f"[{case['test_id']}] Expected '{keyword}' in summary output.\n"
                f"Summary: {summary_text[:500]}"
            )

        # Check error extraction from description
        if expected.get("must_find_errors_in_description"):
            assert len(log_summaries) > 0, (
                f"[{case['test_id']}] Expected errors extracted from description but got none"
            )

            all_errors_text = " ".join(
                str(ls) for ls in log_summaries
            ).lower()
            for keyword in expected.get("errors_must_mention", []):
                assert keyword.lower() in all_errors_text, (
                    f"[{case['test_id']}] Expected '{keyword}' in log summaries"
                )

    @pytest.mark.parametrize("case_idx", dataset_range("integration"))
    def test_pipeline_output_structure(
        self, integration_dataset, make_jira_issue, case_idx
    ):
        """Validate that pipeline output has the expected sections."""
        case = integration_dataset[case_idx]
        jira_issue = make_jira_issue(case["input"]["jira_issue"])

        processor = OpenAIJiraIssueLLMProcessor()

        with patch.object(processor, "jira_client", MagicMock()):
            summary_text, log_summaries = processor.process_issue(jira_issue)

        # Validate the summary text contains expected sections
        assert "Final Issue Summary" in summary_text
        assert "Main Issues" in summary_text
        assert "Root Causes" in summary_text
        assert "Errors Found In Logs" in summary_text or "No errors found" in summary_text

    def test_pipeline_with_no_description(self, make_jira_issue):
        """Pipeline should handle tickets with no description gracefully."""
        jira_issue = make_jira_issue({
            "key": "TEST-EMPTY",
            "summary": "Empty ticket",
            "description": None,
            "labels": [],
            "issue_type": "Bug",
            "priority": "Low",
            "status": "Open",
        })

        processor = OpenAIJiraIssueLLMProcessor()

        with patch.object(processor, "jira_client", MagicMock()):
            summary_text, log_summaries = processor.process_issue(jira_issue)

        assert isinstance(summary_text, str)
        assert isinstance(log_summaries, list)
        # No description means no errors from description
        # (description errors come from _extract_errors_from_description which returns None for empty)

    def test_pipeline_with_empty_attachments_and_comments(self, make_jira_issue):
        """Pipeline should work with a minimal ticket."""
        jira_issue = make_jira_issue({
            "key": "TEST-MINIMAL",
            "summary": "Minimal test ticket with basic info",
            "description": "This is a simple bug report with no attachments or special details. Just a basic ticket.",
            "labels": [],
            "issue_type": "Task",
            "priority": "Minor",
            "status": "Open",
        })

        processor = OpenAIJiraIssueLLMProcessor()

        with patch.object(processor, "jira_client", MagicMock()):
            summary_text, log_summaries = processor.process_issue(jira_issue)

        assert isinstance(summary_text, str)
        assert len(summary_text) > 20
