"""
Unit tests for log_parsing (log filtering + LLM summarization).

Deterministic tests validate the _filter_noise_from_logs logic.
LLM eval tests validate the _summarize_log LLM call.
"""
import pytest
from unittest.mock import patch, MagicMock

from tests.conftest import dataset_range
from utils.openai_jira_ticket_processing import (
    OpenAIJiraIssueLLMProcessor,
    LogAnalysisOutput,
    ErrorInLog,
)


# ===========================================================================
# Deterministic tests: _filter_noise_from_logs (no LLM)
# ===========================================================================
class TestLogFilteringDeterministic:
    """Tests for the deterministic log filtering logic."""

    @pytest.fixture
    def processor(self):
        return OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)

    @pytest.mark.deterministic
    def test_filters_error_lines_with_context(self, processor):
        """Should extract ERROR lines plus surrounding context."""
        log_text = "\n".join(
            [f"2024-01-01 10:00:{i:02d} INFO normal line {i}" for i in range(30)]
            + ["2024-01-01 10:00:30 ERROR Something bad happened"]
            + [f"2024-01-01 10:00:{31+i:02d} INFO normal line {31+i}" for i in range(20)]
        )
        result = processor._filter_noise_from_logs([(log_text, "test.log")])
        assert len(result) == 1
        filtered_text, filename = result[0]
        assert "ERROR" in filtered_text
        assert filename == "test.log"

    @pytest.mark.deterministic
    def test_empty_logs_returns_empty(self, processor):
        result = processor._filter_noise_from_logs([])
        assert result == []

    @pytest.mark.deterministic
    def test_no_error_lines_returns_empty(self, processor):
        """Logs without ERROR should be filtered out entirely."""
        log_text = "2024-01-01 INFO all good\n2024-01-01 WARN minor warning\n2024-01-01 DEBUG debug info"
        result = processor._filter_noise_from_logs([(log_text, "clean.log")])
        assert result == []

    @pytest.mark.deterministic
    def test_only_uppercase_error_matched(self, processor):
        """Only uppercase 'ERROR' should be matched, not 'error' or 'Error'."""
        log_text = "2024-01-01 error lowercase\n2024-01-01 Error mixed\n2024-01-01 ERROR uppercase"
        result = processor._filter_noise_from_logs([(log_text, "test.log")])
        assert len(result) == 1
        assert "ERROR uppercase" in result[0][0]

    @pytest.mark.deterministic
    def test_multiple_log_files(self, processor):
        """Should process multiple log files independently."""
        log1 = "2024-01-01 ERROR error in file 1"
        log2 = "2024-01-01 INFO no errors here"
        log3 = "2024-01-01 ERROR error in file 3"
        result = processor._filter_noise_from_logs([
            (log1, "file1.log"),
            (log2, "file2.log"),
            (log3, "file3.log"),
        ])
        assert len(result) == 2
        filenames = [r[1] for r in result]
        assert "file1.log" in filenames
        assert "file3.log" in filenames

    @pytest.mark.deterministic
    def test_truncates_at_max_lines(self, processor):
        """Should respect MAX_LINES_PER_LOG limit."""
        # Create a log with many ERROR lines
        lines = [f"2024-01-01 ERROR error number {i}" for i in range(200)]
        log_text = "\n".join(lines)
        result = processor._filter_noise_from_logs([(log_text, "big.log")])
        assert len(result) == 1
        # The filtered output should be capped
        output_lines = result[0][0].split("\n")
        assert len(output_lines) <= 55  # MAX_LINES_PER_LOG (50) + header + truncation msg


# ===========================================================================
# Deterministic tests: _filter_out_unwanted_logs
# ===========================================================================
class TestFilterOutUnwantedLogs:

    @pytest.fixture
    def processor(self):
        return OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)

    @pytest.mark.deterministic
    def test_keeps_errors_with_ERROR_keyword(self, processor):
        summary = LogAnalysisOutput(
            log_filename="test.log",
            errors=[
                ErrorInLog(
                    source_code_filename=None,
                    error_lines="2024-01-01 ERROR connection failed",
                    exception_line=None,
                    context="Connection error",
                )
            ],
        )
        result = processor._filter_out_unwanted_logs(summary)
        assert result is not None
        assert len(result.errors) == 1

    @pytest.mark.deterministic
    def test_removes_errors_without_ERROR_keyword(self, processor):
        summary = LogAnalysisOutput(
            log_filename="test.log",
            errors=[
                ErrorInLog(
                    source_code_filename=None,
                    error_lines="Something went wrong but no error keyword",
                    exception_line=None,
                    context="Some issue",
                )
            ],
        )
        result = processor._filter_out_unwanted_logs(summary)
        assert result is not None
        assert len(result.errors) == 0

    @pytest.mark.deterministic
    def test_none_input_returns_none(self, processor):
        result = processor._filter_out_unwanted_logs(None)
        assert result is None


# ===========================================================================
# LLM eval tests: _summarize_log
# ===========================================================================
class TestLogParsingLLMEval:

    @pytest.mark.llm_eval
    @pytest.mark.parametrize("case_idx", dataset_range("log_parsing"))
    def test_log_parsing(self, llm_processor, log_parsing_dataset, case_idx):
        """Test log parsing with real LLM against dataset cases."""
        case = log_parsing_dataset[case_idx]
        log_text = case["input"]["log_text"]
        filename = case["input"]["filename"]
        expected = case["expected"]

        result = llm_processor._summarize_log((log_text, filename))

        assert result is not None, f"[{case['test_id']}] LLM returned None"
        assert isinstance(result, LogAnalysisOutput)
        assert len(result.errors) >= expected["min_errors"], (
            f"[{case['test_id']}] Expected >= {expected['min_errors']} errors, got {len(result.errors)}"
        )

        if "max_errors" in expected:
            assert len(result.errors) <= expected["max_errors"], (
                f"[{case['test_id']}] Expected <= {expected['max_errors']} errors, got {len(result.errors)}"
            )

        # Validate error lines contain ERROR keyword
        for error in result.errors:
            for kw in expected.get("error_lines_must_contain", []):
                assert kw in error.error_lines, (
                    f"[{case['test_id']}] Expected '{kw}' in error_lines: {error.error_lines}"
                )

        # Check overall output keywords
        output_text = str(result)
        for keyword in expected.get("must_contain_keywords", []):
            assert keyword.lower() in output_text.lower(), (
                f"[{case['test_id']}] Expected keyword '{keyword}' not found in output"
            )
