"""
Unit tests for error_extraction_from_description LLM call.

Deterministic tests validate structure and basic expectations without LLM calls.
LLM eval tests call the real LLM and validate output quality.
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
# Deterministic tests (no LLM calls)
# ===========================================================================
class TestErrorExtractionDeterministic:
    """Tests that validate business logic without calling the LLM."""

    @pytest.mark.deterministic
    def test_short_description_returns_none(self):
        """Descriptions under 20 chars should be skipped entirely."""
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        result = processor._extract_errors_from_description("Fix bug")
        assert result is None

    @pytest.mark.deterministic
    def test_empty_description_returns_none(self):
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        result = processor._extract_errors_from_description("")
        assert result is None

    @pytest.mark.deterministic
    def test_none_description_returns_none(self):
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        result = processor._extract_errors_from_description(None)
        assert result is None

    @pytest.mark.deterministic
    def test_log_analysis_output_schema_valid(self):
        """Validate that LogAnalysisOutput Pydantic model works correctly."""
        output = LogAnalysisOutput(
            log_filename="test.log",
            errors=[
                ErrorInLog(
                    source_code_filename="main.py",
                    error_lines="2024-01-01 ERROR Something failed",
                    exception_line="ValueError: invalid value",
                    context="Something went wrong",
                )
            ],
        )
        assert output.log_filename == "test.log"
        assert len(output.errors) == 1
        assert output.errors[0].source_code_filename == "main.py"

    @pytest.mark.deterministic
    def test_log_analysis_output_requires_at_least_one_error(self):
        """LogAnalysisOutput.errors has min_length=1."""
        with pytest.raises(Exception):
            LogAnalysisOutput(log_filename="test.log", errors=[])

    @pytest.mark.deterministic
    def test_mock_llm_returns_structured_output(self):
        """Verify the method correctly processes a mocked LLM response."""
        mock_response = LogAnalysisOutput(
            log_filename="placeholder",
            errors=[
                ErrorInLog(
                    source_code_filename="service.py",
                    error_lines="ERROR: Connection refused",
                    exception_line="ConnectionError: refused",
                    context="Database connection failed",
                )
            ],
        )
        processor = OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)
        processor.llm_deployment = "test-deployment"

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_response
        mock_llm.with_structured_output.return_value = mock_structured

        with patch.object(processor, "_get_llm", return_value=mock_llm):
            result = processor._extract_errors_from_description(
                "The app crashes with ERROR: Connection refused when connecting to the database server."
            )

        assert result is not None
        assert result.log_filename == "ticket_description"
        assert len(result.errors) == 1
        assert "Connection" in result.errors[0].error_lines


# ===========================================================================
# LLM eval tests (real LLM calls)
# ===========================================================================
class TestErrorExtractionLLMEval:
    """Tests that call the real LLM and validate output quality."""

    @pytest.mark.llm_eval
    @pytest.mark.parametrize("case_idx", dataset_range("error_extraction"))
    def test_error_extraction(self, llm_processor, error_extraction_dataset, case_idx):
        """Test error extraction with real LLM against dataset cases."""
        case = error_extraction_dataset[case_idx]
        description = case["input"]["description"]
        expected = case["expected"]

        result = llm_processor._extract_errors_from_description(description)

        if not expected["should_find_errors"]:
            def _is_blank_error(e):
                return not e.error_lines and not e.exception_line and not e.context

            has_no_real_errors = (
                result is None
                or (hasattr(result, "errors") and len(result.errors) == 0)
                or (hasattr(result, "errors") and all(_is_blank_error(e) for e in result.errors))
            )
            assert has_no_real_errors, (
                f"[{case['test_id']}] Expected no errors but got: {result}"
            )
            return

        # Should find errors
        assert result is not None, f"[{case['test_id']}] Expected errors but got None"
        assert isinstance(result, LogAnalysisOutput)
        assert len(result.errors) >= expected["min_errors"], (
            f"[{case['test_id']}] Expected at least {expected['min_errors']} errors, got {len(result.errors)}"
        )
        assert result.log_filename == expected["log_filename"]

        # Check keywords in the full output text
        output_text = str(result)
        for keyword in expected.get("must_contain_keywords", []):
            assert keyword.lower() in output_text.lower(), (
                f"[{case['test_id']}] Expected keyword '{keyword}' in output but not found.\nOutput: {output_text}"
            )

        for keyword in expected.get("must_not_contain", []):
            assert keyword.lower() not in output_text.lower(), (
                f"[{case['test_id']}] Unexpected keyword '{keyword}' found in output.\nOutput: {output_text}"
            )
