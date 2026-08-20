"""Tests for the LLM call log.

The log records prompts and responses verbatim, which is exactly what makes it
useful locally and unacceptable on the shared Remote MCP Server: those prompts
carry raw, unsanitized ticket text belonging to whichever user made the request.
`disable()` is the switch the server flips, so it gets a test.
"""
import pytest

from utils import llm_logger


@pytest.fixture
def restore_logger_state():
    """Leave the module as it was -- it holds process-wide state that the eval
    tests in this session depend on."""
    was_enabled = llm_logger.is_enabled()
    yield
    llm_logger._enabled = was_enabled


@pytest.mark.deterministic
def test_logging_is_on_by_default():
    """Local runs (Streamlit, the indexer) rely on the log being written."""
    assert llm_logger.is_enabled() is True


@pytest.mark.deterministic
def test_disable_stops_prompts_reaching_disk(restore_logger_state, monkeypatch):
    opened = []
    monkeypatch.setattr(llm_logger, "_ensure_log_file", lambda: opened.append(True))

    llm_logger.disable()
    llm_logger.log_llm_call(
        "pii_sanitization",
        "gpt-5-nano",
        [("human", "customer contact is sarah@acmecorp.com")],
        "redacted",
        duration_ms=5,
    )

    assert llm_logger.is_enabled() is False
    assert opened == [], "a log file was opened after logging was disabled"


@pytest.mark.deterministic
def test_disable_also_silences_the_run_summary(restore_logger_state, monkeypatch):
    opened = []
    monkeypatch.setattr(llm_logger, "_ensure_log_file", lambda: opened.append(True))

    llm_logger.disable()
    llm_logger.log_run_summary()

    assert opened == []
