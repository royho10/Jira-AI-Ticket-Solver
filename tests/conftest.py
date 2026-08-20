import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time

from utils.llm_logger import reset_run_stats, get_run_stats, log_llm_call


# ---------------------------------------------------------------------------
# Pytest hooks: Token usage reporting
# ---------------------------------------------------------------------------
def pytest_sessionstart(session):
    """Reset LLM token counters at the start of each test session."""
    reset_run_stats()


def pytest_sessionfinish(session, exitstatus):
    """Print Azure AI token usage summary at the end of the test run."""
    stats = get_run_stats()
    if stats["estimated_total_tokens"] == 0:
        return  # No LLM calls were made (e.g. deterministic-only run)

    lines = [
        "",
        "=" * 50,
        "  Azure AI Token Usage",
        "=" * 50,
        f"  Estimated input tokens:  ~{stats['estimated_input_tokens']:,}",
        f"  Estimated output tokens: ~{stats['estimated_output_tokens']:,}",
        f"  Estimated total tokens:  ~{stats['estimated_total_tokens']:,}",
        f"  Total LLM call duration: {stats['total_duration_ms']:,}ms "
        f"({stats['total_duration_ms'] / 1000:.1f}s)",
    ]
    if stats["wall_time_seconds"] is not None:
        lines.append(f"  Wall time:               {stats['wall_time_seconds']:.1f}s")
    lines.append("=" * 50)
    print("\n".join(lines), flush=True)

DATASETS_DIR = Path(__file__).parent / "datasets"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_dataset(name: str):
    path = DATASETS_DIR / f"{name}.json"
    with open(path, "r") as f:
        return json.load(f)


def dataset_range(name: str) -> range:
    """Return range(len(dataset)) for use in @pytest.mark.parametrize."""
    path = DATASETS_DIR / f"{name}.json"
    with open(path, "r") as f:
        return range(len(json.load(f)))


# ---------------------------------------------------------------------------
# Fixtures: Datasets
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def error_extraction_dataset():
    return load_dataset("error_extraction")


@pytest.fixture(scope="session")
def log_parsing_dataset():
    return load_dataset("log_parsing")


@pytest.fixture(scope="session")
def final_summary_dataset():
    return load_dataset("final_summary")


@pytest.fixture(scope="session")
def intent_classification_dataset():
    return load_dataset("intent_classification")


@pytest.fixture(scope="session")
def reranking_dataset():
    return load_dataset("reranking")


@pytest.fixture(scope="session")
def final_analysis_dataset():
    return load_dataset("final_analysis")


@pytest.fixture(scope="session")
def integration_dataset():
    return load_dataset("integration")


@pytest.fixture(scope="session")
def pii_sanitization_dataset():
    return load_dataset("pii_sanitization")


# ---------------------------------------------------------------------------
# Fixtures: LLM Processor (real, for llm_eval / integration tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def llm_processor():
    """Create a real OpenAIJiraIssueLLMProcessor for LLM eval tests.
    Requires Azure OpenAI env vars to be set."""
    from utils.openai_jira_ticket_processing import OpenAIJiraIssueLLMProcessor
    return OpenAIJiraIssueLLMProcessor()


@pytest.fixture(scope="session")
def llm_chat():
    """Create a real AzureChatOpenAI for LLM eval tests."""
    from langchain_openai import AzureChatOpenAI
    from langchain_core.callbacks import BaseCallbackHandler
    from config.settings import (
        AZURE_OPENAI_LLM_DEPLOYMENT,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_TEMPERATURE,
    )

    class _TokenLoggingCallback(BaseCallbackHandler):
        """Logs every LLM call via llm_logger for token tracking in tests."""

        def __init__(self):
            super().__init__()
            self._start_times = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self._start_times[run_id] = time.time()

        def on_llm_end(self, response, *, run_id, **kwargs):
            duration_ms = None
            start = self._start_times.pop(run_id, None)
            if start is not None:
                duration_ms = int((time.time() - start) * 1000)

            generation = response.generations[0][0] if response.generations else None
            output_text = generation.text if generation else ""
            log_llm_call("test_llm_call", AZURE_OPENAI_LLM_DEPLOYMENT, [], output_text, duration_ms)

    return AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_LLM_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        temperature=AZURE_OPENAI_TEMPERATURE,
        callbacks=[_TokenLoggingCallback()],
    )


# ---------------------------------------------------------------------------
# Fixtures: Mock Jira data builders
# ---------------------------------------------------------------------------
@pytest.fixture
def make_jira_issue():
    """Factory fixture to create JiraIssue objects from dict data."""
    from utils.jira_client import JiraIssue, JiraComment, JiraAttachment, JiraRelatedIssue

    def _make(data: dict) -> JiraIssue:
        comments = []
        for c in data.get("comments", []):
            comments.append(JiraComment(
                id=c.get("id", "1"),
                author_display_name=c.get("author_display_name", "Test User"),
                body=c.get("body", ""),
                created=c.get("created", "2024-01-01T00:00:00.000+0000"),
                updated=c.get("created", "2024-01-01T00:00:00.000+0000"),
            ))

        attachments = []
        for a in data.get("attachments", []):
            attachments.append(JiraAttachment(
                id=a.get("id", "1"),
                filename=a.get("filename", "test.log"),
                size=a.get("size", 1024),
                mime_type=a.get("mime_type", "text/plain"),
                content_url=a.get("content_url", ""),
                created=a.get("created", "2024-01-01T00:00:00.000+0000"),
                author_display_name=a.get("author_display_name", "Test User"),
            ))

        related_issues = []
        for r in data.get("related_issues", []):
            related_issues.append(JiraRelatedIssue(
                key=r.get("key", ""),
                relation_type=r.get("relation_type", "Relates to"),
                direction=r.get("direction", "outward"),
            ))

        return JiraIssue(
            key=data.get("key", "TEST-1"),
            summary=data.get("summary", "Test summary"),
            description=data.get("description"),
            labels=data.get("labels", []),
            comments=comments,
            attachments=attachments,
            related_issues=related_issues,
            priority=data.get("priority"),
            issue_type=data.get("issue_type"),
            components=data.get("components"),
            created=data.get("created", "2024-01-01T00:00:00.000+0000"),
            status=data.get("status"),
        )

    return _make


# ---------------------------------------------------------------------------
# Fixtures: LLM-as-Judge helper
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def llm_judge(llm_chat):
    """Returns a function that uses an LLM to evaluate output against a rubric."""
    from langchain_core.messages import SystemMessage, HumanMessage

    def _judge(input_text: str, output_text: str, rubric: dict, max_score: int = 5) -> dict:
        rubric_text = "\n".join(f"- {k}: {v}" for k, v in rubric.items())

        judge_prompt = f"""You are an evaluation judge for an LLM application.
Score the OUTPUT based on each rubric criterion on a scale of 1-{max_score}.

INPUT given to the system:
{input_text[:2000]}

OUTPUT produced by the system:
{output_text[:2000]}

RUBRIC (evaluate each criterion):
{rubric_text}

Respond in JSON format:
{{
  "scores": {{"criterion_name": score, ...}},
  "average_score": <float>,
  "explanation": "brief explanation"
}}

Return ONLY valid JSON, no markdown fences."""

        messages = [
            SystemMessage(content="You are a strict but fair evaluation judge. Score objectively."),
            HumanMessage(content=judge_prompt),
        ]
        result = llm_chat.invoke(messages)
        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return {"scores": {}, "average_score": 0, "explanation": f"Failed to parse: {result.content[:200]}"}

    return _judge
