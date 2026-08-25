"""Tests for the Remote MCP Server, driven through its HTTP endpoint.

This is the primary test seam for the whole feature: protocol handling, auth,
analysis orchestration, PII sanitization and response formatting are all
exercised end to end through `POST /mcp`, with Jira, the LLM and Weaviate mocked
behind the endpoint.
"""
import json
import logging

import pytest
from fastapi.testclient import TestClient

from core.pii_sanitizer import SanitizationError
from core.ticket_analyzer import ErrorLogHighlight, SimilarTicket, TicketAnalysis
from server.auth import AuthError, AuthErrorCode
from server.mcp_server import create_app

USER_EMAIL = "engineer@guardicore.com"
USER_TOKEN = "super-secret-user-token"
AUTH_HEADERS = {"X-Jira-Email": USER_EMAIL, "X-Jira-Token": USER_TOKEN}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_analysis(**overrides) -> TicketAnalysis:
    data = dict(
        ticket_key="GC-100",
        ticket_title="Collector crashes on startup",
        ticket_status="Open",
        ticket_priority="High",
        ticket_created="2024-01-01T00:00:00.000+0000",
        ticket_labels=["collector"],
        description="The collector crashes with a NullPointerException",
        processed_summary="Processed summary text",
        ticket_summary="The collector fails to start after an upgrade",
        key_issues=["Collector process exits immediately"],
        root_causes=["Null label collection in PolicyCompiler"],
        suggested_solutions=["Return an empty list instead of null"],
        important_notes=["Affects 4.2.1 and later"],
        error_log_highlights=[ErrorLogHighlight(
            log_filename="collector.log",
            context="Collector failed to start",
            error_lines="ERROR NullPointerException at Collector.java:42",
            exception_line="java.lang.NullPointerException",
            source_code_filename="Collector.java",
        )],
        similar_tickets=[SimilarTicket(
            key="GC-50",
            title="Older crash on startup",
            summary="Same crash after upgrading",
            status="Done",
            resolution="Fixed",
            issue_type="Bug",
            relevance_score=9,
            similarity_reason="Identical NPE in Collector.java",
        )],
    )
    data.update(overrides)
    return TicketAnalysis(**data)


class FakeAnalyzer:
    def __init__(self, analysis=None, error=None):
        self._analysis = analysis if analysis is not None else make_analysis()
        self._error = error
        self.calls = []

    def analyze(self, ticket_key, on_progress=None):
        self.calls.append(ticket_key)
        if self._error:
            raise self._error
        return self._analysis


class FakeSanitizer:
    def __init__(self, transform=None, error=None):
        self._transform = transform
        self._error = error
        self.calls = []

    def sanitize(self, analysis):
        self.calls.append(analysis)
        if self._error:
            raise self._error
        return self._transform(analysis) if self._transform else analysis


class FakeAccessChecker:
    def __init__(self, error=None):
        self._error = error
        self.calls = []

    def verify_access(self, email, token, ticket_key):
        self.calls.append({"email": email, "token": token, "ticket_key": ticket_key})
        if self._error:
            raise self._error


def build_client(analyzer=None, sanitizer=None, access_checker=None, **kwargs):
    analyzer = analyzer or FakeAnalyzer()
    sanitizer = sanitizer or FakeSanitizer()
    access_checker = access_checker or FakeAccessChecker()
    app = create_app(
        analyzer=analyzer,
        sanitizer=sanitizer,
        access_checker=access_checker,
        **kwargs,
    )
    return TestClient(app), analyzer, sanitizer, access_checker


def rpc(method, params=None, request_id=1):
    body = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        body["id"] = request_id
    if params is not None:
        body["params"] = params
    return body


def call_tool(client, ticket_key="GC-100", headers=AUTH_HEADERS, name="analyze_ticket"):
    return client.post(
        "/mcp",
        json=rpc("tools/call", {"name": name, "arguments": {"ticket_key": ticket_key}}),
        headers=headers,
    )


def tool_text(response) -> str:
    result = response.json()["result"]
    return "\n".join(block["text"] for block in result["content"])


def assert_tool_error(response) -> str:
    """Auth and analysis failures come back as an `isError` tool result at HTTP
    200 -- a 401 on the MCP endpoint would send the client into OAuth discovery.
    Returns the message text so callers can assert on it."""
    assert response.status_code == 200
    result = response.json()["result"]
    assert result.get("isError") is True
    return "\n".join(block["text"] for block in result["content"])


def structured_payload(response) -> dict:
    text = tool_text(response)
    start = text.index("<ticket_analysis>") + len("<ticket_analysis>")
    end = text.index("</ticket_analysis>")
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Health and protocol handshake
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_health_endpoint_reports_ok():
    client, _, _, _ = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.deterministic
def test_health_endpoint_needs_no_credentials():
    client, _, _, _ = build_client()

    assert client.get("/health").status_code == 200


@pytest.mark.deterministic
def test_initialize_advertises_tool_support():
    client, _, _, _ = build_client()

    response = client.post("/mcp", json=rpc("initialize", {"protocolVersion": "2025-06-18"}))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["protocolVersion"]
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"]
    assert result["serverInfo"]["version"]


@pytest.mark.deterministic
def test_initialized_notification_is_accepted_without_a_response_body():
    client, _, _, _ = build_client()

    response = client.post("/mcp", json=rpc("notifications/initialized", request_id=None))

    assert response.status_code == 202
    assert not response.content


@pytest.mark.deterministic
def test_tools_list_exposes_analyze_ticket():
    client, _, _, _ = build_client()

    response = client.post("/mcp", json=rpc("tools/list"))

    tools = response.json()["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["analyze_ticket"]
    schema = tools[0]["inputSchema"]
    assert "ticket_key" in schema["properties"]
    assert schema["required"] == ["ticket_key"]
    assert tools[0]["description"]


@pytest.mark.deterministic
def test_get_mcp_opens_a_server_sent_event_stream():
    client, _, _, _ = build_client(sse_keepalive_seconds=0.01, sse_stream_seconds=0.05)

    with client.stream("GET", "/mcp", headers=AUTH_HEADERS) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert next(response.iter_raw())


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_tool_call_returns_markdown_and_structured_json():
    client, analyzer, sanitizer, _ = build_client()

    response = call_tool(client)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert "error" not in body
    assert body["result"].get("isError") in (None, False)

    text = tool_text(response)
    assert "GC-100" in text
    assert "Collector crashes on startup" in text
    assert "The collector fails to start after an upgrade" in text
    assert "Collector process exits immediately" in text
    assert "Return an empty list instead of null" in text
    assert "Affects 4.2.1 and later" in text
    assert "GC-50" in text
    assert "Identical NPE in Collector.java" in text
    assert "collector.log" in text

    payload = structured_payload(response)
    assert payload["ticket_key"] == "GC-100"
    assert [t["key"] for t in payload["similar_tickets"]] == ["GC-50"]
    assert analyzer.calls == ["GC-100"]


@pytest.mark.deterministic
def test_the_response_is_built_from_the_sanitized_analysis():
    """The formatter must never see the raw analysis -- that is the whole point
    of the sanitization pass."""
    def redact(analysis):
        return analysis.model_copy(update={"description": "[REDACTED_DOMAIN]"})

    analyzer = FakeAnalyzer(make_analysis(description="host db-prod-01.acmecorp.internal failed"))
    sanitizer = FakeSanitizer(transform=redact)
    client, _, _, _ = build_client(analyzer=analyzer, sanitizer=sanitizer)

    response = call_tool(client)

    assert sanitizer.calls, "the analysis was never sanitized"
    assert "db-prod-01.acmecorp.internal" not in response.text
    assert "[REDACTED_DOMAIN]" in response.text


@pytest.mark.deterministic
def test_an_analysis_without_similar_tickets_or_logs_still_renders():
    analyzer = FakeAnalyzer(make_analysis(similar_tickets=[], error_log_highlights=[]))
    client, _, _, _ = build_client(analyzer=analyzer)

    response = call_tool(client)

    assert response.status_code == 200
    text = tool_text(response)
    assert "GC-100" in text
    assert structured_payload(response)["similar_tickets"] == []


# ---------------------------------------------------------------------------
# Authentication and authorization
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_missing_credentials_are_rejected():
    client, analyzer, _, _ = build_client()

    response = call_tool(client, headers={})

    assert assert_tool_error(response)
    assert analyzer.calls == []


@pytest.mark.deterministic
@pytest.mark.parametrize("headers", [
    {"X-Jira-Email": USER_EMAIL},
    {"X-Jira-Token": USER_TOKEN},
    {"X-Jira-Email": USER_EMAIL, "X-Jira-Token": "   "},
    {"X-Jira-Email": "  ", "X-Jira-Token": USER_TOKEN},
])
def test_incomplete_credentials_are_rejected(headers):
    client, analyzer, _, _ = build_client()

    response = call_tool(client, headers=headers)

    assert_tool_error(response)
    assert analyzer.calls == []


@pytest.mark.deterministic
def test_invalid_jira_credentials_are_rejected():
    checker = FakeAccessChecker(
        error=AuthError(AuthErrorCode.INVALID_CREDENTIALS, "Jira rejected the token")
    )
    client, analyzer, _, _ = build_client(access_checker=checker)

    response = call_tool(client)

    assert_tool_error(response)
    assert analyzer.calls == []


@pytest.mark.deterministic
def test_a_user_without_access_to_the_ticket_is_refused():
    checker = FakeAccessChecker(
        error=AuthError(AuthErrorCode.ACCESS_DENIED, "No access to GC-100")
    )
    client, analyzer, _, _ = build_client(access_checker=checker)

    response = call_tool(client)

    assert "GC-100" in assert_tool_error(response)
    assert analyzer.calls == []


@pytest.mark.deterministic
def test_the_access_check_uses_the_user_token_and_the_analysis_does_not():
    """Dual-token: the user's token proves access, the service account fetches."""
    client, analyzer, _, checker = build_client()

    call_tool(client)

    assert checker.calls == [
        {"email": USER_EMAIL, "token": USER_TOKEN, "ticket_key": "GC-100"}
    ]
    assert analyzer.calls == ["GC-100"]


@pytest.mark.deterministic
def test_the_user_token_never_reaches_the_response_or_the_logs(caplog):
    client, _, _, _ = build_client()

    with caplog.at_level(logging.DEBUG):
        response = call_tool(client)

    assert USER_TOKEN not in response.text
    logged = "\n".join(
        [record.getMessage() for record in caplog.records]
        + [str(record.args) for record in caplog.records]
    )
    assert USER_TOKEN not in logged


@pytest.mark.deterministic
def test_a_failed_access_check_does_not_log_the_token(caplog):
    checker = FakeAccessChecker(
        error=AuthError(AuthErrorCode.ACCESS_DENIED, "No access to GC-100")
    )
    client, _, _, _ = build_client(access_checker=checker)

    with caplog.at_level(logging.DEBUG):
        response = call_tool(client)

    assert_tool_error(response)
    assert USER_TOKEN not in "\n".join(r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Origin validation (DNS rebinding)
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_a_request_from_an_unknown_origin_is_refused():
    client, analyzer, _, _ = build_client(allowed_origins=["https://claude.ai"])

    response = call_tool(
        client, headers={**AUTH_HEADERS, "Origin": "http://evil.example.com"}
    )

    assert response.status_code == 403
    assert analyzer.calls == []


@pytest.mark.deterministic
def test_a_request_from_an_allowed_origin_is_accepted():
    client, analyzer, _, _ = build_client(allowed_origins=["https://claude.ai"])

    response = call_tool(client, headers={**AUTH_HEADERS, "Origin": "https://claude.ai"})

    assert response.status_code == 200
    assert analyzer.calls == ["GC-100"]


@pytest.mark.deterministic
def test_a_request_without_an_origin_header_is_accepted():
    """Claude Code is not a browser and sends no Origin -- only a stated Origin
    has to be on the allowlist."""
    client, analyzer, _, _ = build_client(allowed_origins=["https://claude.ai"])

    response = call_tool(client)

    assert response.status_code == 200
    assert analyzer.calls == ["GC-100"]


@pytest.mark.deterministic
def test_any_stated_origin_is_refused_when_no_allowlist_is_configured():
    client, analyzer, _, _ = build_client(allowed_origins=[])

    response = call_tool(client, headers={**AUTH_HEADERS, "Origin": "http://evil.example.com"})

    assert response.status_code == 403
    assert analyzer.calls == []


# ---------------------------------------------------------------------------
# Protocol errors
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_a_body_that_is_not_json_is_a_parse_error():
    client, _, _, _ = build_client()

    response = client.post("/mcp", content=b"not json at all", headers=AUTH_HEADERS)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


@pytest.mark.deterministic
def test_a_body_without_a_method_is_an_invalid_request():
    client, _, _, _ = build_client()

    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1}, headers=AUTH_HEADERS)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600


@pytest.mark.deterministic
def test_an_unknown_method_is_a_method_not_found_error():
    client, _, _, _ = build_client()

    response = client.post("/mcp", json=rpc("tools/teleport"), headers=AUTH_HEADERS)

    body = response.json()
    assert body["error"]["code"] == -32601
    assert body["id"] == 1


@pytest.mark.deterministic
def test_calling_an_unknown_tool_is_an_invalid_params_error():
    client, analyzer, _, _ = build_client()

    response = call_tool(client, name="delete_everything")

    assert response.json()["error"]["code"] == -32602
    assert analyzer.calls == []


@pytest.mark.deterministic
def test_a_tool_call_without_a_ticket_key_is_an_invalid_params_error():
    client, analyzer, _, _ = build_client()

    response = client.post(
        "/mcp",
        json=rpc("tools/call", {"name": "analyze_ticket", "arguments": {}}),
        headers=AUTH_HEADERS,
    )

    assert response.json()["error"]["code"] == -32602
    assert analyzer.calls == []


# ---------------------------------------------------------------------------
# Failures during analysis
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_an_analysis_failure_is_reported_as_a_tool_error():
    analyzer = FakeAnalyzer(error=RuntimeError("weaviate unreachable"))
    client, _, _, _ = build_client(analyzer=analyzer)

    response = call_tool(client)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert "GC-100" in tool_text(response)


@pytest.mark.deterministic
def test_a_sanitization_failure_returns_an_error_and_no_analysis_text():
    """Fail closed: if redaction could not run, nothing from the analysis may
    appear in the response."""
    analyzer = FakeAnalyzer(make_analysis(description="host db-prod-01.acmecorp.internal"))
    sanitizer = FakeSanitizer(error=SanitizationError("redaction failed"))
    client, _, _, _ = build_client(analyzer=analyzer, sanitizer=sanitizer)

    response = call_tool(client)

    assert response.json()["result"]["isError"] is True
    assert "db-prod-01.acmecorp.internal" not in response.text
    assert "Collector crashes on startup" not in response.text


# ---------------------------------------------------------------------------
# Usage logging
# ---------------------------------------------------------------------------
@pytest.mark.deterministic
def test_successful_calls_are_logged_per_user(caplog):
    client, _, _, _ = build_client()

    with caplog.at_level(logging.INFO, logger="server.usage"):
        call_tool(client)

    records = [r for r in caplog.records if r.name == "server.usage"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert USER_EMAIL in message
    assert "GC-100" in message
    assert "success" in message


@pytest.mark.deterministic
def test_failed_calls_are_logged_too(caplog):
    analyzer = FakeAnalyzer(error=RuntimeError("weaviate unreachable"))
    client, _, _, _ = build_client(analyzer=analyzer)

    with caplog.at_level(logging.INFO, logger="server.usage"):
        call_tool(client)

    records = [r for r in caplog.records if r.name == "server.usage"]
    assert len(records) == 1
    assert "error" in records[0].getMessage()


@pytest.mark.deterministic
def test_a_refused_call_is_logged_with_the_reason(caplog):
    checker = FakeAccessChecker(
        error=AuthError(AuthErrorCode.ACCESS_DENIED, "No access to GC-100")
    )
    client, _, _, _ = build_client(access_checker=checker)

    with caplog.at_level(logging.INFO, logger="server.usage"):
        call_tool(client)

    records = [r for r in caplog.records if r.name == "server.usage"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "access_denied" in message
    assert USER_EMAIL in message


@pytest.mark.deterministic
def test_the_usage_logger_has_somewhere_to_write():
    """uvicorn configures only its own loggers, so the app must attach a handler
    or every usage record is silently dropped in production."""
    build_client()

    usage_logger = logging.getLogger("server.usage")
    assert usage_logger.handlers
    assert usage_logger.isEnabledFor(logging.INFO)


# ---------------------------------------------------------------------------
# Staying answerable during a long analysis
# ---------------------------------------------------------------------------
def ran_off_the_event_loop() -> bool:
    """True when called from a thread that is not running an asyncio loop.

    `get_running_loop` only succeeds in the loop's own thread, so this is an
    exact answer to "was the event loop blocked by this call?".
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return True
    return False


@pytest.mark.deterministic
def test_the_analysis_does_not_run_on_the_event_loop():
    """The analysis is synchronous and can take minutes; on the event loop it
    would stall every other request, `GET /health` included."""
    observed = {}

    class LoopAwareAnalyzer:
        def analyze(self, ticket_key, on_progress=None):
            observed["off_loop"] = ran_off_the_event_loop()
            return make_analysis()

    client, _, _, _ = build_client(analyzer=LoopAwareAnalyzer())

    assert call_tool(client).status_code == 200
    assert observed["off_loop"] is True


@pytest.mark.deterministic
def test_the_redaction_pass_does_not_run_on_the_event_loop():
    observed = {}

    class LoopAwareSanitizer:
        def sanitize(self, analysis):
            observed["off_loop"] = ran_off_the_event_loop()
            return analysis

    client, _, _, _ = build_client(sanitizer=LoopAwareSanitizer())

    assert call_tool(client).status_code == 200
    assert observed["off_loop"] is True


@pytest.mark.deterministic
def test_the_access_check_does_not_run_on_the_event_loop():
    observed = {}

    class LoopAwareChecker:
        def verify_access(self, email, token, ticket_key):
            observed["off_loop"] = ran_off_the_event_loop()

    client, _, _, _ = build_client(access_checker=LoopAwareChecker())

    assert call_tool(client).status_code == 200
    assert observed["off_loop"] is True


# ---------------------------------------------------------------------------
# The access check itself, against a mocked Jira
# ---------------------------------------------------------------------------
def make_access_checker(monkeypatch, can_access=True, raises=None):
    from unittest.mock import MagicMock

    import server.auth as auth_module

    created = MagicMock()
    created.can_access_issue.return_value = can_access
    if raises is not None:
        created.can_access_issue.side_effect = raises
    factory = MagicMock(return_value=created)
    monkeypatch.setattr(auth_module, "JiraClient", factory)
    return auth_module.JiraAccessChecker(), factory, created


def http_error(status_code):
    import requests

    response = requests.Response()
    response.status_code = status_code
    return requests.exceptions.HTTPError(response=response)


@pytest.mark.deterministic
def test_the_access_check_probes_jira_with_the_user_credentials(monkeypatch):
    checker, factory, client = make_access_checker(monkeypatch)

    checker.verify_access(USER_EMAIL, USER_TOKEN, "GC-100")

    assert factory.call_args.kwargs["email"] == USER_EMAIL
    assert factory.call_args.kwargs["api_token"] == USER_TOKEN
    client.can_access_issue.assert_called_once_with("GC-100")


@pytest.mark.deterministic
def test_the_user_token_is_discarded_after_the_access_check(monkeypatch):
    """The per-request client is closed so the token is not held anywhere."""
    checker, _, client = make_access_checker(monkeypatch)

    checker.verify_access(USER_EMAIL, USER_TOKEN, "GC-100")

    client.close.assert_called_once()


@pytest.mark.deterministic
def test_the_user_token_is_discarded_even_when_the_check_fails(monkeypatch):
    checker, _, client = make_access_checker(monkeypatch, raises=http_error(500))

    with pytest.raises(AuthError):
        checker.verify_access(USER_EMAIL, USER_TOKEN, "GC-100")

    client.close.assert_called_once()


@pytest.mark.deterministic
def test_a_ticket_the_user_cannot_read_is_access_denied(monkeypatch):
    checker, _, _ = make_access_checker(monkeypatch, can_access=False)

    with pytest.raises(AuthError) as exc_info:
        checker.verify_access(USER_EMAIL, USER_TOKEN, "GC-100")

    assert exc_info.value.code is AuthErrorCode.ACCESS_DENIED


@pytest.mark.deterministic
def test_a_rejected_token_is_an_invalid_credentials_error(monkeypatch):
    checker, _, _ = make_access_checker(monkeypatch, raises=http_error(401))

    with pytest.raises(AuthError) as exc_info:
        checker.verify_access(USER_EMAIL, USER_TOKEN, "GC-100")

    assert exc_info.value.code is AuthErrorCode.INVALID_CREDENTIALS


@pytest.mark.deterministic
def test_an_unreachable_jira_is_an_upstream_error(monkeypatch):
    import requests

    checker, _, _ = make_access_checker(
        monkeypatch, raises=requests.exceptions.ConnectionError("no route to host")
    )

    with pytest.raises(AuthError) as exc_info:
        checker.verify_access(USER_EMAIL, USER_TOKEN, "GC-100")

    assert exc_info.value.code is AuthErrorCode.UPSTREAM_ERROR


@pytest.mark.deterministic
def test_an_upstream_error_is_surfaced_as_a_bad_gateway():
    checker = FakeAccessChecker(
        error=AuthError(AuthErrorCode.UPSTREAM_ERROR, "Jira is unreachable")
    )
    client, analyzer, _, _ = build_client(access_checker=checker)

    response = call_tool(client)

    assert_tool_error(response)
    assert analyzer.calls == []


# ---------------------------------------------------------------------------
# Progress notifications
#
# An analysis runs for minutes. When the client supplies a progress token the
# tool call is answered with an SSE stream carrying `notifications/progress`
# followed by the result, so the user sees what is happening instead of a
# silent wait. Without a token the response stays a single JSON body.
# ---------------------------------------------------------------------------
class ProgressAnalyzer:
    """An analyzer that reports a few stages, the way the real one does."""

    STAGES = ("Analyzing ticket...", "Finding similar tickets...", "Finalizing...")

    def __init__(self, analysis=None, error=None):
        self._analysis = analysis if analysis is not None else make_analysis()
        self._error = error
        self.progress_callbacks = []

    def analyze(self, ticket_key, on_progress=None):
        self.progress_callbacks.append(on_progress)
        for stage in self.STAGES:
            if on_progress:
                on_progress(stage)
        if self._error:
            raise self._error
        return self._analysis


def call_tool_streaming(client, ticket_key="GC-100", progress_token="tok-1",
                        headers=AUTH_HEADERS, analyzer_name="analyze_ticket"):
    """Drive a tool call that asks for progress, returning the parsed SSE events."""
    params = {"name": analyzer_name, "arguments": {"ticket_key": ticket_key}}
    if progress_token is not None:
        params["_meta"] = {"progressToken": progress_token}
    with client.stream("POST", "/mcp", json=rpc("tools/call", params), headers=headers) as response:
        assert response.status_code == 200
        content_type = response.headers["content-type"]
        events = [
            json.loads(line[len("data:"):].strip())
            for line in response.iter_lines()
            if line.startswith("data:")
        ]
    return content_type, events


def progress_events(events):
    return [e for e in events if e.get("method") == "notifications/progress"]


@pytest.mark.deterministic
def test_a_progress_token_switches_the_response_to_an_event_stream():
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    content_type, _ = call_tool_streaming(client)

    assert content_type.startswith("text/event-stream")


@pytest.mark.deterministic
def test_progress_notifications_echo_the_clients_token():
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    _, events = call_tool_streaming(client, progress_token="tok-abc")
    notifications = progress_events(events)

    assert notifications, "expected at least one progress notification"
    assert all(n["params"]["progressToken"] == "tok-abc" for n in notifications)


@pytest.mark.deterministic
def test_a_numeric_progress_token_stays_numeric():
    """Claude Code sends an integer token; echoing it as a string breaks correlation."""
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    _, events = call_tool_streaming(client, progress_token=2)

    tokens = [n["params"]["progressToken"] for n in progress_events(events)]
    assert tokens and all(t == 2 for t in tokens)


@pytest.mark.deterministic
def test_progress_values_strictly_increase():
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    _, events = call_tool_streaming(client)
    values = [n["params"]["progress"] for n in progress_events(events)]

    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.deterministic
def test_the_analysis_stages_reach_the_client_as_messages():
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    _, events = call_tool_streaming(client)
    messages = [n["params"]["message"] for n in progress_events(events)]

    for stage in ProgressAnalyzer.STAGES:
        assert stage in messages


@pytest.mark.deterministic
def test_the_redaction_pass_reports_progress_too():
    """The sanitizer is another multi-second LLM call; it should not be silent."""
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    _, events = call_tool_streaming(client)
    messages = [n["params"]["message"].lower() for n in progress_events(events)]

    assert any("redact" in m for m in messages)


@pytest.mark.deterministic
def test_the_stream_ends_with_the_analysis_result():
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    _, events = call_tool_streaming(client)
    final = events[-1]

    assert final["id"] == 1
    assert final["jsonrpc"] == "2.0"
    assert "GC-100" in "\n".join(b["text"] for b in final["result"]["content"])


@pytest.mark.deterministic
def test_progress_arrives_before_the_result():
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    _, events = call_tool_streaming(client)
    kinds = ["progress" if e.get("method") else "result" for e in events]

    assert kinds.count("result") == 1
    assert kinds.index("result") == len(kinds) - 1


@pytest.mark.deterministic
def test_a_tool_call_without_a_progress_token_still_returns_plain_json():
    """Back-compat: plain curl and every existing client keep the single JSON body."""
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    response = call_tool(client)

    assert response.headers["content-type"].startswith("application/json")
    assert "GC-100" in tool_text(response)


@pytest.mark.deterministic
def test_the_analyzer_gets_no_callback_when_no_progress_was_asked_for():
    analyzer = ProgressAnalyzer()
    client, _, _, _ = build_client(analyzer=analyzer)

    call_tool(client)

    assert analyzer.progress_callbacks == [None]


@pytest.mark.deterministic
def test_an_access_failure_ends_the_stream_with_an_error_result():
    checker = FakeAccessChecker(
        error=AuthError(AuthErrorCode.ACCESS_DENIED, "You do not have access to GC-100")
    )
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer(), access_checker=checker)

    _, events = call_tool_streaming(client)
    final = events[-1]

    assert final["result"]["isError"] is True
    assert "access" in final["result"]["content"][0]["text"].lower()


@pytest.mark.deterministic
def test_a_failing_analysis_ends_the_stream_with_an_error_result():
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer(error=RuntimeError("boom")))

    _, events = call_tool_streaming(client)
    final = events[-1]

    assert final["result"]["isError"] is True


@pytest.mark.deterministic
def test_a_sanitization_failure_ends_the_stream_without_leaking_the_analysis():
    sanitizer = FakeSanitizer(error=SanitizationError("redaction pass failed"))
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer(), sanitizer=sanitizer)

    _, events = call_tool_streaming(client)
    final = events[-1]
    body = json.dumps(events)

    assert final["result"]["isError"] is True
    assert "Collector crashes on startup" not in body


@pytest.mark.deterministic
def test_the_users_token_never_appears_in_the_stream(caplog):
    client, _, _, _ = build_client(analyzer=ProgressAnalyzer())

    with caplog.at_level(logging.DEBUG):
        _, events = call_tool_streaming(client)

    assert USER_TOKEN not in json.dumps(events)
    assert USER_TOKEN not in caplog.text


@pytest.mark.deterministic
def test_a_refused_redaction_logs_why_without_quoting_the_ticket(caplog):
    """Fail-closed is right, but silent fail-closed cannot be operated: the
    operator needs the reason, and must not get the customer data with it."""
    sanitizer = FakeSanitizer(error=SanitizationError(
        "PII sanitization returned 38 of 41 segments; 3 would have been returned unredacted"
    ))
    client, _, _, _ = build_client(sanitizer=sanitizer)

    with caplog.at_level(logging.ERROR):
        response = call_tool(client)

    assert_tool_error(response)
    assert "38 of 41 segments" in caplog.text
    assert "Collector crashes on startup" not in caplog.text
    assert USER_TOKEN not in caplog.text
