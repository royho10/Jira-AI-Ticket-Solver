"""The Remote MCP Server: MCP Streamable HTTP in front of the Ticket Analyzer.

Claude Code talks JSON-RPC to `POST /mcp`, opens `GET /mcp` for server-initiated
events, and monitoring polls `GET /health`. A tool call runs the full pipeline:

    access check (user token) -> Analysis (service account) -> Sanitization Pass -> formatting

Auth notes. Origin validation happens for every request; credential extraction
and the access check happen inside the tool call, because the ticket key being
authorized lives in the JSON-RPC body rather than the URL. Neither the token nor
any other auth header is ever written to a log.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from config.settings import (
    MCP_ALLOWED_ORIGINS,
    MCP_SSE_KEEPALIVE_SECONDS,
    MCP_SSE_STREAM_SECONDS,
)
from core.pii_sanitizer import SanitizationError
from server.auth import (
    AuthError,
    AuthErrorCode,
    JiraAccessChecker,
    extract_credentials,
    validate_origin,
)
from server.response_formatter import format_analysis

logger = logging.getLogger(__name__)
usage_logger = logging.getLogger("server.usage")

SERVER_NAME = "jira-ticket-solver"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-06-18"

TOOL_NAME = "analyze_ticket"
TOOL_DEFINITION = {
    "name": TOOL_NAME,
    "description": (
        "Analyze a Jira ticket: summarize it, extract errors from attached logs and "
        "images, hypothesize root causes, suggest solutions, and surface similar "
        "historical tickets. Customer-identifying information is redacted before the "
        "result is returned."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "ticket_key": {
                "type": "string",
                "description": "The Jira ticket key, e.g. GC-1234.",
            },
        },
        "required": ["ticket_key"],
    },
}

# JSON-RPC 2.0 error codes, plus the application codes we add for auth.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
AUTH_ERROR = -32001

# Origin rejection is a transport-level refusal, so it answers with a status
# code. Credential and access failures deliberately do NOT: an MCP client reads
# 401 on the MCP endpoint as an OAuth challenge and starts authorization
# discovery, which this token-in-header MVP does not implement. Those failures
# come back as an `isError` tool result, which Claude relays to the user.
ORIGIN_REJECTED_STATUS = 403


def _error(request_id: Any, code: int, message: str, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
    )


def _result(request_id: Any, result: Dict[str, Any]) -> JSONResponse:
    return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "result": result})


def _tool_text(text: str, is_error: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return result


def _elapsed_ms(started: float) -> int:
    return int((time.time() - started) * 1000)


def _sse(payload: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _progress_notification(token: Any, step: int, message: str) -> Dict[str, Any]:
    """A `notifications/progress` for a long-running tool call.

    `total` is deliberately omitted: the number of stages depends on how many
    attachments a ticket carries, so any total we announced would be a guess.
    The token is echoed exactly as the client sent it -- Claude Code sends an
    integer, and stringifying it would break correlation.
    """
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progressToken": token, "progress": step, "message": message},
    }


def _ensure_usage_log_handler() -> None:
    """Give the usage logger somewhere to write.

    uvicorn configures only its own loggers, so without a handler here every
    per-user usage record would be dropped by the last-resort WARNING filter.
    """
    if usage_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    usage_logger.addHandler(handler)
    usage_logger.setLevel(logging.INFO)


def _log_usage(email: str, ticket_key: str, outcome: str, duration_ms: int) -> None:
    """Per-user usage record. Receives the identity and the ticket, never the token."""
    usage_logger.info(
        "mcp_tool_call at=%s user=%s ticket=%s outcome=%s duration_ms=%d",
        datetime.now(timezone.utc).isoformat(),
        email,
        ticket_key,
        outcome,
        duration_ms,
    )


def create_app(
    analyzer: Any = None,
    sanitizer: Any = None,
    access_checker: Any = None,
    allowed_origins: Optional[Sequence[str]] = None,
    sse_keepalive_seconds: Optional[float] = None,
    sse_stream_seconds: Optional[float] = None,
) -> FastAPI:
    """Build the ASGI app. Collaborators are injected by tests and built here in
    production, where the environment is configured."""
    if analyzer is None:
        from core.ticket_analyzer import TicketAnalyzer
        analyzer = TicketAnalyzer()
    if sanitizer is None:
        from core.pii_sanitizer import PiiSanitizer
        sanitizer = PiiSanitizer()
    if access_checker is None:
        access_checker = JiraAccessChecker()

    _ensure_usage_log_handler()

    origins = list(MCP_ALLOWED_ORIGINS if allowed_origins is None else allowed_origins)
    keepalive = MCP_SSE_KEEPALIVE_SECONDS if sse_keepalive_seconds is None else sse_keepalive_seconds
    stream_lifetime = MCP_SSE_STREAM_SECONDS if sse_stream_seconds is None else sse_stream_seconds

    app = FastAPI(title="Jira AI Ticket Solver MCP Server", version=SERVER_VERSION)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "server": SERVER_NAME, "version": SERVER_VERSION}

    @app.get("/mcp")
    async def mcp_stream(request: Request) -> Response:
        """The server-initiated event stream.

        This server answers every request inline on the POST, so the stream
        carries only keep-alive comments; it exists so clients that open it get a
        well-formed response instead of an error. It closes after
        `MCP_SSE_STREAM_SECONDS` so a client that vanished cannot pin a worker
        indefinitely -- MCP clients reconnect.
        """
        try:
            validate_origin(request.headers.get("origin"), origins)
        except AuthError as e:
            return _origin_rejected(e)

        async def keep_alive():
            deadline = time.monotonic() + stream_lifetime
            while time.monotonic() < deadline and not await request.is_disconnected():
                yield b": keep-alive\n\n"
                await asyncio.sleep(keepalive)

        return StreamingResponse(
            keep_alive(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        try:
            validate_origin(request.headers.get("origin"), origins)
        except AuthError as e:
            return _origin_rejected(e)

        raw = await request.body()
        try:
            message = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(None, PARSE_ERROR, "Request body is not valid JSON", status_code=400)

        if not isinstance(message, dict) or not isinstance(message.get("method"), str):
            return _error(
                message.get("id") if isinstance(message, dict) else None,
                INVALID_REQUEST,
                "Request must be a JSON-RPC 2.0 object with a 'method'",
                status_code=400,
            )

        method = message["method"]
        request_id = message.get("id")
        params = message.get("params") or {}

        # Notifications carry no id and get no response body.
        if request_id is None and method.startswith("notifications/"):
            return Response(status_code=202)

        if method == "initialize":
            return _result(request_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })

        if method == "ping":
            return _result(request_id, {})

        if method == "tools/list":
            return _result(request_id, {"tools": [TOOL_DEFINITION]})

        if method == "tools/call":
            return await _handle_tool_call(request, request_id, params)

        return _error(request_id, METHOD_NOT_FOUND, f"Unknown method '{method}'")

    def _origin_rejected(error: AuthError) -> JSONResponse:
        return _error(None, AUTH_ERROR, error.message, status_code=ORIGIN_REJECTED_STATUS)

    async def _handle_tool_call(request: Request, request_id: Any, params: Dict[str, Any]) -> Response:
        if params.get("name") != TOOL_NAME:
            return _error(request_id, INVALID_PARAMS, f"Unknown tool '{params.get('name')}'")

        arguments = params.get("arguments") or {}
        ticket_key = arguments.get("ticket_key")
        if not isinstance(ticket_key, str) or not ticket_key.strip():
            return _error(request_id, INVALID_PARAMS, "'ticket_key' is required")
        ticket_key = ticket_key.strip().upper()

        # A client that wants to watch the analysis sends a progress token. That
        # turns the answer into an event stream; without one it stays a single
        # JSON body, which is what plain curl and any non-progress client expect.
        progress_token = (params.get("_meta") or {}).get("progressToken")
        if progress_token is None:
            return _result(request_id, await _run_analysis(request, ticket_key, None))
        return _streaming_tool_call(request, request_id, ticket_key, progress_token)

    async def _run_analysis(
        request: Request,
        ticket_key: str,
        on_progress: Optional[Callable[[str], None]],
    ) -> Dict[str, Any]:
        """Run the pipeline and return the tool result payload.

        Every failure becomes an `isError` payload rather than an exception, so
        both delivery paths have exactly one thing to send back.
        """
        started = time.time()
        email = None
        try:
            email, token = extract_credentials(request.headers)
            # Blocking Jira call: off the event loop, like the analysis below.
            await run_in_threadpool(access_checker.verify_access, email, token, ticket_key)
        except AuthError as e:
            _log_usage(email or "unknown", ticket_key, e.code.value, _elapsed_ms(started))
            return _tool_text(e.message, is_error=True)

        # The token has done its job; from here on only the identity travels on.
        del token

        try:
            # The analysis and the redaction pass are synchronous and can run for
            # minutes; run them in a worker so /health and other requests stay
            # answerable.
            analysis = await run_in_threadpool(analyzer.analyze, ticket_key, on_progress)
            if on_progress:
                on_progress("🛡️ **Redacting customer data...**")
            sanitized = await run_in_threadpool(sanitizer.sanitize, analysis)
        except SanitizationError as e:
            # Safe to log: `SanitizationError` messages never quote the text that
            # failed redaction. Without this the operator sees only the outcome and
            # cannot tell a model completeness failure from an outage.
            logger.error(f"Redaction refused the analysis of {ticket_key}: {e}")
            _log_usage(email, ticket_key, "sanitization_error", _elapsed_ms(started))
            return _tool_text(
                f"Could not return the analysis of {ticket_key}: the required redaction "
                "pass failed, so nothing from the ticket can be shown. Please retry.",
                is_error=True,
            )
        except Exception as e:
            logger.exception(f"Analysis of {ticket_key} failed")
            _log_usage(email, ticket_key, "error", _elapsed_ms(started))
            return _tool_text(f"Failed to analyze {ticket_key}: {e}", is_error=True)

        _log_usage(email, ticket_key, "success", _elapsed_ms(started))
        return _tool_text(format_analysis(sanitized))

    def _streaming_tool_call(
        request: Request,
        request_id: Any,
        ticket_key: str,
        progress_token: Any,
    ) -> StreamingResponse:
        """Answer the tool call with progress events followed by the result."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(message: str) -> None:
            # Called from the analysis worker thread, so the hand-off back to the
            # event loop has to be thread-safe.
            loop.call_soon_threadsafe(queue.put_nowait, message)

        async def event_stream():
            analysis = asyncio.ensure_future(_run_analysis(request, ticket_key, on_progress))
            step = 0
            while True:
                pending_message = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {pending_message, analysis}, return_when=asyncio.FIRST_COMPLETED
                )
                if pending_message in done:
                    step += 1
                    yield _sse(_progress_notification(
                        progress_token, step, pending_message.result()
                    ))
                    continue
                # The analysis finished. Cancelling the waiter leaves any message
                # already queued in place, so drain before closing the stream.
                pending_message.cancel()
                while not queue.empty():
                    step += 1
                    yield _sse(_progress_notification(
                        progress_token, step, queue.get_nowait()
                    ))
                break

            error = analysis.exception()
            if error is not None:
                logger.exception(f"Streaming analysis of {ticket_key} failed", exc_info=error)
                payload = _tool_text(f"Failed to analyze {ticket_key}: {error}", is_error=True)
            else:
                payload = analysis.result()
            yield _sse({"jsonrpc": "2.0", "id": request_id, "result": payload})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app
