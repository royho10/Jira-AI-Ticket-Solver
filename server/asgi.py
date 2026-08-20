"""ASGI entry point for the Remote MCP Server.

    uvicorn server.asgi:app --host 0.0.0.0 --port 8000

Building the app here (rather than in `mcp_server`) keeps the module importable
without a configured environment: the real Ticket Analyzer, PII Sanitizer and
Jira access checker are constructed only when this module is loaded.
"""
from dotenv import load_dotenv

load_dotenv()

from server.mcp_server import create_app  # noqa: E402  (after load_dotenv by design)
from utils import llm_logger  # noqa: E402

# Prompts contain raw, unsanitized ticket text belonging to whichever user made
# the request. On a shared server that must not be written to disk.
llm_logger.disable()

app = create_app()
