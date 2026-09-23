"""ASGI entry point for the Remote MCP Server.

    uvicorn server.asgi:app --host 0.0.0.0 --port 8000

Building the app here (rather than in `mcp_server`) keeps the module importable
without a configured environment: the real Ticket Analyzer, PII Sanitizer and
Jira access checker are constructed only when this module is loaded.
"""
from dotenv import load_dotenv

load_dotenv()

import logging  # noqa: E402

from config.settings import MCP_LOG_LLM_CALLS  # noqa: E402
from server.mcp_server import create_app  # noqa: E402  (after load_dotenv by design)
from utils import llm_logger  # noqa: E402

# Prompts contain raw, unsanitized ticket text belonging to whichever user made
# the request. On a shared server that must not be written to disk, so logging is
# off unless an operator debugging a local run asks for it explicitly.
if MCP_LOG_LLM_CALLS:
    logging.getLogger(__name__).warning(
        "MCP_LOG_LLM_CALLS is set: LLM prompts and responses, including raw ticket "
        "text, are being written to logs/llm_calls/. Do not use this on a shared server."
    )
else:
    llm_logger.disable()

app = create_app()
