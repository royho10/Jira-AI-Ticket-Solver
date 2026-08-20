# config/settings.py
"""
Centralized configuration for the Jira AI Ticket Solver.
Only shared/generic constants should be defined here.
File-specific constants should remain in their respective files.
"""

# =============================================================================
# Azure OpenAI Configuration (loaded from environment)
# =============================================================================
import os

from dotenv import load_dotenv

# Loaded here so that importing settings is enough to see `.env`, whichever entry
# point (Streamlit, MCP server, indexer, pytest) imported it first.
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Azure OpenAI Deployment Names
AZURE_OPENAI_LLM_DEPLOYMENT = os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-5-nano")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# LLM Temperature Configuration
# Note: gpt-5-nano only supports temperature=1.0
AZURE_OPENAI_TEMPERATURE = 1.0

# =============================================================================
# Weaviate Configuration
# =============================================================================
JIRA_COLLECTION_NAME = "JiraCollection"

# Shared remote Weaviate. When WEAVIATE_URL is unset the client falls back to a
# local Docker instance, so local development needs no extra configuration.
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

# =============================================================================
# Remote MCP Server
# =============================================================================
# Browser-style clients must state an Origin on the allowlist; requests without
# an Origin (Claude Code, curl) are unaffected. Empty means no Origin is allowed,
# which is the right default for a non-browser transport.
MCP_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("MCP_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
MCP_SSE_KEEPALIVE_SECONDS = float(os.environ.get("MCP_SSE_KEEPALIVE_SECONDS", "15"))
# The event stream is closed after this long so a dropped client cannot pin a
# worker forever; MCP clients are expected to reconnect.
MCP_SSE_STREAM_SECONDS = float(os.environ.get("MCP_SSE_STREAM_SECONDS", "300"))

# =============================================================================
# Generic Limits
# =============================================================================
MAX_EMBEDDINGS_INPUT_CHARS = 4000
LLM_CALL_TIMEOUT_SECONDS = 60

# =============================================================================
# Reranking Configuration
# =============================================================================
RERANK_SCORE_THRESHOLD = 5  # 0-10 scale; tickets below this are filtered out
MAX_SIMILAR_TICKETS_AFTER_RERANK = 5
