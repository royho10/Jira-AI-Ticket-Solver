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
