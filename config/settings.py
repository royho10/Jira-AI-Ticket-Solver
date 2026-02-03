# config/settings.py
"""
Centralized configuration for the Jira AI Ticket Solver.
Only shared/generic constants should be defined here.
File-specific constants should remain in their respective files.
"""

# =============================================================================
# Model Configuration
# =============================================================================
EMBEDDING_MODEL_NAME = "nomic-embed-text-v2-moe"
VLM_MODEL_NAME = "qwen2.5vl:7b"
LLM_MODEL_NAME = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# =============================================================================
# Weaviate Configuration
# =============================================================================
JIRA_COLLECTION_NAME = "JiraCollection"

# =============================================================================
# Generic Limits
# =============================================================================
MAX_EMBEDDINGS_INPUT_CHARS = 4000
LLM_CALL_TIMEOUT_SECONDS = 60
