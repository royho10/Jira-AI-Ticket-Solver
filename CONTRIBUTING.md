# Contributing Guide

This guide covers development setup, code patterns, and contribution guidelines for the Jira AI Ticket Solver project.

## Development Setup

### 1. Clone and Create Virtual Environment

```bash
git clone https://github.com/royho10/Jira-AI-Ticket-Solver.git
cd Jira-AI-Ticket-Solver
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Jira credentials
```

### 4. Start Required Services

**Weaviate (Vector Database):**
```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

**Ollama (LLM Server):**
```bash
ollama serve

# In another terminal, pull required models
ollama pull llama3
ollama pull qwen2.5vl:7b
ollama pull nomic-embed-text
```

## Code Structure

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config/settings.py` | Centralized configuration constants |
| `utils/jira_client.py` | Jira API client and Pydantic data models |
| `utils/jira_ticket_processing.py` | LLM/VLM processing logic (Ollama) |
| `utils/openai_jira_ticket_processing.py` | LLM/VLM processing logic (Azure OpenAI) |
| `utils/file_utils.py` | Archive extraction utilities |
| `indexer/index_jira_tickets.py` | Batch indexing pipeline (Ollama) |
| `indexer/openai_index_jira_tickets.py` | Batch indexing pipeline (Azure OpenAI) |
| `app/chatbot.py` | Streamlit UI (Ollama) |
| `app/openai_chatbot.py` | Streamlit UI (Azure OpenAI) |

### Directory Structure

```
Jira-AI-Ticket-Solver/
├── app/                    # Web interfaces (chatbot.py, openai_chatbot.py)
├── config/                 # Configuration (Ollama + Azure OpenAI settings)
├── indexer/                # Batch processing (Ollama + Azure OpenAI indexers)
├── utils/                  # Shared utilities (Ollama + Azure OpenAI processors)
├── requirements.txt        # Dependencies
└── .env.example           # Environment template
```

## Configuration

### Settings File (`config/settings.py`)

All shared constants are centralized here:

```python
# Ollama Model Configuration
EMBEDDING_MODEL_NAME = "nomic-embed-text-v2-moe"
VLM_MODEL_NAME = "qwen2.5vl:7b"
LLM_MODEL_NAME = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# Azure OpenAI Configuration (loaded from environment)
import os
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_LLM_DEPLOYMENT = os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Weaviate Configuration (separate collections due to different embedding dimensions)
JIRA_COLLECTION_NAME = "JiraCollection"           # Ollama: 768 dims
OPENAI_JIRA_COLLECTION_NAME = "JiraCollectionOpenAI"  # Azure OpenAI: 1536 dims

# Generic Limits
MAX_EMBEDDINGS_INPUT_CHARS = 4000
LLM_CALL_TIMEOUT_SECONDS = 60
```

**Guidelines:**
- Only shared/generic constants go in `settings.py`
- Module-specific constants stay in their respective files
- Environment variables are loaded via `python-dotenv`

## Running Locally

### Run the Indexer

```bash
# Ollama indexer (fully local)
python -m indexer.index_jira_tickets

# Azure OpenAI indexer (requires Azure OpenAI credentials)
python -m indexer.openai_index_jira_tickets

# To customize, modify the JQL in the respective indexer's main()
```

### Run the Chatbot

```bash
# Ollama chatbot (fully local)
streamlit run app/chatbot.py

# Azure OpenAI chatbot (requires Azure OpenAI credentials)
streamlit run app/openai_chatbot.py
```

**Note:** Each chatbot queries its corresponding Weaviate collection. Make sure to index tickets with the matching indexer first.

### Verify Services

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check Weaviate
curl http://localhost:8080/v1/.well-known/ready
```

## Key Patterns

### 1. Pydantic for Data Models

All LLM outputs use Pydantic models for structured output:

```python
from pydantic import BaseModel, Field

class LogAnalysisOutput(BaseModel):
    log_filename: str = Field(description="Name of the log file")
    errors: List[ErrorInLog] = Field(min_length=1, description="List of errors")
```

Usage with LangChain:
```python
llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)
structured_llm = llm.with_structured_output(LogAnalysisOutput)
result = structured_llm.invoke(messages)
```

### 2. Thread-Local LLM Instances

To avoid socket churn in multithreaded contexts:

```python
from threading import local

_thread_local = local()

def _get_llm(self) -> ChatOllama:
    if not hasattr(_thread_local, "llm") or _thread_local.llm is None:
        _thread_local.llm = ChatOllama(...)
    return _thread_local.llm
```

### 3. Batch Weaviate Operations

Insert data in batches for efficiency:

```python
WEAVIATE_BATCH_SIZE = 100

for i in range(0, len(data_objects), WEAVIATE_BATCH_SIZE):
    batch = data_objects[i:i + WEAVIATE_BATCH_SIZE]
    collection.data.insert_many(batch)
```

### 4. Jira API Client Session Reuse

The `JiraClient` reuses HTTP sessions:

```python
class JiraClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.auth = HTTPBasicAuth(email, token)
        self._session.headers.update({"Accept": "application/json"})

    def close(self):
        self._session.close()
```

### 5. Graceful Degradation

Skip failed items without stopping the entire process:

```python
for attachment in attachments:
    try:
        result = process_attachment(attachment)
        results.append(result)
    except Exception as e:
        print(f"Error processing {attachment.filename}: {e}")
        continue  # Skip and continue
```

## Testing Changes

### Manual Testing Workflow

1. **Test Indexer:**
   ```bash
   # Modify JQL to test with small dataset
   python -m indexer.index_jira_tickets
   ```

2. **Test Chatbot:**
   ```bash
   streamlit run app/chatbot.py
   # Enter a known ticket key
   # Verify analysis output
   ```

3. **Test Specific Components:**
   ```python
   # In Python REPL
   from utils.jira_client import JiraClient
   client = JiraClient()
   issue = client.fetch_issue_by_key("PROJ-123")
   print(issue)
   ```

### Checking Weaviate Data

```python
import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("JiraCollection")

# Count objects
print(f"Total objects: {len(collection)}")

# Query sample
for item in collection.iterator():
    print(item.properties)
    break

client.close()
```

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Use dataclasses or Pydantic for data structures
- Prefer explicit imports over `from module import *`

### Naming Conventions

- Classes: `PascalCase` (e.g., `JiraClient`, `LogAnalysisOutput`)
- Functions/Methods: `snake_case` (e.g., `fetch_issues`, `_process_attachments`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_LINES_PER_LOG`)
- Private methods: prefix with `_` (e.g., `_get_llm`)

### Documentation

- Docstrings for public functions and classes
- Inline comments for complex logic
- Type hints for all function parameters and returns

### Example Function Style

```python
def extract_content_from_zip(
    file: bytes,
    max_files_to_extract: int = None
) -> List[Tuple[str, str]]:
    """Extract log contents from zip file attachment.

    Args:
        file: Raw bytes of the ZIP file
        max_files_to_extract: Maximum number of files to process

    Returns:
        List of (text_content, filename) tuples
    """
    log_contents = []
    with zipfile.ZipFile(io.BytesIO(file), "r") as z:
        # ... implementation
    return log_contents
```

## PR Guidelines

### Before Submitting

1. **Test your changes** locally with both chatbot interfaces
2. **Verify no regressions** in existing functionality
3. **Update documentation** if adding new features or changing behavior
4. **Check for hardcoded values** - use `config/settings.py` for shared constants

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Bullet point list of modifications

## Testing
How the changes were tested

## Notes
Any additional context or considerations
```

### Commit Messages

Use clear, descriptive commit messages:

```
Add image analysis timeout handling

- Add LLM_CALL_TIMEOUT_SECONDS configuration
- Wrap VLM calls in timeout context
- Return empty result on timeout instead of crashing
```

## Common Development Tasks

### Adding a New Pydantic Model

1. Define in appropriate file (usually `jira_ticket_processing.py`)
2. Use `Field()` with descriptions for LLM structured output
3. Add type hints

```python
class NewOutputModel(BaseModel):
    field_name: str = Field(description="What this field contains")
    optional_field: Optional[int] = Field(default=None, description="Optional info")
```

### Adding a New Processing Step

1. Create processing method in `JiraIssueLLMProcessor`
2. Add system/user prompt methods
3. Call from `process_issue()` pipeline
4. Update `_parse_final_issue_summary_output_to_text()` if output changes

### Modifying Weaviate Schema

1. Update `_setup_collection()` in the respective indexer file
2. Delete existing collection or use new name for testing
3. Re-index tickets

```python
# To delete existing collections (in Python REPL)
import weaviate
client = weaviate.connect_to_local()
client.collections.delete("JiraCollection")       # Ollama collection
client.collections.delete("JiraCollectionOpenAI") # OpenAI collection
client.close()
```

### Updating Environment Variables

1. Add to `.env.example` with placeholder value
2. Load in relevant module with `os.environ.get()` or `os.environ[]`
3. Document in README.md

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

### Weaviate Connection Issues

```bash
# Check if Weaviate is running
curl http://localhost:8080/v1/.well-known/ready

# Restart Weaviate container
docker restart <container_id>
```

### Memory Issues During Indexing

Reduce concurrent workers in `index_jira_tickets.py`:

```python
MAX_PROCESS_TICKET_WORKERS = 2  # Reduce from 5
```

### LLM Timeout Issues

Increase timeout in `config/settings.py`:

```python
LLM_CALL_TIMEOUT_SECONDS = 120  # Increase from 60
```
