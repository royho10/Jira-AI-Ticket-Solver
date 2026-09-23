# Contributing Guide

This guide covers development setup, code patterns, and contribution guidelines for the Jira AI Ticket Solver project.

## Development Setup

### 1. Clone and Create Virtual Environment

Python 3.10+ is required (`langchain` 1.x and `streamlit` 1.51 both floor there).

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
# Edit .env with your Jira and Azure OpenAI credentials
```

### 4. Start Required Services

**Weaviate (Vector Database):**
```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

## Code Structure

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config/settings.py` | Centralized configuration constants |
| `utils/jira_client.py` | Jira API client and Pydantic data models |
| `utils/weaviate_client.py` | Local vs remote Weaviate connection |
| `utils/openai_jira_ticket_processing.py` | LLM/VLM processing logic (Azure OpenAI) |
| `utils/file_utils.py` | Archive extraction utilities |
| `utils/llm_logger.py` | LLM call logging (disabled on the MCP server) |
| `core/ticket_analyzer.py` | The Analysis pipeline, UI-agnostic and stateless |
| `core/pii_sanitizer.py` | Mandatory LLM redaction pass |
| `server/mcp_server.py` | MCP Streamable HTTP app (FastAPI) |
| `server/auth.py` | Jira access check and Origin validation |
| `server/response_formatter.py` | Markdown + `<ticket_analysis>` rendering |
| `server/asgi.py` | uvicorn entry point |
| `indexer/openai_index_jira_tickets.py` | Batch indexing pipeline |
| `app/openai_chatbot.py` | Streamlit UI |

Both interfaces share one analysis path: the Streamlit app and the MCP server each call
`core/ticket_analyzer.py` and neither reimplements any of it. Anything that belongs to
the analysis belongs in `core/`, not in a UI or transport module.

### Directory Structure

```
Jira-AI-Ticket-Solver/
├── app/                    # Streamlit interface (openai_chatbot.py)
├── core/                   # Analysis pipeline + PII sanitizer (UI-agnostic)
├── server/                 # Remote MCP Server (FastAPI, auth, formatting)
├── config/                 # Configuration (Azure OpenAI, Weaviate, MCP settings)
├── indexer/                # Batch processing (indexer)
├── utils/                  # Shared utilities (processor, Jira and Weaviate clients)
├── tests/                  # deterministic / llm_eval / integration tiers
├── requirements.txt        # Dependencies
└── .env.example           # Environment template
```

## Configuration

### Settings File (`config/settings.py`)

All shared constants are centralized here:

```python
# Azure OpenAI Configuration (loaded from environment)
import os
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_LLM_DEPLOYMENT = os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-5-nano")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_OPENAI_TEMPERATURE = 1.0  # gpt-5-nano only supports 1.0

# Weaviate Configuration
JIRA_COLLECTION_NAME = "JiraCollection"
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")  # unset -> local Docker instance
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

# Remote MCP Server
MCP_ALLOWED_ORIGINS = [...]  # parsed from a comma-separated env var
MCP_SSE_KEEPALIVE_SECONDS = float(os.environ.get("MCP_SSE_KEEPALIVE_SECONDS", "15"))
MCP_SSE_STREAM_SECONDS = float(os.environ.get("MCP_SSE_STREAM_SECONDS", "300"))

# Generic Limits
MAX_EMBEDDINGS_INPUT_CHARS = 4000
LLM_CALL_TIMEOUT_SECONDS = 60

# Reranking
RERANK_SCORE_THRESHOLD = 5
MAX_SIMILAR_TICKETS_AFTER_RERANK = 5
```

**Guidelines:**
- Only shared/generic constants go in `settings.py`
- Module-specific constants stay in their respective files
- `settings.py` calls `load_dotenv()` at import, so importing it is enough to see `.env`
  from any entry point (Streamlit, the MCP server, the indexer, pytest)

## Running Locally

### Run the Indexer

```bash
python -m indexer.openai_index_jira_tickets

# To customize, modify the JQL in the indexer's main()
```

### Run the Chatbot

```bash
streamlit run app/openai_chatbot.py
```

### Run the MCP Server

```bash
uvicorn server.asgi:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# Liveness
curl http://localhost:8000/health

# List the tools the server exposes
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

A real tool call needs `X-Jira-Email` and `X-Jira-Token` headers. Terminate TLS in front
of the server in any shared deployment — user Jira tokens travel in those headers.

To watch the analysis progress instead of waiting several minutes in silence, add a
`_meta.progressToken` and read the response as a stream (`curl -N`):

```bash
curl -N -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H "X-Jira-Email: $EMAIL" -H "X-Jira-Token: $TOKEN" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
       "name":"analyze_ticket","arguments":{"ticket_key":"GC-1234"},
       "_meta":{"progressToken":"t1"}}}'
```

```
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"t1","progress":1,"message":"📥 **Analyzing ticket...**"}}
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"t1","progress":2,"message":"🔍 **Finding similar tickets...**"}}
data: {"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"# GC-1234: ..."}]}}
```

The token is echoed exactly as sent — Claude Code sends an integer, so don't coerce it
to a string. Omitting `_meta` keeps the single-JSON-body response.

### Verify Services

```bash
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
llm = AzureChatOpenAI(azure_deployment=deployment, ...)
structured_llm = llm.with_structured_output(LogAnalysisOutput)
result = structured_llm.invoke(messages)
```

### 2. Thread-Local LLM Instances

To avoid socket churn in multithreaded contexts:

```python
from threading import local

_thread_local = local()

def _get_llm(self) -> AzureChatOpenAI:
    if not hasattr(_thread_local, "azure_llm") or _thread_local.azure_llm is None:
        _thread_local.azure_llm = AzureChatOpenAI(...)
    return _thread_local.azure_llm
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

### Automated Tests

Three tiers, selected by pytest marker (`pytest.ini`). Only the first is free:

```bash
./run_tests.sh deterministic   # no LLM calls, safe for CI
./run_tests.sh llm             # real LLM calls, needs Azure OpenAI
./run_tests.sh integration     # full pipeline, needs Azure OpenAI
./run_tests.sh all             # every tier in sequence

python -m pytest -m deterministic -v          # same thing, directly
python -m pytest tests/unit/test_mcp_server.py -v
```

Read `tests/CLAUDE.md` before adding tests — it documents the layout, the two agreed MCP
server seams and the conventions (dataset-driven parametrization, hand-written fakes over
`MagicMock` for `create_app`, the secrets-must-not-leak assertion, covering both
`tools/call` response shapes).

Anything touching the server or the sanitizer should keep the deterministic tier green:

```bash
./run_tests.sh deterministic
```

### Manual Testing Workflow

1. **Test Indexer:**
   ```bash
   # Modify JQL to test with small dataset
   python -m indexer.openai_index_jira_tickets
   ```

2. **Test Chatbot:**
   ```bash
   streamlit run app/openai_chatbot.py
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

Go through the shared helper so you inspect the same instance the app uses — local Docker
when `WEAVIATE_URL` is unset, the configured remote when it is set:

```python
from utils.weaviate_client import connect_to_weaviate

client = connect_to_weaviate()
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

1. **Run the deterministic tier** - `./run_tests.sh deterministic` must be green
2. **Test your changes** locally with the chatbot interface, the MCP server, or both,
   depending on what you touched
3. **Verify no regressions** in existing functionality
4. **Update documentation** if adding new features or changing behavior
5. **Check for hardcoded values** - use `config/settings.py` for shared constants

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

1. Define in `openai_jira_ticket_processing.py`
2. Use `Field()` with descriptions for LLM structured output
3. Add type hints

```python
class NewOutputModel(BaseModel):
    field_name: str = Field(description="What this field contains")
    optional_field: Optional[int] = Field(default=None, description="Optional info")
```

### Adding a New Processing Step

1. Create processing method in `OpenAIJiraIssueLLMProcessor`
2. Add system/user prompt methods
3. Call from `process_issue()` pipeline
4. Update `_parse_final_issue_summary_output_to_text()` if output changes

### Modifying Weaviate Schema

1. Update `_setup_collection()` in the indexer file
2. Delete existing collection or use new name for testing
3. Re-index tickets

```python
# To delete existing collection (in Python REPL)
from utils.weaviate_client import connect_to_weaviate
client = connect_to_weaviate()
client.collections.delete("JiraCollection")
client.close()
```

### Adding a Field to `TicketAnalysis`

A new free-text field on `TicketAnalysis` (or on `SimilarTicket` / `ErrorLogHighlight`)
has to be classified in `core/pii_sanitizer.py`: add it to the sanitized tuples if it can
carry customer data, or to the structural tuples if it must survive character-for-character
(keys, statuses, timestamps, scores). The coverage test in
`tests/unit/test_pii_sanitizer.py` fails on any string field that is in neither, so an
unredacted field cannot ship silently. Then extend `server/response_formatter.py` if the
field should reach Claude.

### Updating Environment Variables

1. Add to `.env.example` with placeholder value
2. Read it in `config/settings.py` if it is shared, otherwise in the owning module
3. Document it in the README and in the ARCHITECTURE.md environment variable table

## Troubleshooting

### Weaviate Connection Issues

```bash
# Check if Weaviate is running
curl http://localhost:8080/v1/.well-known/ready

# Restart Weaviate container
docker restart <container_id>
```

### Memory Issues During Indexing

Reduce concurrent workers in `openai_index_jira_tickets.py`:

```python
MAX_PROCESS_TICKET_WORKERS = 2  # Reduce from 5
```

### LLM Timeout Issues

Increase timeout in `config/settings.py`:

```python
LLM_CALL_TIMEOUT_SECONDS = 120  # Increase from 60
```
