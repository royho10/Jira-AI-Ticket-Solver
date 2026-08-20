# Architecture

This document provides detailed technical documentation for the Jira AI Ticket Solver project.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│                                                                              │
│  ┌───────────────────────────────────┐  ┌──────────────────────────────────┐│
│  │       openai_chatbot.py           │  │      server/mcp_server.py        ││
│  │  ┌─────────────────────────────┐  │  │  ┌────────────────────────────┐  ││
│  │  │     Streamlit UI            │  │  │  │ MCP Streamable HTTP        │  ││
│  │  └───────────┬─────────────────┘  │  │  │ POST /mcp | GET /mcp       │  ││
│  │              │                    │  │  │ GET /health                │  ││
│  │  ┌───────────▼─────────────────┐  │  │  └────────────┬───────────────┘  ││
│  │  │    OpenAIJiraChatBot        │  │  │  ┌────────────▼───────────────┐  ││
│  │  │  • Intent classification    │  │  │  │ server/auth.py             │  ││
│  │  │  • Renders the analysis     │  │  │  │  • Origin validation       │  ││
│  │  └─────────────────────────────┘  │  │  │  • Jira access check       │  ││
│  └───────────────────────────────────┘  │  └────────────────────────────┘  ││
│                                          └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                      │
                    ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ANALYSIS LAYER                                    │
│                                                                              │
│  ┌──────────────────────────────────┐  ┌───────────────────────────────────┐│
│  │  core/ticket_analyzer.py         │  │  core/pii_sanitizer.py            ││
│  │  TicketAnalyzer.analyze()        │  │  PiiSanitizer.sanitize()          ││
│  │  • Jira fetch (service account)  │  │  • One LLM redaction pass over    ││
│  │  • process_issue()               │  │    every free-text field          ││
│  │  • RAG query + LLM reranking     │  │  • Fails closed:                  ││
│  │  • Final analysis                │  │    SanitizationError              ││
│  │  -> TicketAnalysis               │  │                                   ││
│  └──────────────────────────────────┘  └───────────────────────────────────┘│
│                                          ┌───────────────────────────────────┐│
│                                          │  server/response_formatter.py    ││
│                                          │  markdown + <ticket_analysis>    ││
│                                          └───────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING LAYER                                   │
│                                                                              │
│  ┌──────────────────────────────────┐                                       │
│  │  OpenAIJiraIssueLLMProcessor     │                                       │
│  │ (openai_jira_ticket_processing.py)│                                       │
│  │                                  │                                       │
│  │  • AzureChatOpenAI (LLM)        │                                       │
│  │  • Vision (same deployment)      │                                       │
│  │  • AzureOpenAIEmbeddings         │                                       │
│  └──────────────────────────────────┘                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      file_utils.py                                      ││
│  │   extract_content_from_zip() | extract_content_from_tar()               ││
│  │   extract_content_from_rar()                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA LAYER                                       │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────────────────┐  ┌─────────────────┐  │
│  │   JiraClient    │  │        Weaviate             │  │ Azure OpenAI    │  │
│  │ (jira_client.py)│  │     (Vector Store)          │  │                 │  │
│  │                 │  │                             │  │ • gpt-4o-mini   │  │
│  │ • fetch_issues  │  │ • JiraCollection            │  │   (or custom)   │  │
│  │ • fetch_issue_  │  │   1536-dim vectors          │  │ • text-embed-3  │  │
│  │   by_key        │  │                             │  │   -small        │  │
│  │ • download_     │  │ • near_vector queries       │  │                 │  │
│  │   attachment    │  │ • insert_many               │  │                 │  │
│  └─────────────────┘  └─────────────────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### `utils/jira_client.py`

Jira REST API client with Pydantic data models.

**Classes:**
- `JiraComment` - Comment dataclass with author, body, timestamps
- `JiraAttachment` - Attachment dataclass with filename, MIME type, content URL
- `JiraRelatedIssue` - Linked issue with relation type (duplicate, relates to, etc.)
- `JiraIssue` - Main issue dataclass aggregating all ticket data
- `JiraClient` - HTTP client with session reuse for Jira API calls

**Key Functions:**
- `fetch_issues(jql, max_results, next_page_token)` - Paginated issue fetch
- `fetch_issue_by_key(issue_key)` - Single issue retrieval
- `download_attachment(attachment_id)` - Binary attachment download
- `extract_text_from_adf(node)` - Atlassian Document Format parser
- `extract_jira_keys_from_text(text)` - Regex-based key extraction

### `utils/openai_jira_ticket_processing.py`

LLM/VLM processing engine for ticket analysis using Azure OpenAI.

**Main Class: `OpenAIJiraIssueLLMProcessor`**

Initializes with deployment names and provides:
- `process_issue(jira_issue, status_callback)` - Main entry point
- Thread-local LLM/VLM instances to reduce socket churn

**Pydantic Output Models:**
```python
class ImageAnalysisOutput(BaseModel):
    error_messages: Optional[str]
    summary: str

class ErrorInLog(BaseModel):
    source_code_filename: Optional[str]
    error_lines: str
    context: str

class LogAnalysisOutput(BaseModel):
    log_filename: str
    errors: List[ErrorInLog]

class FinalIssueSummeryOutput(BaseModel):
    issue_summery: str
    main_issues: List[str]
    likely_root_causes: List[str]
    comments: str
```

**Processing Pipeline:**
1. Process attachments (images via VLM, logs via LLM)
2. Process comments (truncated to MAX_WORDS_IN_COMMENTS)
3. Process related issues
4. Generate final summary combining all components

### `utils/file_utils.py`

Archive extraction utilities for log file processing.

**Functions:**
- `extract_content_from_zip(file, max_files)` - ZIP archive extraction
- `extract_content_from_tar(file, suffix, max_files)` - TAR/GZ/TGZ extraction
- `extract_content_from_rar(file, max_files)` - RAR extraction (requires `unrar`)

Returns `List[Tuple[str, str]]` of `(content, filename)` pairs.

### `indexer/openai_index_jira_tickets.py`

Batch indexing pipeline for building the vector database.

**Main Class: `OpenAIJiraIndexer`**

**Methods:**
- `index_all(page_size, jql)` - Main indexing loop
- `_setup_collection()` - Weaviate schema creation
- `_get_existing_issue_keys()` - Duplicate detection
- `_prepare_issues_for_inserting_to_db(issues)` - Parallel processing
- `_insert_issues_data_objects_to_db(data_objects)` - Batch insertion

**Weaviate Schema:**
```python
properties=[
    Property(name="issue_key", data_type=DataType.TEXT),
    Property(name="summary", data_type=DataType.TEXT),
    Property(name="clean_description", data_type=DataType.TEXT),
    Property(name="title", data_type=DataType.TEXT),
    Property(name="issue_type", data_type=DataType.TEXT),
    Property(name="priority", data_type=DataType.TEXT),
    Property(name="labels", data_type=DataType.TEXT_ARRAY),
    Property(name="components", data_type=DataType.TEXT_ARRAY),
    Property(name="created", data_type=DataType.DATE),
    Property(name="status", data_type=DataType.TEXT),
]
```

### `utils/weaviate_client.py`

Single place that decides which Weaviate to talk to.

**Functions:**
- `connect_to_weaviate()` - returns a connected client. With `WEAVIATE_URL` unset it
  connects to the local Docker instance on the default ports; with it set it connects
  to that host, using `WEAVIATE_API_KEY` when one is configured and inferring TLS and
  port 443 from an `https://` URL. Both the app and the indexer go through here, so
  pointing the whole system at a shared instance is an environment change only.

### `core/ticket_analyzer.py`

The Analysis pipeline, with no UI framework and no request state. Both the Streamlit
app and the Remote MCP Server call it.

**Main class: `TicketAnalyzer`**
- `analyze(ticket_key, on_progress=None) -> TicketAnalysis` - fetch the ticket with the
  Jira service account, run `process_issue()`, embed, query Weaviate, rerank with the
  LLM, generate the final analysis, assemble the result. `on_progress` is an optional
  plain callable so callers can render status without the analyzer knowing about them.

**Result models:** `TicketAnalysis`, `SimilarTicket`, `ErrorLogHighlight`.

The reranked candidate list is the source of truth for *which* tickets are similar; the
LLM only supplies each similarity reason, so hallucinated ticket keys cannot enter the
result. A vector-search or reranking failure degrades the analysis rather than failing
the request.

### `core/pii_sanitizer.py`

Mandatory LLM redaction pass for anything leaving the Remote MCP Server.

**Main class: `PiiSanitizer`**
- `sanitize(analysis) -> TicketAnalysis` - flattens every free-text field (including
  attachment filenames) into numbered segments, asks the LLM for a redacted counterpart
  of each, and writes them back into a deep copy. Structural fields — ticket keys,
  statuses, priorities, timestamps, scores — are never submitted, so identifiers survive
  character-for-character.

**Fails closed.** If the LLM call errors, or any submitted segment comes back missing,
`SanitizationError` is raised and the server returns an error instead of partially
redacted text. The categories redacted are configurable via `PiiCategory`.

Adding a free-text field to `TicketAnalysis` means adding it to the sanitized tuples in
this module; a coverage test in `tests/unit/test_pii_sanitizer.py` fails if a new string
field is classified as neither sanitized nor structural.

### `server/mcp_server.py`

FastAPI application implementing the MCP Streamable HTTP transport.

**Factory: `create_app(analyzer, sanitizer, access_checker, allowed_origins, ...)`**
Collaborators are injected by tests and built from the environment in production.

**Endpoints:**
- `POST /mcp` - JSON-RPC 2.0: `initialize`, `ping`, `tools/list`, `tools/call`,
  `notifications/*` (202, no body). Unknown methods return -32601.
- `GET /mcp` - the server-initiated event stream. Every answer is produced inline on the
  POST, so this carries keep-alive comments only, and closes after
  `MCP_SSE_STREAM_SECONDS` so a vanished client cannot pin a worker.
- `GET /health` - liveness for monitoring.

**Single tool:** `analyze_ticket(ticket_key: str)`. A call runs
access check -> `TicketAnalyzer.analyze()` -> `PiiSanitizer.sanitize()` -> `format_analysis()`.
The analysis and the redaction pass are synchronous and can run for minutes, so both — and
the Jira access check — run in a worker thread via `run_in_threadpool`, keeping the event
loop free to answer `/health`.

**Auth and access failures are `isError` tool results at HTTP 200**, not 401/403: an MCP
client reads 401 on the MCP endpoint as an OAuth challenge and starts authorization
discovery, which this token-in-header MVP does not implement. Origin rejection is the
exception — it is a transport-level refusal and answers 403.

Every call writes one usage record to the `server.usage` logger (identity, ticket key,
outcome, duration). The token is never part of it.

### `server/auth.py`

**Functions:**
- `extract_credentials(headers)` - pulls `X-Jira-Email` / `X-Jira-Token`, raising
  `AuthError` when either is blank.
- `validate_origin(origin, allowed_origins)` - a *stated* Origin must be on the
  allowlist, which is what stops DNS rebinding. No Origin at all (Claude Code, curl) is
  allowed through.

**Class: `JiraAccessChecker`**
- `verify_access(email, token, ticket_key)` - builds a per-request `JiraClient` with the
  user's own credential, calls `can_access_issue()`, and closes the client in a `finally`
  so the token lives no longer than the check. Dual-token by design: the user's token
  proves access, the service account fetches the data.

Failures log the HTTP status or the exception type only — never the exception itself,
which can carry the request and therefore the token.

### `server/response_formatter.py`

- `format_analysis(analysis) -> str` - renders markdown for a human reader, then the same
  data as JSON inside `<ticket_analysis>` tags so Claude can reason over it in follow-up
  questions without another tool call.

### `server/asgi.py`

The uvicorn entry point (`uvicorn server.asgi:app`). Loads `.env`, disables the LLM call
log — its prompts carry raw, unsanitized ticket text from every user, which a shared
server must not persist — and builds the app.

### `app/openai_chatbot.py`

Streamlit web interface using Azure OpenAI. Delegates the whole analysis to
`core/ticket_analyzer.py` and owns only intent classification and rendering.

**Main Class: `OpenAIJiraChatBot`**

**Intent Classification:**
```python
class IntentClassification(Enum):
    FOLLOW_UP_ON_CURRENT_TICKET = "follow_up_on_current_ticket"
    ANALYZE_NEW_TICKET = "analyze_new_ticket"
    UNRELATED_CHAT = "unrelated_chat"
    MORE_THAN_ONE_KEY = "more_than_one_key"
```

**Flow:**
1. User enters ticket key/URL or question
2. Intent classification (heuristic + LLM fallback)
3. If new ticket: fetch from Jira, process, query RAG, generate analysis
4. If follow-up: continue conversation with context
5. Display structured analysis with similar tickets

**Configuration:**
- LLM: Configurable Azure deployment (default: `gpt-4o-mini`)
- Vision: Same deployment handles vision
- Embeddings: `text-embedding-3-small` (Azure deployment)

### `config/settings.py`

Centralized configuration constants.

```python
# Azure OpenAI Configuration (loaded from environment)
import os
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Azure OpenAI Deployment Names
AZURE_OPENAI_LLM_DEPLOYMENT = os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-5-nano")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Weaviate Configuration
JIRA_COLLECTION_NAME = "JiraCollection"
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")  # unset -> local Docker instance
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")

# Remote MCP Server
MCP_ALLOWED_ORIGINS = [...]  # parsed from a comma-separated env var

# Generic Limits
MAX_EMBEDDINGS_INPUT_CHARS = 4000
LLM_CALL_TIMEOUT_SECONDS = 60
```

## Data Flow

### Indexing Flow

```
Jira API ──fetch_issues()──> List[JiraIssue]
                                   │
                                   ▼
                     OpenAIJiraIssueLLMProcessor.process_issue()
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              Image VLM       Log LLM       Comments
              Analysis        Analysis      Processing
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
                          Final Summary (LLM)
                                   │
                                   ▼
                      AzureOpenAIEmbeddings.embed_query()
                                   │
                                   ▼
                      Weaviate.insert_many(DataObjects)
```

### Query Flow

```
User Input (ticket key/question)
           │
           ▼
    Intent Classification
           │
           ├──> ANALYZE_NEW_TICKET
           │         │
           │         ▼
           │    JiraClient.fetch_issue_by_key()
           │         │
           │         ▼
           │    OpenAIJiraIssueLLMProcessor.process_issue()
           │         │
           │         ▼
           │    Weaviate.near_vector(embedding)
           │         │
           │         ▼
           │    Generate Final Analysis (LLM)
           │
           ├──> FOLLOW_UP_ON_CURRENT_TICKET
           │         │
           │         ▼
           │    Continue conversation with context
           │
           └──> UNRELATED_CHAT
                     │
                     ▼
                General LLM response
```

### MCP Tool Call Flow

```
Claude Code ──POST /mcp {"method":"tools/call"}──> mcp_server
                                   │
                                   ▼
                       validate_origin(Origin)
                                   │
                                   ▼
             extract_credentials(X-Jira-Email, X-Jira-Token)
                                   │
                                   ▼
        JiraAccessChecker.verify_access()   (user token, then discarded)
                                   │
                                   ▼
        TicketAnalyzer.analyze()            (Jira service account)
                                   │
                                   ▼
        PiiSanitizer.sanitize()             (mandatory; fails closed)
                                   │
                                   ▼
        format_analysis()  ->  markdown + <ticket_analysis> JSON
                                   │
                                   ▼
                        usage record (identity, ticket, outcome)
```

## LLM Processing Pipeline

### Image Analysis (VLM)

1. Download attachment via Jira API
2. Convert to base64
3. Send to Azure OpenAI with structured output schema
4. Extract `ImageAnalysisOutput` (error_messages, summary)

### Log Analysis (LLM)

1. Extract log files from archives (ZIP/TAR/RAR)
2. Filter for ERROR lines with context (10 lines before, 20 after)
3. Cap at MAX_LINES_PER_LOG (50 lines)
4. Send to LLM with structured output schema
5. Extract `LogAnalysisOutput` with error details

### Final Summary Generation

1. Aggregate all processed components
2. Send to LLM with comprehensive prompt
3. Generate `FinalIssueSummeryOutput`:
   - Issue summary (4 sentences max)
   - Main issues list
   - Likely root causes
   - Comments summary

## File Type Support

### Log Files
- `.log`, `.txt`, `.out`, `.err`, `.trace`, `.debug`

### Archives
- `.zip` - Standard ZIP extraction
- `.tar`, `.gz`, `.tgz`, `.tar.gz` - TAR with optional gzip
- `.rar` - RAR extraction (requires system `unrar`)

### Images (VLM Analysis)
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`

## Threading Model

### Indexer (`openai_index_jira_tickets.py`)

Uses `ThreadPoolExecutor` for parallel ticket processing:

```python
MAX_PROCESS_TICKET_WORKERS = 5

with ThreadPoolExecutor(max_workers=MAX_PROCESS_TICKET_WORKERS) as executor:
    futures = [executor.submit(process_single_issue, issue) for issue in issues]
    for future in as_completed(futures):
        result = future.result()
```

### Thread-Local Resources

LLM/VLM clients are stored thread-locally to avoid socket churn:

```python
_thread_local = local()

def _get_llm(self) -> AzureChatOpenAI:
    if not hasattr(_thread_local, "azure_llm") or _thread_local.azure_llm is None:
        _thread_local.azure_llm = AzureChatOpenAI(...)
    return _thread_local.azure_llm
```

## Error Handling

### Network Errors
- `requests.exceptions.ConnectionError` - Azure OpenAI/Jira unavailable
- `requests.exceptions.Timeout` - API timeouts
- `requests.exceptions.HTTPError` - 404 for missing tickets

### Processing Errors
- Graceful degradation: skip failed attachments, continue processing
- Timeout protection: `LLM_CALL_TIMEOUT_SECONDS = 60`
- Retry logic with exponential backoff for LLM calls
- Structured error messages to UI

### Archive Extraction Errors
- Cleanup of temporary files in `finally` blocks
- Silent cleanup failures to preserve original exceptions

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ATLASSIAN_INSTANCE_URL` | Yes | Jira instance base URL |
| `ATLASSIAN_EMAIL` | Yes | Jira account email |
| `ATLASSIAN_API_TOKEN` | Yes | Jira API token |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | No | Azure API version (default: 2024-08-01-preview) |
| `AZURE_OPENAI_LLM_DEPLOYMENT` | No | LLM deployment name (default: gpt-5-nano) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | No | Embedding deployment name (default: text-embedding-3-small) |
| `WEAVIATE_URL` | No | Shared remote Weaviate URL. Unset means the local Docker instance |
| `WEAVIATE_API_KEY` | No | API key for the remote Weaviate, if it requires one |
| `WEAVIATE_GRPC_PORT` | No | gRPC port of the remote Weaviate (default: 50051) |
| `MCP_ALLOWED_ORIGINS` | No | Comma-separated browser Origins the MCP server accepts. Empty means no stated Origin is allowed |
| `MCP_SSE_KEEPALIVE_SECONDS` | No | Interval between keep-alive comments on `GET /mcp` (default: 15) |
| `MCP_SSE_STREAM_SECONDS` | No | How long `GET /mcp` stays open before the client must reconnect (default: 300) |

### Settings Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | (from env) | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | (from env) | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` | Azure API version |
| `AZURE_OPENAI_LLM_DEPLOYMENT` | `gpt-5-nano` | Azure LLM deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | Azure embeddings deployment (1536 dims) |
| `JIRA_COLLECTION_NAME` | `JiraCollection` | Weaviate collection name |
| `WEAVIATE_URL` | (from env) | Remote Weaviate URL, or `None` for local Docker |
| `WEAVIATE_API_KEY` | (from env) | Remote Weaviate API key |
| `WEAVIATE_GRPC_PORT` | `50051` | Remote Weaviate gRPC port |
| `MCP_ALLOWED_ORIGINS` | `[]` | Browser Origins the MCP server accepts |
| `MCP_SSE_KEEPALIVE_SECONDS` | `15` | `GET /mcp` keep-alive interval |
| `MCP_SSE_STREAM_SECONDS` | `300` | `GET /mcp` stream lifetime |
| `MAX_EMBEDDINGS_INPUT_CHARS` | `4000` | Max chars for embedding |
| `LLM_CALL_TIMEOUT_SECONDS` | `60` | LLM call timeout |
| `RERANK_SCORE_THRESHOLD` | `5` | Minimum 0-10 rerank score a similar ticket must reach |
| `MAX_SIMILAR_TICKETS_AFTER_RERANK` | `5` | Similar tickets kept after reranking |

### Processing Limits

| Constant | Value | Location |
|----------|-------|----------|
| `MAX_LOG_FILES_TO_PROCESS` | 20 | openai_jira_ticket_processing.py |
| `MAX_LINES_PER_LOG` | 50 | openai_jira_ticket_processing.py |
| `MAX_WORDS_IN_COMMENTS` | 400 | openai_jira_ticket_processing.py |
| `CONTEXT_LINES_BEFORE_ERROR` | 10 | openai_jira_ticket_processing.py |
| `CONTEXT_LINES_AFTER_ERROR` | 20 | openai_jira_ticket_processing.py |
| `WEAVIATE_BATCH_SIZE` | 100 | openai_index_jira_tickets.py |
| `MAX_PROCESS_TICKET_WORKERS` | 5 | openai_index_jira_tickets.py |
