# Architecture

This document provides detailed technical documentation for the Jira AI Ticket Solver project.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│                                                                              │
│  ┌───────────────────────────────┐   ┌───────────────────────────────────┐  │
│  │       chatbot.py              │   │       openai_chatbot.py           │  │
│  │  ┌─────────────────────────┐  │   │  ┌─────────────────────────────┐  │  │
│  │  │     Streamlit UI        │  │   │  │     Streamlit UI            │  │  │
│  │  └───────────┬─────────────┘  │   │  └───────────┬─────────────────┘  │  │
│  │              │                │   │              │                    │  │
│  │  ┌───────────▼─────────────┐  │   │  ┌───────────▼─────────────────┐  │  │
│  │  │    JiraChatBot          │  │   │  │    OpenAIJiraChatBot        │  │  │
│  │  │  • Intent classification│  │   │  │  • Intent classification    │  │  │
│  │  │  • RAG query (Ollama)   │  │   │  │  • RAG query (Azure OpenAI) │  │  │
│  │  │  • Final analysis       │  │   │  │  • Final analysis           │  │  │
│  │  └─────────────────────────┘  │   │  └─────────────────────────────┘  │  │
│  └───────────────────────────────┘   └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING LAYER                                   │
│                                                                              │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │     JiraIssueLLMProcessor        │  │  OpenAIJiraIssueLLMProcessor     │ │
│  │   (jira_ticket_processing.py)    │  │ (openai_jira_ticket_processing.py)│ │
│  │                                  │  │                                  │ │
│  │  • ChatOllama (llama3)           │  │  • AzureChatOpenAI (gpt-4o-mini) │ │
│  │  • VLM (qwen2.5vl:7b)            │  │  • Vision (gpt-4o-mini)          │ │
│  │  • OllamaEmbeddings              │  │  • AzureOpenAIEmbeddings         │ │
│  └──────────────────────────────────┘  └──────────────────────────────────┘ │
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
│  │   JiraClient    │  │        Weaviate             │  │ Ollama / Azure  │  │
│  │ (jira_client.py)│  │     (Vector Store)          │  │                 │  │
│  │                 │  │                             │  │ Ollama:         │  │
│  │ • fetch_issues  │  │ • JiraCollection (Ollama)   │  │ • llama3        │  │
│  │ • fetch_issue_  │  │   768-dim vectors           │  │ • qwen2.5vl     │  │
│  │   by_key        │  │                             │  │ • nomic-embed   │  │
│  │ • download_     │  │ • JiraCollectionOpenAI      │  │                 │  │
│  │   attachment    │  │   1536-dim vectors          │  │ Azure OpenAI:   │  │
│  │                 │  │                             │  │ • gpt-4o-mini   │  │
│  │                 │  │ • near_vector queries       │  │ • text-embed-3  │  │
│  │                 │  │ • insert_many               │  │   -small        │  │
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

### `utils/jira_ticket_processing.py`

LLM/VLM processing engine for ticket analysis.

**Main Class: `JiraIssueLLMProcessor`**

Initializes with model names and provides:
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

### `indexer/index_jira_tickets.py`

Batch indexing pipeline for building the vector database.

**Main Class: `JiraIndexer`**

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

### `app/chatbot.py`

Streamlit web interface using Weaviate and Ollama (fully local).

**Main Class: `JiraChatBot`**

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

### `app/openai_chatbot.py`

Streamlit web interface using Azure OpenAI (cloud-based).

**Main Class: `OpenAIJiraChatBot`**

Same architecture as `chatbot.py` but uses Azure OpenAI:
- `AzureChatOpenAI` instead of `ChatOllama`
- `AzureOpenAIEmbeddings` instead of `OllamaEmbeddings`
- Queries `JiraCollectionOpenAI` Weaviate collection
- Uses `OpenAIJiraIssueLLMProcessor` for ticket processing

**Configuration:**
- LLM: `gpt-4o-mini` (Azure deployment)
- Vision: `gpt-4o-mini` (same deployment handles vision)
- Embeddings: `text-embedding-3-small` (Azure deployment)

### `utils/openai_jira_ticket_processing.py`

Azure OpenAI version of the LLM/VLM processing engine.

**Main Class: `OpenAIJiraIssueLLMProcessor`**

Same interface as `JiraIssueLLMProcessor` but uses Azure OpenAI:
- `AzureChatOpenAI` for LLM calls
- `AzureOpenAIEmbeddings` for vector generation
- Reuses Pydantic models from `jira_ticket_processing.py`

### `indexer/openai_index_jira_tickets.py`

Azure OpenAI version of the batch indexing pipeline.

**Main Class: `OpenAIJiraIndexer`**

Same workflow as `JiraIndexer` but:
- Uses `AzureOpenAIEmbeddings` for vector generation
- Stores in `JiraCollectionOpenAI` Weaviate collection
- Uses `OpenAIJiraIssueLLMProcessor` for ticket processing

### `config/settings.py`

Centralized configuration constants.

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

# Azure OpenAI Deployment Names
AZURE_OPENAI_LLM_DEPLOYMENT = os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Weaviate Configuration
JIRA_COLLECTION_NAME = "JiraCollection"
OPENAI_JIRA_COLLECTION_NAME = "JiraCollectionOpenAI"

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
                     JiraIssueLLMProcessor.process_issue()
                     (or OpenAIJiraIssueLLMProcessor)
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
                      Embeddings.embed_query()
                      (Ollama or Azure OpenAI)
                                   │
                                   ▼
                      Weaviate.insert_many(DataObjects)
                      (JiraCollection or JiraCollectionOpenAI)
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
           │    JiraIssueLLMProcessor.process_issue()
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

## LLM Processing Pipeline

### Image Analysis (VLM)

1. Download attachment via Jira API
2. Convert to base64
3. Send to VLM with structured output schema
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

### Indexer (`index_jira_tickets.py`)

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

def _get_llm(self) -> ChatOllama:
    if not hasattr(_thread_local, "llm") or _thread_local.llm is None:
        _thread_local.llm = ChatOllama(...)
    return _thread_local.llm
```

## Error Handling

### Network Errors
- `requests.exceptions.ConnectionError` - Ollama/Jira unavailable
- `requests.exceptions.Timeout` - API timeouts
- `requests.exceptions.HTTPError` - 404 for missing tickets

### Processing Errors
- Graceful degradation: skip failed attachments, continue processing
- Timeout protection: `LLM_CALL_TIMEOUT_SECONDS = 60`
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
| `AZURE_OPENAI_ENDPOINT` | For Azure OpenAI | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | For Azure OpenAI | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | For Azure OpenAI | Azure API version (default: 2024-02-01) |
| `AZURE_OPENAI_LLM_DEPLOYMENT` | For Azure OpenAI | LLM deployment name (default: gpt-4o-mini) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | For Azure OpenAI | Embedding deployment name (default: text-embedding-3-small) |

### Settings Constants

**Ollama Configuration:**

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `nomic-embed-text-v2-moe` | Ollama embedding model (768 dims) |
| `VLM_MODEL_NAME` | `qwen2.5vl:7b` | Vision-language model |
| `LLM_MODEL_NAME` | `llama3` | Text generation model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `JIRA_COLLECTION_NAME` | `JiraCollection` | Weaviate collection for Ollama |

**Azure OpenAI Configuration:**

| Constant | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | (from env) | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | (from env) | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | `2024-02-01` | Azure API version |
| `AZURE_OPENAI_LLM_DEPLOYMENT` | `gpt-4o-mini` | Azure LLM deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | Azure embeddings deployment (1536 dims) |
| `OPENAI_JIRA_COLLECTION_NAME` | `JiraCollectionOpenAI` | Weaviate collection for Azure OpenAI |

**Generic:**

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_EMBEDDINGS_INPUT_CHARS` | `4000` | Max chars for embedding |
| `LLM_CALL_TIMEOUT_SECONDS` | `60` | LLM call timeout |

### Processing Limits

| Constant | Value | Location |
|----------|-------|----------|
| `MAX_LOG_FILES_TO_PROCESS` | 20 | jira_ticket_processing.py |
| `MAX_LINES_PER_LOG` | 50 | jira_ticket_processing.py |
| `MAX_WORDS_IN_COMMENTS` | 400 | jira_ticket_processing.py |
| `CONTEXT_LINES_BEFORE_ERROR` | 10 | jira_ticket_processing.py |
| `CONTEXT_LINES_AFTER_ERROR` | 20 | jira_ticket_processing.py |
| `WEAVIATE_BATCH_SIZE` | 100 | index_jira_tickets.py |
| `MAX_PROCESS_TICKET_WORKERS` | 5 | index_jira_tickets.py |
