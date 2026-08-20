# Jira AI Ticket Solver

An AI-powered Jira ticket analysis platform that uses Azure OpenAI to provide intelligent insights, find similar tickets via vector similarity search, and suggest solutions based on historical data.

## Features

- **LLM-Based Ticket Analysis**: Generates structured summaries with root cause analysis and suggested solutions
- **Vector Similarity Search**: Finds related tickets using Weaviate vector database
- **Attachment Processing**:
  - Image analysis via VLM (screenshots, error dialogs)
  - Log file parsing with error extraction
  - Archive support (ZIP, TAR, RAR)
- **AI-Powered Root Cause Analysis**: Identifies likely causes based on ticket content and similar resolved tickets
- **Two Interfaces**: a Streamlit chatbot, and a Remote MCP Server that exposes the same analysis to Claude Code
- **PII Sanitization**: every MCP response passes an LLM redaction pass before it leaves the server

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│  ┌─────────────────────────────────┐  ┌───────────────────────────┐ │
│  │   openai_chatbot.py             │  │  server/mcp_server.py     │ │
│  │   (Streamlit + Azure OpenAI)    │  │  (MCP over HTTP, for      │ │
│  │                                 │  │   Claude Code)            │ │
│  └───────────────┬─────────────────┘  └─────────────┬─────────────┘ │
└──────────────────┼──────────────────────────────────┼───────────────┘
                   │                                  │
                   ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Analysis Layer                                │
│  ┌─────────────────────────────────┐  ┌───────────────────────────┐ │
│  │ core/ticket_analyzer.py         │  │ core/pii_sanitizer.py     │ │
│  │ (fetch, process, rerank)        │  │ (LLM redaction pass)      │ │
│  └───────────────┬─────────────────┘  └───────────────────────────┘ │
└──────────────────┼──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Processing Layer                                │
│  ┌─────────────────────────────────┐                                │
│  │ openai_jira_ticket_processing.py│                                │
│  │ (Azure OpenAI GPT / Embeddings) │                                │
│  └─────────────────────────────────┘                                │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │  Jira Cloud    │  │   Weaviate     │  │  Azure OpenAI API      │ │
│  │  REST API      │  │  JiraCollection│  │  (LLM/VLM/Embeddings)  │ │
│  └────────────────┘  └────────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Web UI | Streamlit |
| Vector Database | Weaviate (JiraCollection) |
| LLM | Azure OpenAI (configurable deployment) |
| VLM | Azure OpenAI (same deployment handles vision) |
| Embeddings | text-embedding-3-small (1536 dims, Azure) |
| Data Validation | Pydantic |

## Prerequisites

- Python 3.8+
- [Weaviate](https://weaviate.io/developers/weaviate/installation) (running locally)
- Azure OpenAI resource with API access
- Jira Cloud account with API access

### RAR File Support (Optional)

For processing RAR attachments:

```bash
# macOS
brew install rar

# Ubuntu/Debian
sudo apt-get install unrar
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/royho10/Jira-AI-Ticket-Solver.git
   cd Jira-AI-Ticket-Solver
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## Environment Variables

Create a `.env` file with the following:

```bash
# Required - Jira Configuration
ATLASSIAN_INSTANCE_URL=https://your-company.atlassian.net
ATLASSIAN_EMAIL=your-email@company.com
ATLASSIAN_API_TOKEN=your_api_token_here

# Required - Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### Getting Jira API Token

1. Go to [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
2. Click "Create API token"
3. Copy the token to your `.env` file

## Usage

### 1. Start Weaviate

Ensure Weaviate is running locally (default: `http://localhost:8080`):

```bash
# Using Docker
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

#### Using a shared remote Weaviate instead

Both the app and the indexer connect through the same helper, so pointing them
at a shared remote instance only takes environment variables:

```bash
WEAVIATE_URL=https://weaviate.your-company.internal   # https implies port 443 and TLS on gRPC
WEAVIATE_API_KEY=your_weaviate_api_key_here           # omit for an unauthenticated instance
WEAVIATE_GRPC_PORT=50051                              # override if your instance differs
```

A non-default HTTP port can be given inline (`https://host:9443`). When
`WEAVIATE_URL` is unset, the local Docker instance above is used — so no extra
configuration is needed for local development.

### 2. Index Your Jira Tickets

Build the vector database with your existing tickets:

```bash
python -m indexer.openai_index_jira_tickets
```

The indexer:
- Fetches tickets from Jira based on a configurable JQL query
- Processes attachments (logs, images) using Azure OpenAI LLM/VLM
- Generates embeddings and stores them in Weaviate
- Skips already indexed tickets to avoid duplicates

### 3. Run the Web Interface

```bash
streamlit run app/openai_chatbot.py
```

### 4. Analyze Tickets

1. Open the web interface in your browser (default: `http://localhost:8501`)
2. Enter a Jira ticket key (e.g., `PROJ-123`) or paste a ticket URL
3. Get AI-powered analysis including:
   - Ticket summary
   - Root cause analysis
   - Similar tickets from history
   - Suggested solutions

## Remote MCP Server

The same analysis is available to Claude Code through an MCP server, so nobody
needs a local checkout, a local Weaviate, or their own Azure OpenAI key.

### Running the server

```bash
uvicorn server.asgi:app --host 0.0.0.0 --port 8000
```

The server needs the Jira service account, Azure OpenAI and (in a shared
deployment) Weaviate variables from `.env.example`. Terminate TLS in front of it
— user Jira tokens travel in request headers.

Endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /mcp` | MCP JSON-RPC: `initialize`, `tools/list`, `tools/call` |
| `GET /mcp` | Server-initiated event stream (keep-alive only today) |
| `GET /health` | Liveness check for monitoring |

### Connecting from Claude Code

Add the server to `~/.claude/settings.json` with your own Jira credentials:

```json
{
  "mcpServers": {
    "jira-ticket-solver": {
      "type": "streamable-http",
      "url": "https://ticket-solver.your-company.internal/mcp",
      "headers": {
        "X-Jira-Email": "you@your-company.com",
        "X-Jira-Token": "your_jira_api_token"
      }
    }
  }
}
```

Then ask Claude to analyze a ticket — it calls `analyze_ticket(ticket_key)` and
gets back markdown plus a structured `<ticket_analysis>` block it can reason over
for follow-up questions.

### How credentials are handled

- Your token is used once per request, to confirm you can read the ticket you
  asked about. It is never logged, never stored, and never used to fetch data.
- The ticket itself is fetched with the team's Jira service account.
- Every response passes an LLM redaction pass (`core/pii_sanitizer.py`) that
  strips customer names, emails, IPs, hostnames and tenant identifiers. If that
  pass fails, the server returns an error rather than unredacted content.
- Requests that state a browser `Origin` must match `MCP_ALLOWED_ORIGINS`, which
  is what prevents DNS rebinding.
- The server does not write LLM prompts to disk, so raw ticket text is never
  persisted server-side.

### Optional server settings

```bash
MCP_ALLOWED_ORIGINS=https://claude.ai   # comma-separated; empty refuses any stated Origin
MCP_SSE_KEEPALIVE_SECONDS=15            # keep-alive interval on GET /mcp
MCP_SSE_STREAM_SECONDS=300              # how long GET /mcp stays open before reconnect
```

## Project Structure

```
Jira-AI-Ticket-Solver/
├── app/
│   └── openai_chatbot.py               # Streamlit chatbot (Azure OpenAI)
├── core/
│   ├── ticket_analyzer.py              # Stateless Analysis pipeline (UI-agnostic)
│   └── pii_sanitizer.py                # LLM redaction pass
├── server/
│   ├── mcp_server.py                   # MCP Streamable HTTP app (FastAPI)
│   ├── auth.py                         # Jira access check + Origin validation
│   ├── response_formatter.py           # Markdown + <ticket_analysis> rendering
│   └── asgi.py                         # uvicorn entry point
├── config/
│   └── settings.py                     # Centralized configuration
├── indexer/
│   └── openai_index_jira_tickets.py    # Batch indexing (Azure OpenAI)
├── utils/
│   ├── jira_client.py                  # Jira API client + Pydantic models
│   ├── weaviate_client.py              # Local/remote Weaviate connection
│   ├── openai_jira_ticket_processing.py # LLM/VLM processing (Azure OpenAI)
│   ├── llm_logger.py                   # LLM call logging
│   └── file_utils.py                   # Archive extraction utilities
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment template
├── ARCHITECTURE.md                     # Technical documentation
└── CONTRIBUTING.md                     # Developer guide
```

## Analysis Output

The tool provides structured analysis including:

- **Ticket Summary**: Concise overview of the issue
- **Key Issues**: Main problems identified in the ticket
- **Root Cause Analysis**: Likely causes based on evidence
- **Errors from Logs**: Extracted error messages with context
- **Similar Tickets**: Related tickets with similarity explanations
- **Suggested Solutions**: AI-recommended fixes based on historical resolutions

## License

This project is open source and available under the MIT License.
