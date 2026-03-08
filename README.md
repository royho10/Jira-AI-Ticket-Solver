# Jira AI Ticket Solver

An AI-powered Jira ticket analysis platform that uses LLMs to provide intelligent insights, find similar tickets via vector similarity search, and suggest solutions based on historical data.

## Features

- **LLM-Based Ticket Analysis**: Generates structured summaries with root cause analysis and suggested solutions
- **Vector Similarity Search**: Finds related tickets using Weaviate vector database
- **Attachment Processing**:
  - Image analysis via VLM (screenshots, error dialogs)
  - Log file parsing with error extraction
  - Archive support (ZIP, TAR, RAR)
- **AI-Powered Root Cause Analysis**: Identifies likely causes based on ticket content and similar resolved tickets
- **Two Web Interfaces**:
  - Ollama chatbot using Weaviate + Ollama (fully local)
  - Azure OpenAI chatbot using Weaviate + Azure OpenAI (cloud-based)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐ │
│  │   chatbot.py            │    │   openai_chatbot.py             │ │
│  │   (Streamlit + Ollama)  │    │   (Streamlit + Azure OpenAI)    │ │
│  └───────────┬─────────────┘    └───────────────┬─────────────────┘ │
└──────────────┼──────────────────────────────────┼───────────────────┘
               │                                  │
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Processing Layer                                │
│  ┌───────────────────────────┐  ┌─────────────────────────────────┐ │
│  │ jira_ticket_processing.py │  │ openai_jira_ticket_processing.py│ │
│  │ (Ollama LLM/VLM)          │  │ (OpenAI GPT-4o-mini)            │ │
│  └───────────────────────────┘  └─────────────────────────────────┘ │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │  Jira Cloud    │  │   Weaviate     │  │  Ollama / OpenAI API   │ │
│  │  REST API      │  │  JiraCollection│  │  (LLM/VLM/Embeddings)  │ │
│  │                │  │  + OpenAI ver. │  │                        │ │
│  └────────────────┘  └────────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Ollama Version | Azure OpenAI Version |
|-----------|----------------|----------------------|
| Web UI | Streamlit | Streamlit |
| Vector Database | Weaviate (JiraCollection) | Weaviate (JiraCollectionOpenAI) |
| LLM | llama3 | gpt-4o-mini (Azure deployment) |
| VLM | qwen2.5vl:7b | gpt-4o-mini (Azure deployment) |
| Embeddings | nomic-embed-text (768 dims) | text-embedding-3-small (1536 dims, Azure) |
| Data Validation | Pydantic | Pydantic |

## Prerequisites

- Python 3.8+
- [Weaviate](https://weaviate.io/developers/weaviate/installation) (running locally)
- [Ollama](https://ollama.ai/) with required models
- Jira Cloud account with API access

### Required Ollama Models

```bash
# Pull the required models
ollama pull llama3                    # LLM for text processing
ollama pull qwen2.5vl:7b              # VLM for image analysis
ollama pull nomic-embed-text          # Embeddings model
```

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

# Required for Azure OpenAI chatbot and indexer
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

### 2. Start Ollama

```bash
ollama serve
```

### 3. Index Your Jira Tickets

Build the vector database with your existing tickets:

**Option A: Ollama Indexer (fully local)**
```bash
python -m indexer.index_jira_tickets
```

**Option B: Azure OpenAI Indexer (requires Azure OpenAI credentials)**
```bash
python -m indexer.openai_index_jira_tickets
```

The indexers:
- Fetch tickets from Jira based on a configurable JQL query
- Process attachments (logs, images) using LLM/VLM
- Generate embeddings and store them in Weaviate
- Skip already indexed tickets to avoid duplicates
- Use separate Weaviate collections (Ollama: `JiraCollection`, OpenAI: `JiraCollectionOpenAI`)

### 4. Run the Web Interface

**Option A: Ollama Chatbot (fully local)**
```bash
streamlit run app/chatbot.py
```

**Option B: Azure OpenAI Chatbot (requires Azure OpenAI credentials)**
```bash
streamlit run app/openai_chatbot.py
```

**Note:** Each chatbot queries its corresponding Weaviate collection, so make sure you've indexed tickets with the matching indexer first.

### 5. Analyze Tickets

1. Open the web interface in your browser (default: `http://localhost:8501`)
2. Enter a Jira ticket key (e.g., `PROJ-123`) or paste a ticket URL
3. Get AI-powered analysis including:
   - Ticket summary
   - Root cause analysis
   - Similar tickets from history
   - Suggested solutions

## Project Structure

```
Jira-AI-Ticket-Solver/
├── app/
│   ├── chatbot.py                      # Streamlit chatbot (Ollama)
│   └── openai_chatbot.py               # Streamlit chatbot (Azure OpenAI)
├── config/
│   └── settings.py                     # Centralized configuration
├── indexer/
│   ├── index_jira_tickets.py           # Batch indexing (Ollama)
│   └── openai_index_jira_tickets.py    # Batch indexing (Azure OpenAI)
├── utils/
│   ├── jira_client.py                  # Jira API client + Pydantic models
│   ├── jira_ticket_processing.py       # LLM/VLM processing (Ollama)
│   ├── openai_jira_ticket_processing.py # LLM/VLM processing (Azure OpenAI)
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
