# Jira AI Ticket Solver

An intelligent Jira ticket analysis tool that uses AI to provide insights, suggest solutions, and find similar tickets based on vector similarity search.

## Features

- **Ticket Analysis**: Analyzes Jira tickets and provides structured insights including root cause analysis and suggested solutions
- **Similar Ticket Discovery**: Uses vector embeddings to find similar tickets from your Jira history
- **AI-Powered Suggestions**: Leverages OpenAI's GPT models to provide intelligent recommendations
- **Web Interface**: Easy-to-use Streamlit web interface for ticket analysis

## Architecture

- **Jira Integration**: Custom Jira client for fetching ticket data
- **Vector Database**: Chroma DB for storing and searching ticket embeddings
- **AI Models**: OpenAI embeddings (text-embedding-3-large) and ChatGPT (gpt-4o-mini)
- **Web Interface**: Streamlit for the user interface

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Jira account with API access

### Installation

1. Clone the repository:
```bash
git clone https://github.com/royho10/Jira-AI-Ticket-Solver.git
cd Jira-AI-Ticket-Solver
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual API keys and configuration
```

### Configuration

Edit the `.env` file with your credentials:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ATLASSIAN_INSTANCE_URL`: Your Jira instance URL (e.g., https://company.atlassian.net)
- `ATLASSIAN_EMAIL`: Your Jira email
- `ATLASSIAN_API_TOKEN`: Your Jira API token

## Usage

### 1. Index Your Jira Tickets

First, index your existing Jira tickets to build the vector database:

```bash
python -m indexer.index_jira_tickets
```

### 2. Run the Web Interface

Start the Streamlit application:

```bash
streamlit run app/chatbot.py
```

### 3. Analyze Tickets

- Open the web interface in your browser
- Enter a Jira ticket URL or key (e.g., PROJ-123)
- Click "Analyze" to get AI-powered insights

## Project Structure

```
├── app/
│   └── chatbot.py          # Streamlit web interface
├── indexer/
│   └── index_jira_tickets.py  # Vector indexing for Jira tickets
├── utils/
│   └── jira_client.py      # Jira API client and data models
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── .gitignore            # Git ignore rules
```

## Features in Detail

### Ticket Analysis Output

The tool provides structured analysis including:

- **Summary**: Concise overview of the ticket
- **Root Cause Analysis**: Potential causes based on similar tickets
- **Similar Tickets**: Top 3 most similar tickets with explanations
- **Suggested Solution**: AI-recommended fix based on historical solutions
- **Confidence Level**: Assessment of suggestion reliability

### Similar Ticket Discovery

The system:
- Creates vector embeddings of ticket summaries and descriptions
- Prioritizes resolved tickets in similarity search
- Provides clickable links to original Jira tickets
- Shows resolution status for each similar ticket

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.
