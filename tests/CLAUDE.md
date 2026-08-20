# Tests

## Structure

```
tests/
├── conftest.py              # Shared fixtures, dataset loaders, LLM judge, make_jira_issue factory
├── datasets/                # JSON test datasets (parametrized test cases)
│   ├── error_extraction.json
│   ├── log_parsing.json
│   ├── final_summary.json
│   ├── intent_classification.json
│   ├── reranking.json
│   ├── final_analysis.json
│   ├── pii_sanitization.json
│   └── integration.json
├── unit/                    # Unit tests (deterministic + llm_eval)
│   ├── test_helpers.py          # ADF parsing, JiraIssue.from_dict, file utils
│   ├── test_error_extraction.py # _extract_errors_from_description
│   ├── test_log_parsing.py      # _filter_noise_from_logs, _summarize_log
│   ├── test_final_summary.py    # _create_final_issue_summary, comment/related processing
│   ├── test_intent_classification.py  # Rule-based intent classification
│   ├── test_reranking.py        # Reranking output
│   ├── test_final_analysis.py   # Final analysis output
│   ├── test_jira_client_credentials.py # Injectable credentials + access probe
│   ├── test_weaviate_connection.py     # Local vs remote Weaviate wiring
│   ├── test_llm_logger.py       # The prompt log's off switch (used by the server)
│   ├── test_ticket_analyzer.py  # Analysis pipeline orchestration (services mocked)
│   ├── test_pii_sanitizer.py    # Redaction mechanics and fail-closed behaviour
│   └── test_mcp_server.py       # Remote MCP Server via FastAPI TestClient
├── integration/             # End-to-end pipeline tests
│   └── test_full_pipeline.py    # JiraIssue -> process_issue -> validate output
└── eval/                    # LLM-as-Judge quality evaluation
    ├── test_llm_quality.py      # Rubric-based scoring via separate LLM judge
    └── test_pii_sanitization.py # Real-LLM redaction against curated PII examples
```

## Test Seams

Two agreed seams cover the Remote MCP Server:

- **`POST /mcp` via `TestClient`** (`unit/test_mcp_server.py`) is the primary seam.
  Protocol handling, auth, analysis orchestration, sanitization and response
  formatting are all asserted through it, with Jira/LLM/Weaviate mocked behind the
  endpoint. Prefer adding tests here over reaching into the server modules.
- **The PII Sanitizer module** (`unit/test_pii_sanitizer.py` +
  `eval/test_pii_sanitization.py`) gets its own seam because a silent failure there
  leaks customer data into a user's Claude context.

## Test Tiers (pytest markers)

| Marker          | Cost   | Speed | Description                                      |
|-----------------|--------|-------|--------------------------------------------------|
| `deterministic` | Free   | Fast  | No LLM calls. Schema validation, keyword checks, rule logic |
| `llm_eval`      | $$     | Slow  | Real LLM calls. Validates output quality with assertions + LLM judge |
| `integration`   | $$     | Slow  | Full pipeline end-to-end. Requires Azure OpenAI env vars |

## Running Tests

```bash
# Deterministic only (fast, free, safe for CI)
./run_tests.sh deterministic
python -m pytest -m deterministic -v

# LLM eval tests (costs money, needs Azure OpenAI)
./run_tests.sh llm

# Integration tests (costs money, needs Azure OpenAI)
./run_tests.sh integration

# All tiers
./run_tests.sh all
```

## Key Patterns

- **Dataset-driven**: Tests are parametrized from JSON files in `tests/datasets/`. Each JSON has `input`, `expected`, and optionally `rubric` fields. Use `dataset_range("name")` for `@pytest.mark.parametrize`.
- **Processor instantiation without `__init__`**: Deterministic tests use `OpenAIJiraIssueLLMProcessor.__new__(OpenAIJiraIssueLLMProcessor)` to skip `__init__` (avoids requiring Azure env vars). Only set attributes needed for the specific method under test.
- **Mock LLM**: Unit tests mock via `patch.object(processor, "_get_llm", return_value=mock_llm)` where `mock_llm.with_structured_output().invoke()` returns Pydantic model instances.
- **Mock JiraClient**: Integration tests use `patch.object(processor, "jira_client", MagicMock())` to avoid real Jira API calls.
- **LLM-as-Judge**: `llm_judge` fixture in conftest scores output against a rubric dict, returns `{"scores": {...}, "average_score": float}`. Pass threshold is 3.0/5.
- **`make_jira_issue` factory fixture**: Builds `JiraIssue` objects from dicts with sensible defaults. Used across integration and eval tests.
- **Session-scoped expensive fixtures**: `llm_processor`, `llm_chat`, `llm_judge` are `scope="session"` to reuse across all tests.
- **MCP server tests**: `create_app(analyzer=..., sanitizer=..., access_checker=...)` takes hand-written fakes (not `MagicMock`) so each test can assert what the collaborator was called with; drive it with `TestClient` and JSON-RPC bodies.
- **Secrets must not leak**: tests that send a Jira token assert it appears in neither the response body nor `caplog`. Keep that assertion in any new test that carries credentials.
- **Auth failures are `isError` tool results, not HTTP 4xx**: use the `assert_tool_error(response)` helper. A 401 on `/mcp` would send a real MCP client into OAuth discovery, so only Origin rejection answers with a status code (403).
- **"Did it block the event loop?"**: `ran_off_the_event_loop()` in `test_mcp_server.py` calls `asyncio.get_running_loop()` from inside a fake collaborator — it only succeeds on the loop's own thread, so it is an exact check. Don't try to prove this with concurrent `TestClient` requests: starlette spins a fresh event loop per request, so such a test passes even when the offload is removed.

## Adding New Tests

1. **Deterministic test**: Add to the appropriate `unit/test_*.py`, mark with `@pytest.mark.deterministic`.
2. **Dataset-driven test**: Add cases to the relevant JSON in `tests/datasets/`, tests auto-parametrize.
3. **New LLM feature test**: Add both deterministic tests (mock LLM, validate structure) and `@pytest.mark.llm_eval` tests (real LLM, validate quality).
4. **New dataset file**: Add a loader fixture in `conftest.py` following the existing pattern.

## Environment Requirements

- `deterministic` tests: No external services needed.
- `llm_eval` and `integration` tests: Require Azure OpenAI env vars (`AZURE_OPENAI_LLM_DEPLOYMENT`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_KEY`).
