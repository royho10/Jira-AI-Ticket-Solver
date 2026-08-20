"""Tests for injectable JiraClient credentials.

The MCP server needs two credential sets live in one process: the user's Jira
API token (for access checks) and the team service account (for data fetching).
These tests pin the constructor contract that makes that possible.
"""
import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from utils import jira_client as jira_client_module
from utils.jira_client import JiraClient


@pytest.mark.deterministic
def test_explicit_credentials_are_used_over_environment():
    client = JiraClient(
        instance_url="https://explicit.atlassian.net",
        email="explicit@example.com",
        api_token="explicit-token",
    )

    assert client.base_url == "https://explicit.atlassian.net"
    assert client.auth.username == "explicit@example.com"
    assert client.auth.password == "explicit-token"


@pytest.mark.deterministic
def test_credentials_fall_back_to_module_defaults():
    with patch.object(jira_client_module, "ATLASSIAN_INSTANCE_URL", "https://env.atlassian.net"), \
         patch.object(jira_client_module, "ATLASSIAN_EMAIL", "env@example.com"), \
         patch.object(jira_client_module, "ATLASSIAN_TOKEN", "env-token"):
        client = JiraClient()

    assert client.base_url == "https://env.atlassian.net"
    assert client.auth.username == "env@example.com"
    assert client.auth.password == "env-token"


@pytest.mark.deterministic
def test_partial_explicit_credentials_fall_back_for_the_rest():
    """A caller supplying only a token (the access-check case) still gets the
    configured instance URL."""
    with patch.object(jira_client_module, "ATLASSIAN_INSTANCE_URL", "https://env.atlassian.net"), \
         patch.object(jira_client_module, "ATLASSIAN_EMAIL", "env@example.com"), \
         patch.object(jira_client_module, "ATLASSIAN_TOKEN", "env-token"):
        client = JiraClient(email="user@example.com", api_token="user-token")

    assert client.base_url == "https://env.atlassian.net"
    assert client.auth.username == "user@example.com"
    assert client.auth.password == "user-token"


@pytest.mark.deterministic
def test_missing_credentials_raise_at_instantiation():
    with patch.object(jira_client_module, "ATLASSIAN_INSTANCE_URL", None), \
         patch.object(jira_client_module, "ATLASSIAN_EMAIL", None), \
         patch.object(jira_client_module, "ATLASSIAN_TOKEN", None):
        with pytest.raises(ValueError) as exc_info:
            JiraClient()

    message = str(exc_info.value)
    assert "ATLASSIAN_INSTANCE_URL" in message
    assert "ATLASSIAN_EMAIL" in message
    assert "ATLASSIAN_API_TOKEN" in message


@pytest.mark.deterministic
def test_error_message_names_only_the_missing_credentials():
    with patch.object(jira_client_module, "ATLASSIAN_INSTANCE_URL", "https://env.atlassian.net"), \
         patch.object(jira_client_module, "ATLASSIAN_EMAIL", "env@example.com"), \
         patch.object(jira_client_module, "ATLASSIAN_TOKEN", None):
        with pytest.raises(ValueError) as exc_info:
            JiraClient()

    message = str(exc_info.value)
    assert "ATLASSIAN_API_TOKEN" in message
    assert "ATLASSIAN_EMAIL" not in message


@pytest.mark.deterministic
def test_module_imports_without_jira_environment_variables():
    """Importing the module must not crash when env vars are absent -- the MCP
    server imports this module before any credentials are resolved."""
    jira_env_vars = ["ATLASSIAN_INSTANCE_URL", "ATLASSIAN_EMAIL", "ATLASSIAN_API_TOKEN"]
    saved = {k: os.environ.get(k) for k in jira_env_vars}
    try:
        for key in jira_env_vars:
            os.environ.pop(key, None)
        # Patch at the source module: reload re-runs `from dotenv import
        # load_dotenv`, so patching the already-bound name would be overwritten.
        with patch("dotenv.load_dotenv", lambda *a, **kw: None):
            reloaded = importlib.reload(jira_client_module)
        assert reloaded.ATLASSIAN_INSTANCE_URL is None
        assert reloaded.ATLASSIAN_EMAIL is None
        assert reloaded.ATLASSIAN_TOKEN is None
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        importlib.reload(jira_client_module)


# ---------------------------------------------------------------------------
# The lightweight access probe used by the MCP server's auth check
# ---------------------------------------------------------------------------
def make_probe_client(status_code: int) -> tuple:
    client = JiraClient(
        instance_url="https://example.atlassian.net",
        email="user@example.com",
        api_token="user-token",
    )
    response = requests.Response()
    response.status_code = status_code
    client._session = MagicMock()
    client._session.get.return_value = response
    return client, client._session


@pytest.mark.deterministic
def test_access_probe_requests_only_the_key_field():
    """A permission probe must not drag down comments and attachments."""
    client, session = make_probe_client(200)

    assert client.can_access_issue("GC-100") is True

    url, kwargs = session.get.call_args.args[0], session.get.call_args.kwargs
    assert url.endswith("/rest/api/3/issue/GC-100")
    assert kwargs["params"] == {"fields": "key"}


@pytest.mark.deterministic
@pytest.mark.parametrize("status_code", [403, 404])
def test_access_probe_reports_no_access_for_forbidden_and_missing(status_code):
    """Jira answers 404 for issues the caller may not see, so both mean 'no'."""
    client, _ = make_probe_client(status_code)

    assert client.can_access_issue("GC-100") is False


@pytest.mark.deterministic
def test_access_probe_raises_for_a_rejected_credential():
    client, _ = make_probe_client(401)

    with pytest.raises(requests.exceptions.HTTPError):
        client.can_access_issue("GC-100")


@pytest.mark.deterministic
def test_access_probe_raises_for_an_upstream_failure():
    client, _ = make_probe_client(503)

    with pytest.raises(requests.exceptions.HTTPError):
        client.can_access_issue("GC-100")
