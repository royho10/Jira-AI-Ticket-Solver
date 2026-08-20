"""Tests for shared/remote Weaviate connection resolution.

The indexer and the analyzer both need to reach a shared remote Weaviate, while
local Docker must keep working for local development.
"""
from unittest.mock import patch

import pytest

from utils import weaviate_client as weaviate_client_module
from utils.weaviate_client import connect_to_weaviate


@pytest.mark.deterministic
def test_connects_to_local_when_no_remote_url_configured():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", None), \
         patch.object(weaviate_client_module.weaviate, "connect_to_local") as mock_local, \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate()

    mock_local.assert_called_once()
    mock_custom.assert_not_called()


@pytest.mark.deterministic
def test_connects_to_remote_https_url_with_default_ports():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", "https://weaviate.example.com"), \
         patch.object(weaviate_client_module, "WEAVIATE_API_KEY", None), \
         patch.object(weaviate_client_module.weaviate, "connect_to_local") as mock_local, \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate()

    mock_local.assert_not_called()
    kwargs = mock_custom.call_args.kwargs
    assert kwargs["http_host"] == "weaviate.example.com"
    assert kwargs["http_port"] == 443
    assert kwargs["http_secure"] is True
    assert kwargs["grpc_host"] == "weaviate.example.com"
    assert kwargs["grpc_secure"] is True
    assert kwargs["auth_credentials"] is None


@pytest.mark.deterministic
def test_connects_to_remote_http_url_with_default_ports():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", "http://weaviate.internal"), \
         patch.object(weaviate_client_module, "WEAVIATE_API_KEY", None), \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate()

    kwargs = mock_custom.call_args.kwargs
    assert kwargs["http_host"] == "weaviate.internal"
    assert kwargs["http_port"] == 80
    assert kwargs["http_secure"] is False
    assert kwargs["grpc_secure"] is False


@pytest.mark.deterministic
def test_explicit_port_in_url_is_respected():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", "http://weaviate.internal:9090"), \
         patch.object(weaviate_client_module, "WEAVIATE_API_KEY", None), \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate()

    assert mock_custom.call_args.kwargs["http_port"] == 9090


@pytest.mark.deterministic
def test_url_without_scheme_defaults_to_https():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", "weaviate.example.com"), \
         patch.object(weaviate_client_module, "WEAVIATE_API_KEY", None), \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate()

    kwargs = mock_custom.call_args.kwargs
    assert kwargs["http_host"] == "weaviate.example.com"
    assert kwargs["http_secure"] is True


@pytest.mark.deterministic
def test_api_key_is_passed_as_auth_credentials():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", "https://weaviate.example.com"), \
         patch.object(weaviate_client_module, "WEAVIATE_API_KEY", "secret-key"), \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate()

    assert mock_custom.call_args.kwargs["auth_credentials"] is not None


@pytest.mark.deterministic
def test_explicit_arguments_override_configured_settings():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", None), \
         patch.object(weaviate_client_module.weaviate, "connect_to_local") as mock_local, \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate(url="https://override.example.com")

    mock_local.assert_not_called()
    assert mock_custom.call_args.kwargs["http_host"] == "override.example.com"


@pytest.mark.deterministic
def test_grpc_port_is_configurable():
    with patch.object(weaviate_client_module, "WEAVIATE_URL", "https://weaviate.example.com"), \
         patch.object(weaviate_client_module, "WEAVIATE_API_KEY", None), \
         patch.object(weaviate_client_module.weaviate, "connect_to_custom") as mock_custom:
        connect_to_weaviate(grpc_port=12345)

    assert mock_custom.call_args.kwargs["grpc_port"] == 12345
