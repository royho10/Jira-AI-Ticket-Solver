"""Weaviate connection resolution.

Single place that decides whether to talk to the shared remote Weaviate or to a
local Docker instance, so the indexer and the analyzer always agree.
"""
from typing import Optional
from urllib.parse import urlparse

import weaviate
from weaviate.classes.init import Auth

from config.settings import WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_GRPC_PORT

DEFAULT_HTTPS_PORT = 443
DEFAULT_HTTP_PORT = 80


def connect_to_weaviate(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    grpc_port: Optional[int] = None,
) -> weaviate.WeaviateClient:
    """Connect to the configured Weaviate instance.

    Falls back to local Docker when no remote URL is configured. Arguments
    override the configured settings.
    """
    resolved_url = url or WEAVIATE_URL
    if not resolved_url:
        return weaviate.connect_to_local()

    resolved_api_key = api_key if api_key is not None else WEAVIATE_API_KEY
    resolved_grpc_port = grpc_port or WEAVIATE_GRPC_PORT

    # Bare hostnames are treated as HTTPS; urlparse needs an explicit scheme.
    if "://" not in resolved_url:
        resolved_url = f"https://{resolved_url}"
    parsed = urlparse(resolved_url)

    secure = parsed.scheme == "https"
    http_port = parsed.port or (DEFAULT_HTTPS_PORT if secure else DEFAULT_HTTP_PORT)

    return weaviate.connect_to_custom(
        http_host=parsed.hostname,
        http_port=http_port,
        http_secure=secure,
        grpc_host=parsed.hostname,
        grpc_port=resolved_grpc_port,
        grpc_secure=secure,
        auth_credentials=Auth.api_key(resolved_api_key) if resolved_api_key else None,
    )
