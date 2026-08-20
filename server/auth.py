"""Authentication and authorization for the Remote MCP Server.

The user's Jira API token arrives in a request header, is used once to prove the
user may read the requested ticket, and is then discarded. It is never logged,
never persisted, and never handed to the Ticket Analyzer -- the analyzer uses the
team service account.
"""
import logging
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

import requests

from utils.jira_client import JiraClient

logger = logging.getLogger(__name__)

EMAIL_HEADER = "X-Jira-Email"
TOKEN_HEADER = "X-Jira-Token"


class AuthErrorCode(str, Enum):
    """Why a request was refused. Chooses the caller-facing message and status."""

    MISSING_CREDENTIALS = "missing_credentials"
    INVALID_CREDENTIALS = "invalid_credentials"
    ACCESS_DENIED = "access_denied"
    INVALID_ORIGIN = "invalid_origin"
    UPSTREAM_ERROR = "upstream_error"


class AuthError(Exception):
    """A request may not proceed. `message` is safe to return to the caller."""

    def __init__(self, code: AuthErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def extract_credentials(headers: Mapping[str, str]) -> Tuple[str, str]:
    """Pull the caller's Jira identity out of the request headers.

    Returns `(email, token)`. Raises `AuthError` when either is absent or blank,
    which is the common case of a user who has not filled in their MCP config.
    """
    email = (headers.get(EMAIL_HEADER) or "").strip()
    token = (headers.get(TOKEN_HEADER) or "").strip()
    if not email or not token:
        raise AuthError(
            AuthErrorCode.MISSING_CREDENTIALS,
            f"Both {EMAIL_HEADER} and {TOKEN_HEADER} headers are required. "
            "Add them to your MCP server configuration.",
        )
    return email, token


def validate_origin(origin: Optional[str], allowed_origins: Sequence[str]) -> None:
    """Reject browser requests from origins we do not know.

    A stated Origin that is not on the allowlist is refused; this is what stops a
    DNS rebinding attack from driving the server through a victim's browser.
    Requests with no Origin at all come from non-browser clients such as Claude
    Code and are allowed through.
    """
    if origin is None:
        return
    if origin not in allowed_origins:
        raise AuthError(
            AuthErrorCode.INVALID_ORIGIN,
            f"Origin '{origin}' is not allowed.",
        )


class JiraAccessChecker:
    """Verifies a user may read a ticket, using their own Jira credential."""

    def __init__(self, instance_url: Optional[str] = None):
        self.instance_url = instance_url

    def verify_access(self, email: str, token: str, ticket_key: str) -> None:
        """Raise `AuthError` unless `email`/`token` can read `ticket_key`.

        The per-request client is always closed, so the token lives no longer
        than the check itself.
        """
        client = None
        try:
            client = JiraClient(
                instance_url=self.instance_url,
                email=email,
                api_token=token,
            )
            allowed = client.can_access_issue(ticket_key)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401:
                raise AuthError(
                    AuthErrorCode.INVALID_CREDENTIALS,
                    "Jira rejected your credentials. Check your Jira email and API token.",
                ) from e
            logger.error(f"Jira access check failed with HTTP {status}")
            raise AuthError(
                AuthErrorCode.UPSTREAM_ERROR,
                "Could not verify Jira access right now. Please retry.",
            ) from e
        except Exception as e:
            # Deliberately logs the type only: the exception may carry the
            # request, and the request carries the token.
            logger.error(f"Jira access check failed ({type(e).__name__})")
            raise AuthError(
                AuthErrorCode.UPSTREAM_ERROR,
                "Could not verify Jira access right now. Please retry.",
            ) from e
        finally:
            if client is not None:
                client.close()

        if not allowed:
            raise AuthError(
                AuthErrorCode.ACCESS_DENIED,
                f"You do not have access to {ticket_key}, or it does not exist.",
            )
