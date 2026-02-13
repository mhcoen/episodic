"""Per-request MCP auth context propagated via ContextVar."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Dict, Optional


_CLIENT_ID: ContextVar[Optional[str]] = ContextVar("mcp_client_id", default=None)
_TOKEN_ID: ContextVar[Optional[str]] = ContextVar("mcp_token_id", default=None)
_SCOPES: ContextVar[list[str]] = ContextVar("mcp_scopes", default=[])


def set_request_context(
    client_id: Optional[str],
    token_id: Optional[str],
    scopes: Optional[list[str]],
) -> Dict[str, Token]:
    """Set request-scoped auth context and return reset tokens."""
    return {
        "client_id": _CLIENT_ID.set(client_id),
        "token_id": _TOKEN_ID.set(token_id),
        "scopes": _SCOPES.set(scopes or []),
    }


def reset_request_context(tokens: Dict[str, Token]) -> None:
    """Reset context vars using tokens returned from set_request_context()."""
    _CLIENT_ID.reset(tokens["client_id"])
    _TOKEN_ID.reset(tokens["token_id"])
    _SCOPES.reset(tokens["scopes"])


def get_current_client_id() -> Optional[str]:
    """Get request-scoped client_id, if present."""
    return _CLIENT_ID.get()


def get_current_token_id() -> Optional[str]:
    """Get request-scoped token_id, if present."""
    return _TOKEN_ID.get()


def get_current_scopes() -> list[str]:
    """Get request-scoped token scopes, if present."""
    return _SCOPES.get()
