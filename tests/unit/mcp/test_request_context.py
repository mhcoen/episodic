"""Tests for MCP request ContextVar propagation helpers."""

from episodic.mcp.request_context import (
    get_current_client_id,
    get_current_scopes,
    get_current_token_id,
    reset_request_context,
    set_request_context,
)


def test_set_and_get_request_context():
    tokens = set_request_context("client-a", "token-a", ["read", "write"])
    try:
        assert get_current_client_id() == "client-a"
        assert get_current_token_id() == "token-a"
        assert get_current_scopes() == ["read", "write"]
    finally:
        reset_request_context(tokens)


def test_reset_request_context_restores_defaults():
    tokens = set_request_context("client-a", "token-a", ["read"])
    reset_request_context(tokens)
    assert get_current_client_id() is None
    assert get_current_token_id() is None
    assert get_current_scopes() == []
