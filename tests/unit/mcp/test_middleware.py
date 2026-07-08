"""Tests for episodic.mcp.middleware module."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from episodic.mcp.auth import _ensure_tables
from episodic.mcp.middleware import (
    PUBLIC_PATHS,
    get_client_id,
    get_scopes,
    has_scope,
)


class TestPublicPaths:
    """Only the health check is public; everything else requires a token."""

    def test_health_is_public(self):
        assert "/health" in PUBLIC_PATHS

    def test_sse_requires_auth(self):
        # The SSE stream endpoint must NOT be public, or unauthenticated
        # clients could open/exhaust event streams.
        assert "/sse" not in PUBLIC_PATHS
        assert "/sse/" not in PUBLIC_PATHS

    def test_messages_not_public(self):
        assert "/messages" not in PUBLIC_PATHS


class TestHelperFunctions:
    """Tests for get_client_id, get_scopes, has_scope."""

    def test_get_client_id_from_state(self):
        request = MagicMock()
        request.state.client_id = "my-client"
        assert get_client_id(request) == "my-client"

    def test_get_client_id_missing(self):
        request = MagicMock(spec=[])
        assert get_client_id(request) is None

    def test_get_scopes_from_state(self):
        request = MagicMock()
        request.state.scopes = ["read", "write"]
        assert get_scopes(request) == ["read", "write"]

    def test_get_scopes_missing(self):
        request = MagicMock(spec=[])
        assert get_scopes(request) == []

    def test_has_scope_present(self):
        request = MagicMock()
        request.state.scopes = ["read", "write"]
        assert has_scope(request, "read") is True

    def test_has_scope_absent(self):
        request = MagicMock()
        request.state.scopes = ["read"]
        assert has_scope(request, "write") is False

    def test_has_scope_empty_means_all(self):
        request = MagicMock()
        request.state.scopes = []
        assert has_scope(request, "anything") is True


class TestCreateAuthMiddleware:
    """Tests for create_auth_middleware (requires starlette)."""

    def test_create_middleware_without_starlette(self, tmp_path):
        """Importing the module works without starlette installed."""
        # The module itself should import fine — starlette is only
        # needed when create_auth_middleware is called.
        from episodic.mcp.middleware import create_auth_middleware
        # We can't call it without starlette, but import works
        assert callable(create_auth_middleware)
