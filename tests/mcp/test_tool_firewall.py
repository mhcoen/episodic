"""Tests for MCP security PolicyEngine (tool firewall)."""

import time

import pytest

from episodic.mcp.security.policy_engine import DESTRUCTIVE_TOOLS, PolicyEngine
from episodic.mcp.security.types import (
    AuthorizationEvent,
    PolicyConfig,
    SecurityContext,
    TrustLevel,
)


@pytest.fixture
def engine():
    return PolicyEngine()


@pytest.fixture
def ctx():
    return SecurityContext(
        mode="client",
        source_type="mcp_server",
        source_id="server-1",
        policy=PolicyConfig(enable_destructive=True),
    )


@pytest.fixture
def ctx_no_destructive():
    return SecurityContext(
        mode="client",
        source_type="mcp_server",
        source_id="server-1",
        policy=PolicyConfig(enable_destructive=False),
    )


_SENTINEL = object()


def _make_auth(
    action="gmail_send_email",
    scope=_SENTINEL,
    timestamp=None,
):
    return AuthorizationEvent(
        action=action,
        scope={"to": "alice@example.com"} if scope is _SENTINEL else scope,
        message_hash="msg_hash_abc",
        timestamp=timestamp or time.time(),
        source="slash_command",
    )


# ---------------------------------------------------------------------------
# Client mode
# ---------------------------------------------------------------------------


class TestPolicyEngineClient:
    def test_non_destructive_without_auth_allows(self, engine, ctx):
        """Non-destructive tools are implicitly allowed without auth."""
        result = engine.check_tool_execution(
            tool="gmail_read_email",
            args={},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is True
        assert result.confidence == "implicit"

    def test_destructive_without_auth_denies(self, engine, ctx):
        """Destructive tools require auth event."""
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "alice@example.com"},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "requires authorization" in result.reason

    def test_destructive_not_enabled_denies(self, engine, ctx_no_destructive):
        """Destructive tools must be enabled in policy."""
        auth = _make_auth()
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "alice@example.com"},
            auth_event=auth,
            ctx=ctx_no_destructive,
        )
        assert result.allowed is False
        assert "not enabled" in result.reason

    def test_matching_auth_allows(self, engine, ctx):
        """Correct auth event allows execution."""
        auth = _make_auth()
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "alice@example.com"},
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is True
        assert result.confidence == "explicit"
        assert result.matched_directive == "slash_command"

    def test_action_mismatch_denies(self, engine, ctx):
        """Auth event action must match the tool."""
        auth = _make_auth(action="slack_send_message")
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "alice@example.com"},
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "Action mismatch" in result.reason

    def test_scope_key_missing_denies(self, engine, ctx):
        """Scope key must exist in args."""
        auth = _make_auth(scope={"to": "alice@example.com"})
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"body": "Hello"},  # missing "to"
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "missing from args" in result.reason

    def test_scope_value_mismatch_denies(self, engine, ctx):
        """Scope value must match."""
        auth = _make_auth(scope={"to": "alice@example.com"})
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "eve@example.com"},
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "Scope violation" in result.reason

    def test_expired_auth_denies(self, engine, ctx):
        """Expired authorization is denied."""
        auth = _make_auth(timestamp=time.time() - 600)
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "alice@example.com"},
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "expired" in result.reason

    def test_custom_ttl(self, ctx):
        """Custom TTL is respected."""
        engine = PolicyEngine(auth_ttl=10.0)
        auth = _make_auth(timestamp=time.time() - 15)
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "alice@example.com"},
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is False

    def test_non_destructive_with_auth_verifies(self, engine, ctx):
        """Non-destructive tool with auth still verifies the auth."""
        auth = _make_auth(action="gmail_read_email", scope={})
        result = engine.check_tool_execution(
            tool="gmail_read_email",
            args={},
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is True
        assert result.confidence == "explicit"

    def test_empty_scope_allows(self, engine, ctx):
        """Auth with empty scope has no scope constraints."""
        auth = _make_auth(scope={})
        result = engine.check_tool_execution(
            tool="gmail_send_email",
            args={"to": "anyone@example.com", "body": "hi"},
            auth_event=auth,
            ctx=ctx,
        )
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Server mode
# ---------------------------------------------------------------------------


class TestPolicyEngineServer:
    def test_server_allows_when_no_scopes(self, engine):
        ctx = SecurityContext(
            mode="server",
            source_type="mcp_client",
            source_id="client-1",
            policy=PolicyConfig(),
        )
        result = engine.check_tool_execution(
            tool="episodic_search",
            args={"query": "test"},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is True
        assert result.confidence == "implicit"

    def test_server_denies_tool_not_in_scopes(self, engine):
        ctx = SecurityContext(
            mode="server",
            source_type="mcp_client",
            source_id="client-1",
            policy=PolicyConfig(
                capability_scopes={"episodic_search": {}, "episodic_chat": {}}
            ),
        )
        result = engine.check_tool_execution(
            tool="episodic_delete_all",
            args={},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "not in capability scopes" in result.reason

    def test_server_allows_tool_in_scopes(self, engine):
        ctx = SecurityContext(
            mode="server",
            source_type="mcp_client",
            source_id="client-1",
            policy=PolicyConfig(
                capability_scopes={"episodic_search": {}}
            ),
        )
        result = engine.check_tool_execution(
            tool="episodic_search",
            args={"query": "test"},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is True

    def test_server_destructive_not_enabled_denies(self, engine):
        ctx = SecurityContext(
            mode="server",
            source_type="mcp_client",
            source_id="client-1",
            policy=PolicyConfig(enable_destructive=False),
        )
        result = engine.check_tool_execution(
            tool="file_delete",
            args={"path": "/tmp/test"},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "not enabled" in result.reason

    def test_server_destructive_enabled_allows(self, engine):
        ctx = SecurityContext(
            mode="server",
            source_type="mcp_client",
            source_id="client-1",
            policy=PolicyConfig(enable_destructive=True),
        )
        result = engine.check_tool_execution(
            tool="file_delete",
            args={"path": "/tmp/test"},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Custom destructive tools
# ---------------------------------------------------------------------------


class TestPolicyEngineCustomDestructive:
    def test_custom_destructive_set(self):
        engine = PolicyEngine(destructive_tools=frozenset({"my_custom_tool"}))
        ctx = SecurityContext(
            mode="client",
            source_type="mcp_server",
            source_id="server-1",
            policy=PolicyConfig(enable_destructive=True),
        )
        result = engine.check_tool_execution(
            tool="my_custom_tool",
            args={},
            auth_event=None,
            ctx=ctx,
        )
        assert result.allowed is False
        assert "requires authorization" in result.reason

    def test_default_destructive_set_contains_expected(self):
        assert "gmail_send_email" in DESTRUCTIVE_TOOLS
        assert "file_delete" in DESTRUCTIVE_TOOLS
        assert "shell_execute" in DESTRUCTIVE_TOOLS
