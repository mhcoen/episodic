"""
Tests for MCP dispatch wiring — verify dispatch_mcp() uses argument adapters.

These tests mock the MCP client to verify that:
1. Adapted args (not {}) are passed to call_tool
2. __user_id__ is injected for all gsuite tools
3. Query args are correctly translated per-adapter
4. Unknown commands pass args through unchanged
5. Full chain from CLI slash commands creates correct tool calls
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from episodic.mcp.dispatch import dispatch_mcp, MCPResolver
from episodic.mcp.dispatch_types import MCPResolution, DEFAULT_INTENT_MAPPING
from episodic.mcp.adapters import ARGUMENT_ADAPTERS
from episodic.mcp.adapters.base import DEFAULT_ACCOUNT
from episodic.utility.types import UtilityQuery


def _make_query(category, command, args=None, raw_input=""):
    return UtilityQuery(
        category=category,
        command=command,
        args=args or {},
        confidence=1.0,
        source="cli",
        raw_input=raw_input,
    )


def _make_resolution(tool_name, sensitivity="read"):
    return MCPResolution(
        server_id="gsuite",
        tool_name=tool_name,
        sensitivity=sensitivity,
        requires_auth_event=(sensitivity != "read"),
    )


def _mock_client(return_value=None):
    client = MagicMock()
    client.call_tool = AsyncMock(
        return_value=return_value or {"content": ["ok"], "is_error": False}
    )
    return client


def _call_args(mock_client):
    """Extract (tool_name, params) from mock call_tool invocation."""
    assert mock_client.call_tool.called, "call_tool was never called"
    args = mock_client.call_tool.call_args[0]
    return args[0], args[1]


# ============================================================
# Adapter wiring: dispatch_mcp passes adapted args, not {}
# ============================================================


class TestDispatchAdapterWiring:
    """Verify dispatch_mcp() wires through ARGUMENT_ADAPTERS."""

    def test_email_search_injects_user_id(self):
        """email.search → query_gmail_emails with __user_id__."""
        query = _make_query("email", "email.search", {"query": "budget"})
        resolution = _make_resolution("query_gmail_emails")
        client = _mock_client()

        result = asyncio.run(dispatch_mcp(
            query, resolution, "/email about budget", mcp_client=client,
        ))

        tool, params = _call_args(client)
        assert tool == "gsuite.query_gmail_emails"
        assert params["__user_id__"] == DEFAULT_ACCOUNT
        assert "budget" in params["query"]
        assert result.success

    def test_email_search_builds_gmail_query(self):
        """from_addr and unread_only are folded into Gmail query string."""
        query = _make_query("email", "email.search", {
            "from_addr": "alice@example.com",
            "unread_only": True,
        })
        resolution = _make_resolution("query_gmail_emails")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/email from alice", mcp_client=client,
        ))

        _, params = _call_args(client)
        assert "from:alice@example.com" in params["query"]
        assert "is:unread" in params["query"]

    def test_email_search_default_query(self):
        """Empty email.search produces is:unread default."""
        query = _make_query("email", "email.search", {})
        resolution = _make_resolution("query_gmail_emails")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/email", mcp_client=client,
        ))

        _, params = _call_args(client)
        assert params["query"] == "is:unread"

    def test_calendar_query_passes_time_range(self):
        """calendar.query → get_calendar_events with time_min/time_max."""
        query = _make_query("calendar", "calendar.query", {
            "time_min": "2026-02-13T00:00:00-06:00",
            "time_max": "2026-02-14T00:00:00-06:00",
        })
        resolution = _make_resolution("get_calendar_events")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/cal tomorrow", mcp_client=client,
        ))

        tool, params = _call_args(client)
        assert tool == "gsuite.get_calendar_events"
        assert params["__user_id__"] == DEFAULT_ACCOUNT
        assert params["time_min"] == "2026-02-13T00:00:00-06:00"
        assert params["time_max"] == "2026-02-14T00:00:00-06:00"

    def test_calendar_query_no_time_range(self):
        """calendar.query with no time args still sends __user_id__."""
        query = _make_query("calendar", "calendar.query", {"query": "tomorrow"})
        resolution = _make_resolution("get_calendar_events")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/cal tomorrow", mcp_client=client,
        ))

        _, params = _call_args(client)
        assert params["__user_id__"] == DEFAULT_ACCOUNT
        # query arg from CFG is not a gsuite field — adapter strips it
        assert "time_min" not in params

    def test_calendar_list_minimal_args(self):
        """calendar.list → list_calendars with just __user_id__."""
        query = _make_query("calendar", "calendar.list", {})
        resolution = _make_resolution("list_calendars")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/calendars", mcp_client=client,
        ))

        tool, params = _call_args(client)
        assert tool == "gsuite.list_calendars"
        assert params == {"__user_id__": DEFAULT_ACCOUNT}

    def test_email_draft_passes_fields(self):
        """email.create_draft → create_gmail_draft with to/subject/body."""
        query = _make_query("email", "email.create_draft", {
            "to": "bob@example.com",
            "subject": "hello",
            "body": "world",
        })
        resolution = _make_resolution("create_gmail_draft", "draft")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/draft to bob", mcp_client=client,
        ))

        _, params = _call_args(client)
        assert params["__user_id__"] == DEFAULT_ACCOUNT
        assert params["to"] == "bob@example.com"
        assert params["subject"] == "hello"
        assert params["body"] == "world"

    def test_unknown_command_passthrough(self):
        """Commands not in ARGUMENT_ADAPTERS pass args through unchanged."""
        query = _make_query("test", "test.unknown", {"foo": "bar", "n": 42})
        resolution = _make_resolution("test_tool")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/test", mcp_client=client,
        ))

        _, params = _call_args(client)
        assert params == {"foo": "bar", "n": 42}

    def test_no_args_no_adapter_passes_empty(self):
        """Unknown command with no args passes {}."""
        query = _make_query("test", "test.unknown", {})
        resolution = _make_resolution("test_tool")
        client = _mock_client()

        asyncio.run(dispatch_mcp(
            query, resolution, "/test", mcp_client=client,
        ))

        _, params = _call_args(client)
        assert params == {}

    def test_pipeline_receives_adapted_args(self):
        """Security pipeline check gets adapted args (not raw/empty args)."""
        query = _make_query("email", "email.search", {"query": "budget"})
        resolution = _make_resolution("query_gmail_emails")
        client = _mock_client()

        class _Pipeline:
            def __init__(self):
                self.captured_args = None

            def check_tool_execution(self, tool, args, ctx, auth_event=None):
                self.captured_args = args
                return MagicMock(allowed=True, reason="ok")

            def process_inbound(self, content, ctx):
                return MagicMock(content=content)

        pipe = _Pipeline()

        result = asyncio.run(dispatch_mcp(
            query, resolution, "/email about budget", mcp_client=client, pipeline=pipe
        ))

        assert result.success
        assert pipe.captured_args["__user_id__"] == DEFAULT_ACCOUNT
        assert "budget" in pipe.captured_args["query"]

    def test_action_gate_cancelled_blocks_execution(self):
        """Write/destructive dispatch respects action-gate cancellation."""
        query = _make_query("email", "email.create_draft", {"to": "a@b.com"})
        resolution = _make_resolution("create_gmail_draft", sensitivity="write")
        client = _mock_client()

        class _Pipeline:
            def check_tool_execution(self, tool, args, ctx, auth_event=None):
                return MagicMock(allowed=True, reason="ok")

            def process_inbound(self, content, ctx):
                return MagicMock(content=content)

        class _DenyConfirm:
            async def confirm(self, tool, args, context):
                return False

        result = asyncio.run(dispatch_mcp(
            query,
            resolution,
            "/draft to a@b.com",
            mcp_client=client,
            pipeline=_Pipeline(),
            confirm_handler=_DenyConfirm(),
        ))

        assert not result.success
        assert result.error_type == "cancelled"
        assert not client.call_tool.called


# ============================================================
# Resolver → dispatch_mcp integration
# ============================================================


class TestResolverAdapterConsistency:
    """Verify every intent in the resolver has a matching adapter."""

    def test_all_resolved_intents_have_adapters(self):
        """Every intent that resolves to a tool has an adapter."""
        resolver = MCPResolver()
        for intent in DEFAULT_INTENT_MAPPING:
            resolution = resolver.resolve(intent)
            if resolution is not None:
                assert intent in ARGUMENT_ADAPTERS, (
                    f"Intent '{intent}' resolves to tool "
                    f"'{resolution.tool_name}' but has no adapter"
                )

    def test_adapters_produce_user_id(self):
        """Every adapter injects __user_id__ for gsuite."""
        resolver = MCPResolver()
        for intent, adapter_cls in ARGUMENT_ADAPTERS.items():
            resolution = resolver.resolve(intent)
            if resolution is None:
                continue
            adapter = adapter_cls()
            result = adapter.adapt({}, resolution)
            assert "__user_id__" in result, (
                f"Adapter for '{intent}' does not inject __user_id__"
            )


# ============================================================
# Error handling
# ============================================================


class TestDispatchErrorHandling:
    """Verify dispatch_mcp handles errors gracefully."""

    def test_no_mcp_client(self):
        """None mcp_client returns not_connected error."""
        query = _make_query("email", "email.search", {})
        resolution = _make_resolution("query_gmail_emails")

        result = asyncio.run(dispatch_mcp(
            query, resolution, "/email", mcp_client=None,
        ))

        assert not result.success
        assert result.error_type == "not_connected"

    def test_client_error_response(self):
        """Client returning error dict is handled."""
        query = _make_query("email", "email.search", {})
        resolution = _make_resolution("query_gmail_emails")
        client = _mock_client(return_value={
            "error": "connection_failed",
            "message": "Failed to connect to gsuite",
        })

        result = asyncio.run(dispatch_mcp(
            query, resolution, "/email", mcp_client=client,
        ))

        assert not result.success
        assert result.error_type == "connection_failed"

    def test_client_exception(self):
        """Exception from call_tool is caught and sanitized."""
        query = _make_query("email", "email.search", {})
        resolution = _make_resolution("query_gmail_emails")
        client = MagicMock()
        client.call_tool = AsyncMock(side_effect=RuntimeError("boom"))

        result = asyncio.run(dispatch_mcp(
            query, resolution, "/email", mcp_client=client,
        ))

        assert not result.success
        assert result.error_type == "tool_error"


# ============================================================
# Result context update
# ============================================================


class TestResultContextUpdate:
    """Verify result context is updated from tool responses."""

    def test_email_search_updates_context(self):
        """Successful email.search updates anaphoric result context."""
        import json
        from episodic.mcp.dispatch import get_result_context

        query = _make_query("email", "email.search", {"query": "test"})
        resolution = _make_resolution("query_gmail_emails")
        client = _mock_client(return_value={
            "content": [json.dumps([{"id": "msg123", "subject": "Test"}])],
            "is_error": False,
        })

        asyncio.run(dispatch_mcp(
            query, resolution, "/email about test", mcp_client=client,
        ))

        ctx = get_result_context()
        # Context should have been updated with the email result
        assert ctx.last_emails  # populated by update_emails()

    def test_calendar_query_updates_context(self):
        """Successful calendar.query updates event result context."""
        import json
        from episodic.mcp.dispatch import get_result_context

        query = _make_query("calendar", "calendar.query", {})
        resolution = _make_resolution("get_calendar_events")
        client = _mock_client(return_value={
            "content": [json.dumps([{"id": "evt456", "summary": "Standup"}])],
            "is_error": False,
        })

        asyncio.run(dispatch_mcp(
            query, resolution, "/cal today", mcp_client=client,
        ))

        ctx = get_result_context()
        assert ctx.last_events  # populated by update_events()
