"""
E2E tests for MCP dispatch — full chain from slash command to tool call.

Tests the complete flow:
  /cal tomorrow → parse_cal_args → UtilityQuery → async_dispatch_utility
    → MCPResolver → dispatch_mcp → adapter.adapt → client.call_tool

Uses a mock MCP client injected at the client manager level to verify
the entire wiring without needing a live server.

Tests marked mcp_live require a running mcp-gsuite server and are
skipped by default. Run with: pytest -m mcp_live
"""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from episodic.mcp.adapters.base import DEFAULT_ACCOUNT
from episodic.utility.types import ResultStatus


def _mock_manager(return_value=None):
    """Create a mock MCPClientManager."""
    mgr = MagicMock()
    mgr.call_tool = AsyncMock(
        return_value=return_value or {"content": ["ok"], "is_error": False},
    )
    return mgr


def _tool_call_args(mock_mgr):
    """Extract (namespaced_tool, params) from mock call."""
    assert mock_mgr.call_tool.called, "call_tool was never called"
    return mock_mgr.call_tool.call_args[0]


def _init_test_db():
    """Create in-memory DB with utility schema for event logging."""
    conn = sqlite3.connect(":memory:")
    from episodic.utility.db import init_utility_schema
    init_utility_schema(conn)
    return conn


# ============================================================
# Full chain: slash command → async_dispatch_utility → tool call
# ============================================================


class TestE2ECalendarSlashCommands:
    """Full chain from /cal and /calendars to mock tool call."""

    def _dispatch(self, mock_mgr, query):
        """Run async_dispatch_utility with mock manager."""
        from episodic.utility.dispatcher import async_dispatch_utility

        conn = _init_test_db()
        result = asyncio.run(async_dispatch_utility(
            query, conn=conn, user_tz="America/Chicago",
            mcp_client=mock_mgr,
        ))
        conn.close()
        return result

    def test_cal_tomorrow(self):
        """/cal tomorrow → get_calendar_events with query='tomorrow'."""
        from episodic.utility.cli_slash_calendar_email import parse_cal_args

        query = parse_cal_args("tomorrow")
        mock_mgr = _mock_manager({"events": [], "content": ["No events"]})

        result = self._dispatch(mock_mgr, query)

        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.get_calendar_events"
        assert params["__user_id__"] == DEFAULT_ACCOUNT
        assert result.status == ResultStatus.OK

    def test_cal_no_args(self):
        """/cal with no args → calendar.query with empty query."""
        from episodic.utility.cli_slash_calendar_email import parse_cal_args

        query = parse_cal_args("")
        mock_mgr = _mock_manager({"events": [], "content": ["No events"]})

        result = self._dispatch(mock_mgr, query)

        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.get_calendar_events"
        assert params["__user_id__"] == DEFAULT_ACCOUNT
        assert result.status == ResultStatus.OK

    def test_calendars_list(self):
        """/calendars → list_calendars with just __user_id__."""
        from episodic.utility.dispatcher import create_utility_query

        query = create_utility_query(
            "calendar", "calendar.list",
            args={}, source="cli", raw_input="/calendars",
        )
        mock_mgr = _mock_manager({
            "calendars": [{"id": "primary", "summary": "Main"}],
            "content": ["Main"],
        })

        result = self._dispatch(mock_mgr, query)

        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.list_calendars"
        assert params == {"__user_id__": DEFAULT_ACCOUNT}
        assert result.status == ResultStatus.OK


class TestE2EEmailSlashCommands:
    """Full chain from /email, /inbox, /draft to mock tool call."""

    def _dispatch(self, mock_mgr, query):
        from episodic.utility.dispatcher import async_dispatch_utility

        conn = _init_test_db()
        result = asyncio.run(async_dispatch_utility(
            query, conn=conn, user_tz="America/Chicago",
            mcp_client=mock_mgr,
        ))
        conn.close()
        return result

    def test_email_recent(self):
        """/email recent → query_gmail_emails with query='recent'."""
        from episodic.utility.cli_slash_calendar_email import parse_email_args

        query = parse_email_args("recent")
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})

        result = self._dispatch(mock_mgr, query)

        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.query_gmail_emails"
        assert params["__user_id__"] == DEFAULT_ACCOUNT
        assert "recent" in params["query"]
        assert result.status == ResultStatus.OK

    def test_email_from_filter(self):
        """/email from alice → query with from:alice."""
        from episodic.utility.cli_slash_calendar_email import parse_email_args

        query = parse_email_args("from alice@example.com")
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})

        self._dispatch(mock_mgr, query)

        _, params = _tool_call_args(mock_mgr)
        assert "from:alice@example.com" in params["query"]

    def test_email_unread(self):
        """/email unread → query with is:unread."""
        from episodic.utility.cli_slash_calendar_email import parse_email_args

        query = parse_email_args("unread")
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})

        self._dispatch(mock_mgr, query)

        _, params = _tool_call_args(mock_mgr)
        assert "is:unread" in params["query"]

    def test_inbox(self):
        """/inbox → email.search with unread_only=True."""
        from episodic.utility.dispatcher import create_utility_query

        query = create_utility_query(
            "email", "email.search",
            args={"unread_only": True}, source="cli",
            raw_input="/inbox",
        )
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})

        self._dispatch(mock_mgr, query)

        _, params = _tool_call_args(mock_mgr)
        assert "is:unread" in params["query"]

    def test_draft_to_about(self):
        """/draft to bob about lunch → create_gmail_draft with to/subject."""
        from episodic.utility.cli_slash_calendar_email import parse_draft_args

        query = parse_draft_args("to bob@test.com about lunch plans")
        mock_mgr = _mock_manager({"content": ["Draft created"]})

        self._dispatch(mock_mgr, query)

        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.create_gmail_draft"
        assert params["to"] == "bob@test.com"
        assert params["subject"] == "lunch plans"

    def test_email_no_args_default(self):
        """/email with no args → default is:unread query."""
        from episodic.utility.cli_slash_calendar_email import parse_email_args

        query = parse_email_args("")
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})

        self._dispatch(mock_mgr, query)

        _, params = _tool_call_args(mock_mgr)
        assert params["query"] == "is:unread"


# ============================================================
# Full chain via handle_utility_command (sync CLI entry point)
# ============================================================


class TestE2EHandleUtilityCommand:
    """Test handle_utility_command() with mocked MCP client manager.

    Patches _execute_async_utility_query to inject a mock manager and
    use an in-memory DB, avoiding the real episodic database.
    """

    def _run(self, cmd, args_str, mock_mgr):
        """Run handle_utility_command with patched async dispatch."""
        import episodic.utility.cli_integration as cli_mod

        def _mock_async_dispatch(query):
            """Replace _execute_async_utility_query with direct async call."""
            from episodic.utility.dispatcher import async_dispatch_utility
            conn = _init_test_db()
            result = asyncio.run(async_dispatch_utility(
                query, conn=conn, user_tz="America/Chicago",
                mcp_client=mock_mgr,
            ))
            conn.close()
            return result

        with patch.object(cli_mod, "_execute_async_utility_query",
                          side_effect=_mock_async_dispatch):
            return cli_mod.handle_utility_command(cmd, args_str)

    def test_cal_via_handle(self):
        """/cal tomorrow routed through handle_utility_command."""
        mock_mgr = _mock_manager({"events": [], "content": ["No events"]})
        result = self._run("cal", "tomorrow", mock_mgr)

        assert result is not None
        assert result.status == ResultStatus.OK
        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.get_calendar_events"
        assert params["__user_id__"] == DEFAULT_ACCOUNT

    def test_email_via_handle(self):
        """/email unread routed through handle_utility_command."""
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})
        result = self._run("email", "unread", mock_mgr)

        assert result is not None
        assert result.status == ResultStatus.OK
        _, params = _tool_call_args(mock_mgr)
        assert "is:unread" in params["query"]

    def test_inbox_via_handle(self):
        """/inbox routed through handle_utility_command."""
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})
        result = self._run("inbox", "", mock_mgr)

        assert result is not None
        assert result.status == ResultStatus.OK

    def test_calendars_via_handle(self):
        """/calendars routed through handle_utility_command."""
        mock_mgr = _mock_manager({
            "calendars": [{"id": "primary"}],
            "content": ["Primary"],
        })
        result = self._run("calendars", "", mock_mgr)

        assert result is not None
        assert result.status == ResultStatus.OK
        tool, _ = _tool_call_args(mock_mgr)
        assert tool == "gsuite.list_calendars"

    def test_schedule_via_handle(self):
        """/schedule meeting routed through handle_utility_command."""
        mock_mgr = _mock_manager({"content": ["Event created"]})
        result = self._run("schedule", "team standup", mock_mgr)

        assert result is not None
        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.create_calendar_event"
        assert params.get("summary") == "team standup"

    def test_draft_via_handle(self):
        """/draft to X about Y routed through handle_utility_command."""
        mock_mgr = _mock_manager({"content": ["Draft created"]})
        result = self._run("draft", "to bob@test.com about project update", mock_mgr)

        assert result is not None
        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.create_gmail_draft"
        assert params["to"] == "bob@test.com"

    def test_mail_alias(self):
        """/mail is an alias for /email."""
        mock_mgr = _mock_manager({"emails": [], "content": ["No emails"]})
        result = self._run("mail", "unread", mock_mgr)

        assert result is not None
        assert result.status == ResultStatus.OK

    def test_forward_via_handle(self):
        """/forward to X routed through handle_utility_command."""
        mock_mgr = _mock_manager({"content": ["Forwarded"]})
        result = self._run("forward", "to carol@test.com", mock_mgr)

        assert result is not None
        tool, params = _tool_call_args(mock_mgr)
        assert params.get("to") == "carol@test.com"

    def test_reply_via_handle(self):
        """/reply body text routed through handle_utility_command."""
        mock_mgr = _mock_manager({"content": ["Replied"]})
        result = self._run("reply", "sounds good", mock_mgr)

        assert result is not None
        tool, params = _tool_call_args(mock_mgr)
        assert tool == "gsuite.reply_gmail_email"
        assert params.get("body") == "sounds good"


# ============================================================
# Live MCP server tests (require real mcp-gsuite)
# ============================================================


@pytest.mark.mcp_live
class TestLiveMCPGsuite:
    """
    E2E tests against the real mcp-gsuite server.

    These tests launch the actual MCP server, connect, and make real
    API calls to Google Calendar and Gmail. They require:
    - uvx available in PATH
    - mcp-gsuite credentials in ~/.episodic/mcp/gsuite/
    - Network access to Google APIs

    Run with: pytest -m mcp_live -v
    """

    @pytest.fixture(scope="class")
    def manager(self):
        """Create and connect to the real MCP server."""
        from episodic.mcp.client_manager import MCPClientManager
        mgr = MCPClientManager()

        async def _connect():
            ok = await mgr.connect("gsuite")
            if not ok:
                pytest.skip("Could not connect to gsuite MCP server")
            return mgr

        yield asyncio.run(_connect())

        asyncio.run(mgr.disconnect_all())

    def test_tool_discovery(self, manager):
        """Server exposes expected tools."""
        tools = manager.get_all_tools()
        tool_names = {t["namespaced_name"] for t in tools}
        assert "gsuite.list_calendars" in tool_names
        assert "gsuite.get_calendar_events" in tool_names
        assert "gsuite.query_gmail_emails" in tool_names

    def test_list_calendars(self, manager):
        """list_calendars returns calendar data."""
        result = asyncio.run(
            manager.call_tool("gsuite.list_calendars", {
                "__user_id__": DEFAULT_ACCOUNT,
            })
        )
        assert "error" not in result
        assert "content" in result

    def test_query_calendar_events(self, manager):
        """get_calendar_events returns without error."""
        result = asyncio.run(
            manager.call_tool("gsuite.get_calendar_events", {
                "__user_id__": DEFAULT_ACCOUNT,
            })
        )
        assert "error" not in result
        assert "content" in result

    def test_query_gmail_emails(self, manager):
        """query_gmail_emails returns email data."""
        result = asyncio.run(
            manager.call_tool("gsuite.query_gmail_emails", {
                "__user_id__": DEFAULT_ACCOUNT,
                "query": "is:unread",
                "max_results": 3,
            })
        )
        assert "error" not in result
        assert "content" in result

    def test_full_chain_cal(self, manager):
        """Full dispatch chain: /cal → adapted args → real API."""
        from episodic.utility.cli_slash_calendar_email import parse_cal_args
        from episodic.utility.dispatcher import async_dispatch_utility

        query = parse_cal_args("today")
        conn = _init_test_db()
        result = asyncio.run(async_dispatch_utility(
            query, conn=conn, mcp_client=manager,
        ))
        conn.close()
        assert result.status == ResultStatus.OK

    def test_full_chain_email(self, manager):
        """Full dispatch chain: /email → adapted args → real API."""
        from episodic.utility.cli_slash_calendar_email import parse_email_args
        from episodic.utility.dispatcher import async_dispatch_utility

        query = parse_email_args("unread")
        conn = _init_test_db()
        result = asyncio.run(async_dispatch_utility(
            query, conn=conn, mcp_client=manager,
        ))
        conn.close()
        assert result.status == ResultStatus.OK
