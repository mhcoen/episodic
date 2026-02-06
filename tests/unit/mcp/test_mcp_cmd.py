"""Tests for episodic.commands.mcp_cmd module."""

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from episodic.commands.mcp_cmd import (
    _format_uptime,
    _check_mcp_available,
    mcp_command,
    mcp_token,
    mcp_servers,
    mcp_connect,
    mcp_disconnect,
    mcp_tools,
)


class TestFormatUptime:
    """Tests for _format_uptime."""

    def test_seconds_only(self):
        assert _format_uptime(30) == "30s"
        assert _format_uptime(0) == "0s"
        assert _format_uptime(59) == "59s"

    def test_minutes_and_seconds(self):
        assert _format_uptime(90) == "1m 30s"
        assert _format_uptime(60) == "1m"
        assert _format_uptime(3599) == "59m 59s"

    def test_hours_and_minutes(self):
        assert _format_uptime(3600) == "1h"
        assert _format_uptime(3660) == "1h 1m"
        assert _format_uptime(7200) == "2h"

    def test_days(self):
        assert _format_uptime(86400) == "1d"
        assert _format_uptime(90000) == "1d 1h"

    def test_float_input(self):
        assert _format_uptime(30.7) == "30s"


class TestCheckMcpAvailable:
    """Tests for _check_mcp_available."""

    def test_available(self):
        with patch.dict("sys.modules", {"mcp": MagicMock()}):
            assert _check_mcp_available() is True

    def test_not_available(self):
        with patch.dict("sys.modules", {"mcp": None}):
            # importlib raises ImportError for None modules
            assert _check_mcp_available() is False


class TestMcpCommand:
    """Tests for mcp_command routing."""

    @patch("episodic.commands.mcp_cmd.mcp_status")
    def test_no_action_shows_status(self, mock_status):
        mcp_command()
        mock_status.assert_called_once()

    @patch("episodic.commands.mcp_cmd.mcp_status")
    def test_status_action(self, mock_status):
        mcp_command("status")
        mock_status.assert_called_once()

    @patch("episodic.commands.mcp_cmd.mcp_start")
    def test_start_action(self, mock_start):
        mcp_command("start")
        mock_start.assert_called_once_with([])

    @patch("episodic.commands.mcp_cmd.mcp_start")
    def test_start_with_args(self, mock_start):
        mcp_command("start", "--port", "9999")
        mock_start.assert_called_once_with(["--port", "9999"])

    @patch("episodic.commands.mcp_cmd.mcp_stop")
    def test_stop_action(self, mock_stop):
        mcp_command("stop")
        mock_stop.assert_called_once()

    @patch("episodic.commands.mcp_cmd.mcp_token")
    def test_token_action(self, mock_token):
        mcp_command("token", "list")
        mock_token.assert_called_once_with(["list"])

    def test_unknown_action(self, capsys):
        mcp_command("bogus")
        # Should print an error — just check it doesn't crash


class TestMcpTokenRouting:
    """Tests for mcp_token subcommand routing."""

    @patch("episodic.commands.mcp_cmd.mcp_token_list")
    def test_no_args_shows_list(self, mock_list):
        mcp_token([])
        mock_list.assert_called_once()

    @patch("episodic.commands.mcp_cmd.mcp_token_list")
    def test_list_subcommand(self, mock_list):
        mcp_token(["list"])
        mock_list.assert_called_once()

    @patch("episodic.commands.mcp_cmd.mcp_token_create")
    def test_create_subcommand(self, mock_create):
        mcp_token(["create", "my-client"])
        mock_create.assert_called_once_with(["my-client"])

    @patch("episodic.commands.mcp_cmd.mcp_token_revoke")
    def test_revoke_subcommand(self, mock_revoke):
        mcp_token(["revoke", "some-id"])
        mock_revoke.assert_called_once_with(["some-id"])

    @patch("episodic.commands.mcp_cmd.mcp_token_rotate")
    def test_rotate_subcommand(self, mock_rotate):
        mcp_token(["rotate", "some-id", "--grace", "60"])
        mock_rotate.assert_called_once_with(["some-id", "--grace", "60"])

    def test_unknown_token_action(self):
        mcp_token(["bogus"])
        # Should print error, not crash


class TestMcpTokenCreateIntegration:
    """Integration tests for token create/list/revoke via CLI functions."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Provide a temp DB and mock _get_db_connection."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        from episodic.mcp.auth import _ensure_tables
        _ensure_tables(conn)

        def get_conn():
            return sqlite3.connect(db_path)

        with patch("episodic.commands.mcp_cmd._get_db_connection", side_effect=get_conn):
            yield db_path

    def test_create_and_list(self, mock_db, capsys):
        from episodic.commands.mcp_cmd import mcp_token_create, mcp_token_list
        mcp_token_create(["test-client"])
        mcp_token_list()
        # Just verify no crash — output goes through typer.secho

    def test_create_with_scopes(self, mock_db, capsys):
        from episodic.commands.mcp_cmd import mcp_token_create
        mcp_token_create(["test-client", "--scopes", "read,write"])

    def test_create_no_client_id(self, mock_db, capsys):
        from episodic.commands.mcp_cmd import mcp_token_create
        mcp_token_create([])
        # Should show usage, not crash

    def test_revoke_nonexistent(self, mock_db, capsys):
        from episodic.commands.mcp_cmd import mcp_token_revoke
        mcp_token_revoke(["nonexistent"])

    def test_rotate_nonexistent(self, mock_db, capsys):
        from episodic.commands.mcp_cmd import mcp_token_rotate
        mcp_token_rotate(["nonexistent"])


# ---------------------------------------------------------------------------
# Routing for new client-mode commands
# ---------------------------------------------------------------------------

class TestMcpCommandRoutingExtended:
    """Test routing for /mcp servers|connect|disconnect|tools."""

    @patch("episodic.commands.mcp_cmd.mcp_servers")
    def test_servers_action(self, mock_servers):
        mcp_command("servers")
        mock_servers.assert_called_once()

    @patch("episodic.commands.mcp_cmd.mcp_connect")
    def test_connect_action(self, mock_connect):
        mcp_command("connect", "filesystem")
        mock_connect.assert_called_once_with(["filesystem"])

    @patch("episodic.commands.mcp_cmd.mcp_disconnect")
    def test_disconnect_action(self, mock_disconnect):
        mcp_command("disconnect", "filesystem")
        mock_disconnect.assert_called_once_with(["filesystem"])

    @patch("episodic.commands.mcp_cmd.mcp_tools")
    def test_tools_action(self, mock_tools):
        mcp_command("tools")
        mock_tools.assert_called_once_with([])

    @patch("episodic.commands.mcp_cmd.mcp_tools")
    def test_tools_action_with_filter(self, mock_tools):
        mcp_command("tools", "filesystem")
        mock_tools.assert_called_once_with(["filesystem"])


# ---------------------------------------------------------------------------
# /mcp servers
# ---------------------------------------------------------------------------

class TestMcpServers:
    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_no_servers_configured(self, mock_mgr):
        mgr = MagicMock()
        mgr.server_ids = []
        mock_mgr.return_value = mgr
        mcp_servers()  # Should print "No external MCP servers configured."

    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_shows_configured_servers(self, mock_mgr):
        import asyncio

        mgr = MagicMock()
        mgr.server_ids = ["filesystem", "calendar"]

        async def fake_health():
            return {
                "filesystem": {
                    "connected": True,
                    "tool_count": 3,
                    "command": "npx",
                    "lifecycle": "on-demand",
                    "uptime_seconds": 120,
                },
                "calendar": {
                    "connected": False,
                    "command": "python",
                    "lifecycle": "on-demand",
                },
            }

        mgr.health_check_all = fake_health
        mock_mgr.return_value = mgr

        mcp_servers()  # Should not crash


# ---------------------------------------------------------------------------
# /mcp connect
# ---------------------------------------------------------------------------

class TestMcpConnect:
    def test_no_args_shows_usage(self):
        mcp_connect([])  # Should print usage, not crash

    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_unknown_server(self, mock_mgr):
        mgr = MagicMock()
        mgr.server_ids = ["filesystem"]
        mock_mgr.return_value = mgr

        mcp_connect(["unknown"])  # Should print error

    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_successful_connect(self, mock_mgr):
        import asyncio

        mgr = MagicMock()
        mgr.server_ids = ["filesystem"]

        async def fake_connect(sid):
            return True

        mgr.connect = fake_connect
        mock_client = MagicMock()
        mock_client.tools = {"a": {}, "b": {}}
        mgr.get_client.return_value = mock_client
        mock_mgr.return_value = mgr

        mcp_connect(["filesystem"])  # Should succeed


# ---------------------------------------------------------------------------
# /mcp disconnect
# ---------------------------------------------------------------------------

class TestMcpDisconnect:
    def test_no_args_shows_usage(self):
        mcp_disconnect([])  # Should print usage

    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_disconnect(self, mock_mgr):
        import asyncio

        mgr = MagicMock()

        async def fake_disconnect(sid):
            pass

        mgr.disconnect = fake_disconnect
        mock_mgr.return_value = mgr

        mcp_disconnect(["filesystem"])  # Should succeed


# ---------------------------------------------------------------------------
# /mcp tools
# ---------------------------------------------------------------------------

class TestMcpTools:
    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_no_tools_no_connections(self, mock_mgr):
        mgr = MagicMock()
        mgr.get_all_tools.return_value = []
        mgr.connected_servers = []
        mock_mgr.return_value = mgr

        mcp_tools([])  # Should show "no servers connected"

    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_shows_tools(self, mock_mgr):
        mgr = MagicMock()
        mgr.get_all_tools.return_value = [
            {
                "namespaced_name": "filesystem.read_file",
                "server_id": "filesystem",
                "name": "read_file",
                "description": "Read contents of a file",
            },
        ]
        mock_mgr.return_value = mgr

        mcp_tools([])  # Should list the tool

    @patch("episodic.commands.mcp_cmd._get_client_manager")
    def test_filter_by_server(self, mock_mgr):
        mgr = MagicMock()
        mgr.get_all_tools.return_value = [
            {"namespaced_name": "fs.read", "server_id": "fs", "description": ""},
            {"namespaced_name": "cal.events", "server_id": "cal", "description": ""},
        ]
        mock_mgr.return_value = mgr

        mcp_tools(["cal"])  # Should filter to only cal tools
