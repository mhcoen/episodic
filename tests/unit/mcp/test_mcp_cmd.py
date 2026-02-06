"""Tests for episodic.commands.mcp_cmd module."""

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from episodic.commands.mcp_cmd import (
    _format_uptime,
    _check_mcp_available,
    mcp_command,
    mcp_token,
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
