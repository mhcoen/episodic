"""Tests for episodic.commands.mcp_cmd module."""

from unittest.mock import patch, MagicMock

import pytest

from episodic.commands.mcp_cmd import (
    _format_uptime,
    _check_mcp_available,
    mcp_command,
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

    def test_unknown_action(self, capsys):
        mcp_command("bogus")
        # Should print an error — just check it doesn't crash
