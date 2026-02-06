"""Tests for episodic.mcp.client_manager — multi-server orchestration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from episodic.mcp.client_manager import MCPClientManager


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server_configs():
    return {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp/claude"],
            "env": {},
            "lifecycle": "on-demand",
        },
        "calendar": {
            "command": "python",
            "args": ["-m", "mcp_server_calendar"],
            "env": {"CALENDAR_TOKEN": "secret"},
            "lifecycle": "on-demand",
        },
    }


@pytest.fixture
def manager(server_configs):
    return MCPClientManager(config=server_configs)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_server_configs(self, manager):
        assert "filesystem" in manager.server_ids
        assert "calendar" in manager.server_ids

    def test_empty_config(self):
        m = MCPClientManager(config={})
        assert m.server_ids == []

    def test_none_config_loads_from_episodic(self):
        with patch("episodic.config.config") as mock_config:
            mock_config.get = MagicMock(return_value={"test": {"command": "echo"}})
            m = MCPClientManager(config=None)
            assert "test" in m.server_ids

    def test_no_connected_servers_initially(self, manager):
        assert manager.connected_servers == []


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------

class TestConnectDisconnect:
    def test_connect_creates_client(self, manager):
        with patch("episodic.mcp.client.MCPClient.connect", new_callable=AsyncMock, return_value=True):
            result = _run(manager.connect("filesystem"))
            assert result is True
            assert manager.get_client("filesystem") is not None

    def test_connect_unknown_server_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown server"):
            _run(manager.connect("nonexistent"))

    def test_disconnect_works(self, manager):
        mock_client = AsyncMock()
        mock_client.is_connected = False
        manager._clients["filesystem"] = mock_client

        _run(manager.disconnect("filesystem"))
        mock_client.disconnect.assert_called_once()

    def test_disconnect_all(self, manager):
        mock_fs = AsyncMock()
        mock_cal = AsyncMock()
        manager._clients = {"filesystem": mock_fs, "calendar": mock_cal}

        _run(manager.disconnect_all())
        mock_fs.disconnect.assert_called_once()
        mock_cal.disconnect.assert_called_once()

    def test_disconnect_nonexistent_is_noop(self, manager):
        # Should not raise
        _run(manager.disconnect("nonexistent"))


# ---------------------------------------------------------------------------
# get_all_tools
# ---------------------------------------------------------------------------

class TestGetAllTools:
    def test_empty_when_no_connections(self, manager):
        assert manager.get_all_tools() == []

    def test_returns_namespaced_tools(self, manager):
        mock_client = MagicMock()
        mock_client.is_connected = True
        mock_client.tools = {
            "read_file": {"name": "read_file", "description": "Read"},
            "write_file": {"name": "write_file", "description": "Write"},
        }
        manager._clients["filesystem"] = mock_client

        tools = manager.get_all_tools()
        assert len(tools) == 2
        names = {t["namespaced_name"] for t in tools}
        assert "filesystem.read_file" in names
        assert "filesystem.write_file" in names
        assert all(t["server_id"] == "filesystem" for t in tools)

    def test_skips_disconnected_clients(self, manager):
        mock_client = MagicMock()
        mock_client.is_connected = False
        mock_client.tools = {"read": {"name": "read"}}
        manager._clients["filesystem"] = mock_client

        assert manager.get_all_tools() == []

    def test_combines_multiple_servers(self, manager):
        mock_fs = MagicMock()
        mock_fs.is_connected = True
        mock_fs.tools = {"read": {"name": "read"}}

        mock_cal = MagicMock()
        mock_cal.is_connected = True
        mock_cal.tools = {"events": {"name": "events"}}

        manager._clients = {"filesystem": mock_fs, "calendar": mock_cal}

        tools = manager.get_all_tools()
        assert len(tools) == 2
        names = {t["namespaced_name"] for t in tools}
        assert "filesystem.read" in names
        assert "calendar.events" in names


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------

class TestCallTool:
    def test_invalid_format(self, manager):
        result = _run(manager.call_tool("no_dot", {}))
        assert result["error"] == "invalid_tool"

    def test_unknown_server(self, manager):
        result = _run(manager.call_tool("unknown.tool", {}))
        assert result["error"] == "unknown_server"

    def test_auto_connect_on_demand(self, manager):
        mock_client = AsyncMock()
        mock_client.is_connected = False
        mock_client.lifecycle = "on-demand"
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.call_tool = AsyncMock(return_value={"content": ["ok"]})
        manager._clients["filesystem"] = mock_client

        with patch.object(manager, "_record_trace"):
            result = _run(manager.call_tool("filesystem.read", {}))

        mock_client.connect.assert_called_once()
        assert result["content"] == ["ok"]

    def test_not_connected_non_on_demand(self, manager):
        mock_client = AsyncMock()
        mock_client.is_connected = False
        mock_client.lifecycle = "persistent"
        manager._clients["filesystem"] = mock_client

        result = _run(manager.call_tool("filesystem.read", {}))
        assert result["error"] == "not_connected"

    def test_connection_failure(self, manager):
        mock_client = AsyncMock()
        mock_client.is_connected = False
        mock_client.lifecycle = "on-demand"
        mock_client.connect = AsyncMock(return_value=False)
        manager._clients["filesystem"] = mock_client

        result = _run(manager.call_tool("filesystem.read", {}))
        assert result["error"] == "connection_failed"

    def test_successful_call_with_trace(self, manager):
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.call_tool = AsyncMock(return_value={"content": ["data"]})
        manager._clients["filesystem"] = mock_client

        with patch.object(manager, "_record_trace") as mock_trace:
            result = _run(manager.call_tool("filesystem.read", {"path": "/x"}))

        assert result["content"] == ["data"]
        mock_trace.assert_called_once()
        args = mock_trace.call_args
        assert args[0][0] == "filesystem"  # server_id
        assert args[0][1] == "read"  # tool_name


# ---------------------------------------------------------------------------
# health_check_all
# ---------------------------------------------------------------------------

class TestHealthCheckAll:
    def test_returns_all_configured(self, manager):
        statuses = _run(manager.health_check_all())
        assert "filesystem" in statuses
        assert "calendar" in statuses
        assert statuses["filesystem"]["connected"] is False
        assert statuses["filesystem"]["health"] == "unknown"

    def test_includes_connected_status(self, manager):
        mock_client = MagicMock()
        mock_client.get_status.return_value = {
            "server_id": "filesystem",
            "health": "healthy",
            "connected": True,
            "command": "npx",
            "tool_count": 3,
            "lifecycle": "on-demand",
        }
        manager._clients["filesystem"] = mock_client

        statuses = _run(manager.health_check_all())
        assert statuses["filesystem"]["health"] == "healthy"
        assert statuses["filesystem"]["tool_count"] == 3


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_server_ids(self, manager):
        ids = manager.server_ids
        assert set(ids) == {"filesystem", "calendar"}

    def test_connected_servers(self, manager):
        mock_client = MagicMock()
        mock_client.is_connected = True
        manager._clients["filesystem"] = mock_client

        assert manager.connected_servers == ["filesystem"]

    def test_connected_servers_empty(self, manager):
        assert manager.connected_servers == []


# ---------------------------------------------------------------------------
# _record_trace
# ---------------------------------------------------------------------------

class TestRecordTrace:
    def test_trace_fails_silently(self, manager):
        """Trace recording should not raise even when import fails."""
        # _record_trace catches all exceptions internally
        manager._record_trace("fs", "read", {}, {"content": ["ok"]})
        # No assert needed — just verifying no exception raised
