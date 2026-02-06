"""Tests for episodic.mcp.client — MCP client for external servers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from episodic.mcp.client import MCPClient


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
def basic_config():
    return {
        "command": "python",
        "args": ["-m", "some_server"],
        "env": {"SOME_KEY": "value"},
        "lifecycle": "on-demand",
    }


@pytest.fixture
def client(basic_config):
    return MCPClient("test-server", basic_config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_server_id(self, client):
        assert client.server_id == "test-server"

    def test_stores_command(self, client):
        assert client.command == "python"

    def test_stores_args(self, client):
        assert client.args == ["-m", "some_server"]

    def test_stores_env(self, client):
        assert client.env == {"SOME_KEY": "value"}

    def test_default_lifecycle(self):
        c = MCPClient("s", {"command": "echo"})
        assert c.lifecycle == "on-demand"

    def test_initial_health_unknown(self, client):
        assert client.health == "unknown"

    def test_not_connected_initially(self, client):
        assert client.is_connected is False

    def test_empty_tools_initially(self, client):
        assert client.tools == {}


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

class TestConnect:
    def test_already_connected_returns_true(self, client):
        """If already connected, connect() returns True immediately."""
        client._session = MagicMock()
        result = _run(client.connect())
        assert result is True

    def test_connect_sets_state_on_success(self, client):
        """Manually simulate successful connection state."""
        client._session = MagicMock()
        client._health = "healthy"
        client._connected_at = 1000.0

        assert client.health == "healthy"
        assert client.is_connected is True


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------

class TestDisconnect:
    def test_disconnect_clears_state(self, client):
        client._session = MagicMock()
        client._exit_stack = AsyncMock()
        client._tools = {"read_file": {}}
        client._health = "healthy"
        client._connected_at = 1000.0

        _run(client.disconnect())

        assert client._session is None
        assert client.is_connected is False
        assert client.tools == {}
        assert client.health == "unknown"
        assert client._connected_at is None

    def test_disconnect_when_not_connected(self, client):
        """Disconnect when not connected should not raise."""
        _run(client.disconnect())
        assert client.is_connected is False


# ---------------------------------------------------------------------------
# discover_tools
# ---------------------------------------------------------------------------

class TestDiscoverTools:
    def test_returns_empty_when_not_connected(self, client):
        result = _run(client.discover_tools())
        assert result == []

    def test_discovers_tools_from_session(self, client):
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.inputSchema = {"type": "object"}

        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        client._session = AsyncMock()
        client._session.list_tools = AsyncMock(return_value=mock_response)

        tools = _run(client.discover_tools())
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
        assert tools[0]["description"] == "Read a file"
        assert "read_file" in client.tools

    def test_handles_discovery_error(self, client):
        client._session = AsyncMock()
        client._session.list_tools = AsyncMock(side_effect=Exception("timeout"))

        tools = _run(client.discover_tools())
        assert tools == []
        assert client._last_error == "timeout"

    def test_clears_old_tools_on_rediscovery(self, client):
        """Rediscovering tools should clear old ones."""
        client._tools = {"old_tool": {"name": "old_tool"}}

        mock_tool = MagicMock()
        mock_tool.name = "new_tool"
        mock_tool.description = "New"
        mock_tool.inputSchema = {}

        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        client._session = AsyncMock()
        client._session.list_tools = AsyncMock(return_value=mock_response)

        tools = _run(client.discover_tools())
        assert "old_tool" not in client.tools
        assert "new_tool" in client.tools


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------

class TestCallTool:
    def test_error_when_not_connected(self, client):
        result = _run(client.call_tool("read_file", {}))
        assert result["error"] == "not_connected"

    def test_error_for_unknown_tool(self, client):
        client._session = AsyncMock()
        client._tools = {"write_file": {}}

        result = _run(client.call_tool("read_file", {}))
        assert result["error"] == "unknown_tool"

    def test_calls_tool_successfully(self, client):
        mock_item = MagicMock()
        mock_item.text = "file contents"
        mock_result = MagicMock()
        mock_result.content = [mock_item]
        mock_result.isError = False

        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)
        client._tools = {"read_file": {"name": "read_file"}}

        result = _run(client.call_tool("read_file", {"path": "/tmp/claude/x"}))
        assert result["content"] == ["file contents"]
        assert result["is_error"] is False

    def test_handles_call_error(self, client):
        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(side_effect=Exception("network"))
        client._tools = {"read_file": {"name": "read_file"}}

        result = _run(client.call_tool("read_file", {}))
        assert result["error"] == "call_failed"
        assert "network" in result["message"]

    def test_handles_non_text_content(self, client):
        """Content items without .text should be str()-ified."""
        class FakeItem:
            def __str__(self):
                return "binary_data"

        mock_result = MagicMock()
        mock_result.content = [FakeItem()]
        mock_result.isError = False

        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)
        client._tools = {"get_data": {"name": "get_data"}}

        result = _run(client.call_tool("get_data", {}))
        assert len(result["content"]) == 1
        assert result["content"][0] == "binary_data"


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_when_disconnected(self, client):
        status = client.get_status()
        assert status["server_id"] == "test-server"
        assert status["health"] == "unknown"
        assert status["connected"] is False
        assert status["command"] == "python"
        assert status["tool_count"] == 0

    def test_status_when_connected(self, client):
        client._session = MagicMock()
        client._health = "healthy"
        client._connected_at = 1000.0
        client._tools = {"a": {}, "b": {}}

        with patch("time.time", return_value=1060.0):
            status = client.get_status()

        assert status["connected"] is True
        assert status["health"] == "healthy"
        assert status["tool_count"] == 2
        assert status["uptime_seconds"] == 60.0

    def test_status_includes_last_error(self, client):
        client._last_error = "connection refused"
        status = client.get_status()
        assert status["last_error"] == "connection refused"

    def test_status_includes_lifecycle(self, client):
        status = client.get_status()
        assert status["lifecycle"] == "on-demand"
