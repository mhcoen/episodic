"""
MCP client — connect to and consume external MCP servers.

Manages a single MCP server connection via stdio transport.
Supports tool discovery, tool calling, health tracking, and tracing.
"""

import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for a single external MCP server.

    Connects via stdio transport, discovers tools, and calls them.
    The async lifecycle is: connect → discover_tools → call_tool → disconnect.
    """

    def __init__(self, server_id: str, config: dict):
        """Initialize client for a specific server.

        Args:
            server_id: Unique identifier for this server (e.g., "filesystem").
            config: Server config dict with keys:
                command (str): Executable to run.
                args (list[str]): Command arguments.
                env (dict, optional): Extra environment variables.
                lifecycle (str, optional): "on-demand" or "persistent".
        """
        self.server_id = server_id
        self.command = config.get("command", "")
        self.args = config.get("args", [])
        self.env = config.get("env", {})
        self.lifecycle = config.get("lifecycle", "on-demand")

        self._session = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._tools: Dict[str, dict] = {}
        self._health = "unknown"  # unknown | healthy | unhealthy
        self._connected_at: Optional[float] = None
        self._last_error: Optional[str] = None

    @property
    def health(self) -> str:
        return self._health

    @property
    def tools(self) -> Dict[str, dict]:
        return dict(self._tools)

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> bool:
        """Connect to the MCP server via stdio transport.

        Returns:
            True if connection succeeded, False otherwise.
        """
        if self._session is not None:
            return True

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            self._health = "unhealthy"
            self._last_error = "mcp package not installed"
            logger.error("MCP package not installed — cannot connect to %s", self.server_id)
            return False

        try:
            # Build environment: inherit current env + server-specific overrides
            env = dict(os.environ)
            env.update(self.env)

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=env,
            )

            self._exit_stack = AsyncExitStack()
            # Redirect subprocess stderr to devnull to suppress
            # noisy INFO logs from mcp-gsuite (credentials, discovery cache)
            devnull = open(os.devnull, "w")  # noqa: SIM115
            self._exit_stack.callback(devnull.close)
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params, errlog=devnull)
            )
            read_stream, write_stream = stdio_transport

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()

            self._health = "healthy"
            self._connected_at = time.time()
            self._last_error = None

            # Auto-discover tools on connect
            await self.discover_tools()

            logger.info(
                "Connected to MCP server %s (%d tools)",
                self.server_id, len(self._tools),
            )
            return True

        except Exception as e:
            self._health = "unhealthy"
            self._last_error = str(e)
            logger.error("Failed to connect to %s: %s", self.server_id, e)
            # Clean up partial connection
            if self._exit_stack:
                try:
                    await self._exit_stack.aclose()
                except Exception:
                    pass
            self._exit_stack = None
            self._session = None
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except (RuntimeError, GeneratorExit, BaseExceptionGroup):
                # Known noise from anyio cancel-scope cleanup when the
                # event loop is shutting down from a different context.
                pass
            except Exception as e:
                logger.warning("Error disconnecting from %s: %s", self.server_id, e)
        self._session = None
        self._exit_stack = None
        self._tools.clear()
        self._health = "unknown"
        self._connected_at = None

    async def discover_tools(self) -> List[dict]:
        """Discover available tools from the connected server.

        Returns:
            List of tool dicts with name, description, inputSchema.
        """
        if self._session is None:
            return []

        try:
            response = await self._session.list_tools()
            self._tools.clear()
            for tool in response.tools:
                self._tools[tool.name] = {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "input_schema": (
                        tool.inputSchema
                        if hasattr(tool, "inputSchema")
                        else {}
                    ),
                }
            return list(self._tools.values())

        except Exception as e:
            logger.warning("Tool discovery failed for %s: %s", self.server_id, e)
            self._last_error = str(e)
            return []

    async def call_tool(self, tool_name: str, params: dict) -> dict:
        """Call a tool on the connected server.

        Args:
            tool_name: Tool name (without server prefix).
            params: Tool parameters dict.

        Returns:
            Dict with 'content' (list of text items) and 'is_error' flag.
        """
        if self._session is None:
            return {
                "error": "not_connected",
                "message": f"Not connected to {self.server_id}",
            }

        if tool_name not in self._tools:
            return {
                "error": "unknown_tool",
                "message": f"Tool '{tool_name}' not found on {self.server_id}",
            }

        try:
            result = await self._session.call_tool(tool_name, params)
            content = []
            for item in result.content:
                if hasattr(item, "text"):
                    content.append(item.text)
                else:
                    content.append(str(item))

            return {
                "content": content,
                "is_error": getattr(result, "isError", False),
            }

        except Exception as e:
            self._last_error = str(e)
            logger.warning(
                "Tool call %s.%s failed: %s", self.server_id, tool_name, e,
            )
            return {"error": "call_failed", "message": str(e)}

    def get_status(self) -> dict:
        """Return status information about this client."""
        status = {
            "server_id": self.server_id,
            "health": self._health,
            "connected": self.is_connected,
            "command": self.command,
            "tool_count": len(self._tools),
            "lifecycle": self.lifecycle,
        }
        if self._connected_at:
            status["uptime_seconds"] = round(time.time() - self._connected_at, 1)
        if self._last_error:
            status["last_error"] = self._last_error
        return status
