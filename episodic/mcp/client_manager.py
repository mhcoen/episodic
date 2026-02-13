"""
MCP client manager — orchestrates connections to multiple external MCP servers.

Provides namespaced tool access (server_id.tool_name), connection lifecycle
management, and health checking across all configured servers.

PluginConnectionManager bridges the plugin registry with the client manager,
resolving env vars from plugin manifests and managing plugin state transitions.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manages multiple MCPClient instances for external MCP servers.

    Tools from external servers are namespaced as ``server_id.tool_name``
    to avoid collisions.
    """

    _logging_configured = False

    def __init__(self, config: Optional[dict] = None):
        """Initialize the client manager.

        Args:
            config: Dict of server configs keyed by server_id. Each value
                has keys: command, args, env, lifecycle.  If None, reads
                from the Episodic config at init time.
        """
        if config is None:
            config = self._load_config()
        self._server_configs = config or {}
        self._clients: Dict[str, "MCPClient"] = {}
        self._suppress_library_noise()

    @classmethod
    def _suppress_library_noise(cls) -> None:
        """Suppress noisy logs from MCP and Google libraries.

        The mcp-gsuite subprocess prints INFO lines to stderr (credentials,
        discovery cache) and the MCP SDK logs ERROR tracebacks for non-JSON
        lines the server emits during startup (e.g. ``'darwin'``).  Neither
        affects functionality, so we silence them here.
        """
        if cls._logging_configured:
            return
        cls._logging_configured = True
        # Suppress JSONRPC parse-error tracebacks for non-JSON server output
        logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)
        for name in (
            "mcp",
            "mcp.server.lowlevel.server",
            "googleapiclient.discovery_cache",
            "httpcore",
            "httpx",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    @staticmethod
    def _load_config() -> dict:
        """Load mcp_servers config from Episodic config."""
        try:
            from episodic.config import config
            return config.get("mcp_servers", {}) or {}
        except ImportError:
            return {}

    def _get_or_create_client(self, server_id: str) -> "MCPClient":
        """Get existing client or create a new one."""
        if server_id not in self._clients:
            if server_id not in self._server_configs:
                raise ValueError(f"Unknown server: {server_id}")
            from episodic.mcp.client import MCPClient
            self._clients[server_id] = MCPClient(
                server_id, self._server_configs[server_id],
            )
        return self._clients[server_id]

    async def connect(self, server_id: str) -> bool:
        """Connect to a specific server.

        Args:
            server_id: The server to connect to.

        Returns:
            True if connected successfully.
        """
        client = self._get_or_create_client(server_id)
        return await client.connect()

    async def disconnect(self, server_id: str) -> None:
        """Disconnect from a specific server."""
        if server_id in self._clients:
            await self._clients[server_id].disconnect()

    async def disconnect_all(self) -> None:
        """Disconnect from all connected servers."""
        for server_id in list(self._clients.keys()):
            await self.disconnect(server_id)

    def get_client(self, server_id: str) -> Optional["MCPClient"]:
        """Get an existing client instance (or None if not created)."""
        return self._clients.get(server_id)

    def get_all_tools(self) -> List[dict]:
        """Get all tools from all connected servers, namespaced.

        Returns:
            List of tool dicts with ``namespaced_name`` field added
            (``server_id.tool_name``).
        """
        tools = []
        for server_id, client in self._clients.items():
            if not client.is_connected:
                continue
            for tool_name, tool_info in client.tools.items():
                namespaced = dict(tool_info)
                namespaced["namespaced_name"] = f"{server_id}.{tool_name}"
                namespaced["server_id"] = server_id
                tools.append(namespaced)
        return tools

    async def call_tool(self, namespaced_tool: str, params: dict) -> dict:
        """Call a namespaced tool (``server_id.tool_name``).

        Auto-connects if the server uses on-demand lifecycle.

        Args:
            namespaced_tool: Tool identifier as ``server_id.tool_name``.
            params: Tool parameters.

        Returns:
            Tool result dict.
        """
        parts = namespaced_tool.split(".", 1)
        if len(parts) != 2:
            return {
                "error": "invalid_tool",
                "message": f"Invalid tool name '{namespaced_tool}', expected 'server_id.tool_name'",
            }

        server_id, tool_name = parts

        # Ensure client exists
        try:
            client = self._get_or_create_client(server_id)
        except ValueError as e:
            return {"error": "unknown_server", "message": str(e)}

        # Auto-connect on-demand servers
        if not client.is_connected:
            if client.lifecycle == "on-demand":
                success = await client.connect()
                if not success:
                    return {
                        "error": "connection_failed",
                        "message": f"Failed to connect to {server_id}",
                    }
            else:
                return {
                    "error": "not_connected",
                    "message": f"Server {server_id} is not connected",
                }

        # Trace the call
        result = await client.call_tool(tool_name, params)

        # Record trace if DB available
        self._record_trace(server_id, tool_name, params, result)

        return result

    def _record_trace(
        self,
        server_id: str,
        tool_name: str,
        params: dict,
        result: dict,
    ) -> None:
        """Record a client tool call trace (best-effort)."""
        try:
            from episodic.mcp.trace import trace_tool_call
            from episodic.mcp.tools import _get_db_connection

            conn = _get_db_connection()
            try:
                status = "error" if result.get("error") else "ok"
                with trace_tool_call(
                    conn,
                    tool_name=f"{server_id}.{tool_name}",
                    parameters=params,
                    direction="client_tool_call",
                ) as ctx:
                    ctx["output"] = result
                    if status == "error":
                        ctx["status"] = "error"
            finally:
                conn.close()
        except Exception as e:
            logger.debug("Failed to record client trace: %s", e)

    async def health_check_all(self) -> Dict[str, dict]:
        """Check health of all configured servers.

        Returns:
            Dict mapping server_id to status dict.
        """
        statuses = {}
        for server_id in self._server_configs:
            if server_id in self._clients:
                statuses[server_id] = self._clients[server_id].get_status()
            else:
                statuses[server_id] = {
                    "server_id": server_id,
                    "health": "unknown",
                    "connected": False,
                    "command": self._server_configs[server_id].get("command", ""),
                    "tool_count": 0,
                    "lifecycle": self._server_configs[server_id].get("lifecycle", "on-demand"),
                }
        return statuses

    @property
    def server_ids(self) -> List[str]:
        """List all configured server IDs."""
        return list(self._server_configs.keys())

    @property
    def connected_servers(self) -> List[str]:
        """List server IDs that are currently connected."""
        return [
            sid for sid, client in self._clients.items()
            if client.is_connected
        ]


class PluginConnectionManager:
    """Bridges the plugin registry with the MCPClientManager.

    Reads ServerManifest from plugin registrations, resolves environment
    variables, and manages connect/disconnect with plugin state transitions.
    """

    def __init__(self, client_manager: Optional[MCPClientManager] = None) -> None:
        self._client_manager = client_manager

    @property
    def client_manager(self) -> MCPClientManager:
        if self._client_manager is None:
            self._client_manager = MCPClientManager()
        return self._client_manager

    def _resolve_env(self, manifest: Any) -> Tuple[Dict[str, str], List[str]]:
        """Resolve required env vars from config → process env.

        Returns (resolved_env, missing_vars).
        """
        from episodic.config import config

        resolved: Dict[str, str] = {}
        missing: List[str] = []

        for var in getattr(manifest, "env_vars", []):
            # Check episodic config first, then process env
            value = config.get(var) or config.get(var.lower()) or os.environ.get(var)
            if value:
                resolved[var] = str(value)
            else:
                missing.append(var)

        return resolved, missing

    def _build_server_config(self, manifest: Any, env: Dict[str, str]) -> dict:
        """Build MCPClientManager server config from a ServerManifest."""
        return {
            "command": manifest.command,
            "args": list(manifest.args),
            "env": env,
            "lifecycle": manifest.connect_policy or "manual",
        }

    async def connect_plugin(self, name: str) -> Tuple[bool, str]:
        """Connect to a plugin's MCP server.

        Resolves env from manifest, registers the server config with
        the client manager, connects, and transitions plugin state.

        Returns (success, message).
        """
        from episodic.mcp.plugins import get_plugin_registry, PluginState
        registry = get_plugin_registry()

        reg = registry.get(name)
        if reg is None:
            return False, f"Plugin '{name}' not found"

        state = registry.get_state(name)
        if state == PluginState.ACTIVE or state == PluginState.CONNECTED:
            return True, f"Plugin '{name}' is already connected"

        manifest = reg.manifest
        env, missing = self._resolve_env(manifest)
        if missing:
            return False, f"Missing env vars: {', '.join(missing)}"

        # Register server config with client manager
        server_id = manifest.server_id
        mgr = self.client_manager
        mgr._server_configs[server_id] = self._build_server_config(manifest, env)

        # Connect
        try:
            success = await mgr.connect(server_id)
        except Exception as e:
            return False, f"Connection failed: {e}"

        if success:
            # Transition plugin state
            try:
                registry.transition(name, PluginState.CONNECTED)
            except ValueError:
                pass  # Already in valid state
            try:
                registry.transition(name, PluginState.ACTIVE)
            except ValueError:
                pass

            client = mgr.get_client(server_id)
            tool_count = len(client.tools) if client else 0
            return True, f"Connected ({tool_count} tools)"
        else:
            client = mgr.get_client(server_id)
            err = getattr(client, "_last_error", "unknown error") if client else "unknown error"
            return False, f"Failed to connect: {err}"

    async def disconnect_plugin(self, name: str) -> Tuple[bool, str]:
        """Disconnect a plugin's MCP server.

        Returns (success, message).
        """
        from episodic.mcp.plugins import get_plugin_registry, PluginState
        registry = get_plugin_registry()

        reg = registry.get(name)
        if reg is None:
            return False, f"Plugin '{name}' not found"

        server_id = reg.manifest.server_id
        mgr = self.client_manager

        try:
            await mgr.disconnect(server_id)
        except Exception as e:
            return False, f"Disconnect failed: {e}"

        # Transition plugin state
        try:
            registry.transition(name, PluginState.DISCONNECTED)
        except ValueError:
            pass  # State transition not valid, that's OK

        return True, f"Disconnected from {name}"

    def get_plugin_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a plugin."""
        from episodic.mcp.plugins import get_plugin_registry
        registry = get_plugin_registry()

        reg = registry.get(name)
        if reg is None:
            return None

        state = registry.get_state(name)
        manifest = reg.manifest
        server_id = manifest.server_id

        mgr = self.client_manager
        client = mgr.get_client(server_id) if server_id in mgr._server_configs else None

        status: Dict[str, Any] = {
            "name": name,
            "state": state.value if state else "unknown",
            "server_id": server_id,
            "command": manifest.command,
            "connect_policy": manifest.connect_policy,
            "slash_commands": [sc.name for sc in reg.slash_commands],
            "intent_count": len(reg.extraction_contribution.intents) if reg.extraction_contribution else 0,
            "token_count": len(reg.tokens),
            "grammar_rule_count": len(reg.grammar_rules),
            "connected": client.is_connected if client else False,
            "tool_count": len(client.tools) if client and client.is_connected else 0,
        }

        return status
