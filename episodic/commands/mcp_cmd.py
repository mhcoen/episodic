"""
CLI command handler for /mcp (start, stop, status, token).
"""

import typer
from typing import List

from episodic.configuration import (
    get_text_color, get_heading_color, get_success_color,
    get_error_color, get_warning_color,
)


def _check_mcp_available() -> bool:
    """Check if the mcp package is installed."""
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        typer.secho("MCP package is not installed.", fg=get_error_color())
        typer.secho('Install with: pip install "mcp>=1.26.0,<2.0.0"', fg=get_text_color())
        return False


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if hours < 24:
        parts = [f"{hours}h"]
        if mins:
            parts.append(f"{mins}m")
        return " ".join(parts)
    days = hours // 24
    hrs = hours % 24
    parts = [f"{days}d"]
    if hrs:
        parts.append(f"{hrs}h")
    return " ".join(parts)


def mcp_command(action: str = None, *args: str) -> None:
    """Handle /mcp command routing."""
    if action is None:
        mcp_status()
    elif action == "status":
        mcp_status()
    elif action == "start":
        mcp_start(list(args))
    elif action == "stop":
        mcp_stop()
    elif action == "token":
        mcp_token(list(args))
    elif action == "traces":
        mcp_traces(list(args))
    elif action == "servers":
        mcp_servers()
    elif action == "connect":
        mcp_connect(list(args))
    elif action == "disconnect":
        mcp_disconnect(list(args))
    elif action == "tools":
        mcp_tools(list(args))
    elif action == "plugins":
        mcp_plugins()
    elif action == "plugin":
        mcp_plugin_status(list(args))
    else:
        typer.secho(f"Unknown MCP action: {action}", fg=get_error_color())
        typer.secho(
            "Usage: /mcp [start|stop|status|token|traces|servers|connect|disconnect|tools|plugins|plugin]",
            fg=get_text_color(),
        )


def mcp_status() -> None:
    """Show MCP server status."""
    from episodic.mcp.lifecycle import get_server_status

    status, data = get_server_status()

    if status == "stopped":
        typer.secho("MCP Server: STOPPED", fg=get_warning_color())
        typer.secho("Start with: /mcp start", fg=get_text_color(), dim=True)
    elif status == "stale":
        typer.secho("MCP Server: STALE (cleaning up)", fg=get_warning_color())
        from episodic.mcp.lifecycle import _clean_stale_pidfile
        _clean_stale_pidfile()
        typer.secho("Cleaned up stale pidfile. Server is stopped.", fg=get_text_color())
    elif status == "running":
        typer.secho("MCP Server: RUNNING", fg=get_success_color(), bold=True)
        if data:
            pid = data.get("pid", "?")
            port = data.get("port", "?")
            typer.secho(f"  PID:   {pid}", fg=get_text_color())
            typer.secho(f"  Port:  {port}", fg=get_text_color())
            uptime = data.get("uptime_seconds")
            if uptime is not None:
                typer.secho(f"  Up:    {_format_uptime(uptime)}", fg=get_text_color())
            node_count = data.get("node_count")
            if node_count is not None:
                typer.secho(f"  Nodes: {node_count}", fg=get_text_color())


def mcp_start(args: List[str]) -> None:
    """Start the MCP server."""
    if not _check_mcp_available():
        return

    from episodic.mcp import DEFAULT_MCP_PORT, DEFAULT_MCP_HOST
    from episodic.config import config

    port = config.get("mcp_port", DEFAULT_MCP_PORT)
    host = config.get("mcp_host", DEFAULT_MCP_HOST)
    foreground = False

    # Parse args
    i = 0
    while i < len(args):
        if args[i] == "--foreground":
            foreground = True
        elif args[i] == "--port" and i + 1 < len(args):
            i += 1
            try:
                port = int(args[i])
            except ValueError:
                typer.secho(f"Invalid port: {args[i]}", fg=get_error_color())
                return
        elif args[i] == "--host" and i + 1 < len(args):
            i += 1
            host = args[i]
        i += 1

    from episodic.mcp.lifecycle import start_server

    typer.secho(f"Starting MCP server on {host}:{port}...", fg=get_text_color())
    success, message = start_server(port, host, foreground)

    if success:
        typer.secho(message, fg=get_success_color())
    else:
        typer.secho(message, fg=get_error_color())


def mcp_stop() -> None:
    """Stop the MCP server."""
    from episodic.mcp.lifecycle import stop_server

    success, message = stop_server()

    if success:
        typer.secho(message, fg=get_success_color())
    else:
        typer.secho(message, fg=get_error_color())


from episodic.commands.mcp_token_cmds import (  # noqa: F401  (re-exported)
    _get_db_connection, mcp_token, mcp_token_create,
    mcp_token_revoke, mcp_token_rotate, mcp_token_list,
)
# ---------------------------------------------------------------------------
# Trace viewing
# ---------------------------------------------------------------------------

def mcp_traces(args: List[str]) -> None:
    """Show recent MCP traces."""
    from episodic.mcp.trace import get_traces

    limit = 20
    tool_filter = None

    # Parse args
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            i += 1
            try:
                limit = int(args[i])
            except ValueError:
                typer.secho(f"Invalid limit: {args[i]}", fg=get_error_color())
                return
        elif args[i] == "--tool" and i + 1 < len(args):
            i += 1
            tool_filter = args[i]
        i += 1

    conn = _get_db_connection()
    try:
        traces = get_traces(conn, limit=limit, tool_name=tool_filter)
    finally:
        conn.close()

    if not traces:
        typer.secho("No MCP traces recorded.", fg=get_text_color())
        return

    typer.secho(f"Recent MCP Traces ({len(traces)}):", fg=get_heading_color(), bold=True)
    for t in traces:
        status_color = get_success_color() if t["status"] == "ok" else get_error_color()
        dur = t.get("duration_ms", 0)
        ts = t.get("timestamp_start", "?")
        if len(ts) > 19:
            ts = ts[:19]  # Trim to datetime without tz for display
        typer.secho(
            f"  {ts}  {t['tool_name']:<25}  "
            f"{t['status']:<5}  {dur}ms",
            fg=status_color,
        )
        if t.get("error_code"):
            typer.secho(
                f"    error: {t['error_code']}: {t.get('message_safe', '')}",
                fg=get_error_color(), dim=True,
            )


# ---------------------------------------------------------------------------
# External server management (client mode)
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from sync code."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _get_client_manager():
    """Get or create the global MCPClientManager instance."""
    from episodic.mcp.client_manager import MCPClientManager
    # Use a module-level cache for the manager instance
    if not hasattr(_get_client_manager, "_instance"):
        _get_client_manager._instance = MCPClientManager()
    return _get_client_manager._instance


def mcp_servers() -> None:
    """List configured external MCP servers and their status."""
    manager = _get_client_manager()
    server_ids = manager.server_ids

    if not server_ids:
        typer.secho("No external MCP servers configured.", fg=get_text_color())
        typer.secho(
            'Add servers to config: /set mcp_servers {"name": {"command": "...", "args": [...]}}',
            fg=get_text_color(), dim=True,
        )
        return

    statuses = _run_async(manager.health_check_all())

    typer.secho(
        f"External MCP Servers ({len(server_ids)}):",
        fg=get_heading_color(), bold=True,
    )
    for sid, info in statuses.items():
        if info["connected"]:
            health_color = get_success_color()
            status_text = f"CONNECTED ({info['tool_count']} tools)"
            if info.get("uptime_seconds"):
                status_text += f" up {_format_uptime(info['uptime_seconds'])}"
        else:
            health_color = get_text_color()
            status_text = "disconnected"

        typer.secho(
            f"  {sid:<20} {status_text}",
            fg=health_color,
        )
        typer.secho(
            f"    cmd: {info['command']}  lifecycle: {info['lifecycle']}",
            fg=get_text_color(), dim=True,
        )
        if info.get("last_error"):
            typer.secho(
                f"    error: {info['last_error']}",
                fg=get_error_color(), dim=True,
            )


def mcp_connect(args: List[str]) -> None:
    """Connect to an external MCP server."""
    if not args:
        typer.secho(
            "Usage: /mcp connect <server_id>",
            fg=get_error_color(),
        )
        return

    server_id = args[0]
    manager = _get_client_manager()

    if server_id not in manager.server_ids:
        typer.secho(
            f"Unknown server: {server_id}", fg=get_error_color(),
        )
        typer.secho(
            f"Available: {', '.join(manager.server_ids) or '(none)'}",
            fg=get_text_color(), dim=True,
        )
        return

    typer.secho(f"Connecting to {server_id}...", fg=get_text_color())
    success = _run_async(manager.connect(server_id))

    if success:
        client = manager.get_client(server_id)
        tool_count = len(client.tools) if client else 0
        typer.secho(
            f"Connected to {server_id} ({tool_count} tools discovered).",
            fg=get_success_color(),
        )
    else:
        client = manager.get_client(server_id)
        err = client._last_error if client else "unknown error"
        typer.secho(
            f"Failed to connect to {server_id}: {err}",
            fg=get_error_color(),
        )


def mcp_disconnect(args: List[str]) -> None:
    """Disconnect from an external MCP server."""
    if not args:
        typer.secho(
            "Usage: /mcp disconnect <server_id>",
            fg=get_error_color(),
        )
        return

    server_id = args[0]
    manager = _get_client_manager()
    _run_async(manager.disconnect(server_id))

    typer.secho(f"Disconnected from {server_id}.", fg=get_success_color())


def mcp_tools(args: List[str]) -> None:
    """List discovered tools from connected external servers."""
    manager = _get_client_manager()

    # Optional filter by server_id
    filter_server = args[0] if args else None

    all_tools = manager.get_all_tools()

    if filter_server:
        all_tools = [t for t in all_tools if t["server_id"] == filter_server]

    if not all_tools:
        if filter_server:
            typer.secho(
                f"No tools found for server '{filter_server}' (is it connected?).",
                fg=get_text_color(),
            )
        else:
            connected = manager.connected_servers
            if connected:
                typer.secho(
                    "No tools discovered from connected servers.",
                    fg=get_text_color(),
                )
            else:
                typer.secho(
                    "No servers connected. Use /mcp connect <server_id> first.",
                    fg=get_text_color(),
                )
        return

    typer.secho(
        f"External MCP Tools ({len(all_tools)}):",
        fg=get_heading_color(), bold=True,
    )
    for tool in all_tools:
        typer.secho(
            f"  {tool['namespaced_name']:<40}",
            fg=get_text_color(), bold=True,
        )
        desc = tool.get("description", "")
        if desc:
            # Truncate long descriptions
            if len(desc) > 80:
                desc = desc[:77] + "..."
            typer.secho(f"    {desc}", fg=get_text_color(), dim=True)


# ---------------------------------------------------------------------------
# Plugin management
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "discovered": "yellow",
    "registered": "cyan",
    "connected": "green",
    "active": "green",
    "disconnected": "yellow",
    "disabled": "red",
}


def mcp_plugins() -> None:
    """List discovered plugins and their states."""
    try:
        from episodic.mcp.plugins import get_plugin_registry
        registry = get_plugin_registry()
        if not registry.initialized:
            registry.register_all()
    except ImportError:
        typer.secho("Plugin registry not available.", fg=get_error_color())
        return

    names = registry.names()
    if not names:
        typer.secho("No plugins discovered.", fg=get_text_color())
        return

    states = registry.states()
    typer.secho(
        f"MCP Plugins ({len(names)}):",
        fg=get_heading_color(), bold=True,
    )
    for name in sorted(names):
        state = states[name]
        color = _STATE_COLORS.get(state.value, get_text_color())
        reg = registry.get(name)
        cmds = ""
        if reg and reg.slash_commands:
            cmds = "  " + " ".join(sc.name for sc in reg.slash_commands)
        typer.secho(
            f"  {name:<15} {state.value:<14}{cmds}",
            fg=color,
        )

    typer.echo()
    typer.secho(
        "Use /mcp plugin <name> for details, /mcp connect <server> to connect.",
        fg=get_text_color(), dim=True,
    )


def mcp_plugin_status(args: List[str]) -> None:
    """Show detailed status for a specific plugin."""
    if not args:
        typer.secho("Usage: /mcp plugin <name>", fg=get_error_color())
        return

    name = args[0]

    try:
        from episodic.mcp.client_manager import PluginConnectionManager
        pcm = PluginConnectionManager(_get_client_manager())
        status = pcm.get_plugin_status(name)
    except ImportError:
        typer.secho("Plugin system not available.", fg=get_error_color())
        return

    if status is None:
        typer.secho(f"Plugin '{name}' not found.", fg=get_error_color())
        return

    color = _STATE_COLORS.get(status["state"], get_text_color())
    typer.secho(
        f"Plugin: {status['name']}",
        fg=get_heading_color(), bold=True,
    )
    typer.secho(f"  State:          {status['state']}", fg=color)
    typer.secho(f"  Server:         {status['server_id']}", fg=get_text_color())
    typer.secho(f"  Command:        {status['command']}", fg=get_text_color())
    typer.secho(f"  Policy:         {status['connect_policy']}", fg=get_text_color())
    typer.secho(
        f"  Slash commands: {', '.join(status['slash_commands']) or '(none)'}",
        fg=get_text_color(),
    )
    typer.secho(f"  Intents:        {status['intent_count']}", fg=get_text_color())
    typer.secho(f"  Tokens:         {status['token_count']}", fg=get_text_color())
    typer.secho(f"  Grammar rules:  {status['grammar_rule_count']}", fg=get_text_color())

    if status["connected"]:
        typer.secho(
            f"  Connected:      yes ({status['tool_count']} tools)",
            fg=get_success_color(),
        )
    else:
        typer.secho("  Connected:      no", fg=get_text_color(), dim=True)
