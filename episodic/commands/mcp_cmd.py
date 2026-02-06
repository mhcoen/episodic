"""
CLI command handler for /mcp (start, stop, status).
"""

import typer
from typing import List

from episodic.configuration import (
    get_text_color, get_heading_color, get_success_color,
    get_error_color, get_warning_color, get_system_color,
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
    else:
        typer.secho(f"Unknown MCP action: {action}", fg=get_error_color())
        typer.secho("Usage: /mcp [start|stop|status]", fg=get_text_color())


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
