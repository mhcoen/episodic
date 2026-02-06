"""
MCP server process lifecycle management.

This module manages starting, stopping, and checking the MCP server process.
It never imports the `mcp` package itself — it spawns the server as a subprocess.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


def _get_data_dir() -> Path:
    """Get the Episodic data directory."""
    return Path(os.environ.get("EPISODIC_DATA_DIR", Path.home() / ".episodic"))


def get_pidfile_path() -> Path:
    """Get the path to the MCP server pidfile."""
    return _get_data_dir() / "mcp-server.pid"


def read_pidfile() -> Optional[Dict]:
    """Read the pidfile and return its contents.

    Returns:
        Dict with 'pid', 'port', 'started_at' or None if no pidfile.
    """
    pidfile = get_pidfile_path()
    if not pidfile.exists():
        return None
    try:
        data = json.loads(pidfile.read_text())
        # Validate required keys
        if "pid" not in data or "port" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def health_check(host: str, port: int, timeout: float = 2.0) -> Optional[Dict]:
    """Perform an HTTP health check against the server.

    Returns:
        Health response dict, or None if unreachable.
    """
    try:
        import httpx
        url = f"http://{host}:{port}/health"
        resp = httpx.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_server_status() -> Tuple[str, Optional[Dict]]:
    """Get the current server status.

    Returns:
        Tuple of (status_string, health_data_or_pidfile_data).
        status_string is one of: "running", "stale", "stopped".
    """
    pidfile_data = read_pidfile()
    if pidfile_data is None:
        return "stopped", None

    pid = pidfile_data["pid"]
    port = pidfile_data["port"]

    if not is_process_alive(pid):
        # Process is dead but pidfile exists — stale
        return "stale", pidfile_data

    # Process alive — try health check
    from episodic.mcp import DEFAULT_MCP_HOST
    host = DEFAULT_MCP_HOST
    health = health_check(host, port)
    if health:
        return "running", health

    # Process alive but health check fails — still report running
    # (could be starting up)
    return "running", pidfile_data


def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def start_server(
    port: int,
    host: str,
    foreground: bool = False,
) -> Tuple[bool, str]:
    """Start the MCP server.

    Args:
        port: Port to listen on.
        host: Host to bind to.
        foreground: If True, run in foreground (blocks). Otherwise, spawn subprocess.

    Returns:
        Tuple of (success, message).
    """
    # Check if already running
    status, data = get_server_status()
    if status == "running":
        existing_port = data.get("port", "?")
        existing_pid = data.get("pid", "?")
        return False, f"MCP server already running (PID {existing_pid}, port {existing_port})"

    # Clean up stale pidfile
    if status == "stale":
        _clean_stale_pidfile()

    # Check port availability
    if not check_port_available(host, port):
        return False, f"Port {port} is already in use"

    if foreground:
        # Run in foreground — blocks
        from episodic.mcp.server import main as server_main
        server_main()
        return True, "Server stopped"

    # Spawn as subprocess
    env = os.environ.copy()
    data_dir = _get_data_dir()
    env["EPISODIC_DATA_DIR"] = str(data_dir)
    db_path = data_dir / "episodic.db"
    if db_path.exists():
        env["EPISODIC_DB_PATH"] = str(db_path)

    cmd = [
        sys.executable, "-m", "episodic.mcp",
        "--port", str(port),
        "--host", host,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return False, f"Failed to start server: {e}"

    # Wait for health check (up to 5s)
    for _ in range(25):
        time.sleep(0.2)
        # Check if process died immediately
        if proc.poll() is not None:
            return False, f"Server process exited immediately (code {proc.returncode})"
        health = health_check(host, port, timeout=1.0)
        if health:
            return True, f"MCP server started (PID {proc.pid}, port {port})"

    # Process is alive but health check not responding yet
    # Still report success since the process is running
    if proc.poll() is None:
        return True, f"MCP server started (PID {proc.pid}, port {port}) — health check pending"

    return False, "Server process started but became unresponsive"


def stop_server() -> Tuple[bool, str]:
    """Stop the MCP server.

    Returns:
        Tuple of (success, message).
    """
    pidfile_data = read_pidfile()
    if pidfile_data is None:
        return False, "MCP server is not running (no pidfile)"

    pid = pidfile_data["pid"]

    if not is_process_alive(pid):
        _clean_stale_pidfile()
        return True, "Cleaned up stale pidfile (server was not running)"

    # Send SIGTERM
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        return False, f"Failed to send SIGTERM to PID {pid}: {e}"

    # Wait up to 5s for process to exit
    for _ in range(25):
        time.sleep(0.2)
        if not is_process_alive(pid):
            _clean_stale_pidfile()
            return True, f"MCP server stopped (PID {pid})"

    # Force kill
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        _clean_stale_pidfile()
        return True, f"MCP server force-killed (PID {pid})"
    except OSError as e:
        return False, f"Failed to kill PID {pid}: {e}"


def _clean_stale_pidfile() -> None:
    """Remove a stale pidfile."""
    pidfile = get_pidfile_path()
    try:
        pidfile.unlink(missing_ok=True)
    except OSError:
        pass
