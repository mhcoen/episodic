"""
FastMCP server definition and entry point for Episodic MCP.

Run directly: python -m episodic.mcp --port 51983
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

from episodic.mcp import __version__, DEFAULT_MCP_HOST, DEFAULT_MCP_PORT

_start_time: float = 0.0


def _get_data_dir() -> Path:
    """Get the Episodic data directory."""
    return Path(os.environ.get("EPISODIC_DATA_DIR", Path.home() / ".episodic"))


def _get_pidfile_path() -> Path:
    """Get the pidfile path."""
    return _get_data_dir() / "mcp-server.pid"


def write_pidfile(port: int) -> None:
    """Write PID and port to the pidfile."""
    pidfile = _get_pidfile_path()
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    data = {"pid": os.getpid(), "port": port, "started_at": time.time()}
    pidfile.write_text(json.dumps(data))


def remove_pidfile() -> None:
    """Remove the pidfile if it exists."""
    pidfile = _get_pidfile_path()
    try:
        pidfile.unlink(missing_ok=True)
    except OSError:
        pass


def _get_node_count() -> int:
    """Get the number of nodes in the database (best-effort)."""
    try:
        db_path = os.environ.get("EPISODIC_DB_PATH")
        if not db_path:
            db_path = str(_get_data_dir() / "episodic.db")
        if not Path(db_path).exists():
            return 0
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM nodes")
            return cursor.fetchone()[0]
        except Exception:
            return 0
        finally:
            conn.close()
    except Exception:
        return 0


def _build_health_response() -> Dict[str, Any]:
    """Build the health check response."""
    uptime = time.time() - _start_time if _start_time else 0
    return {
        "status": "ok",
        "version": __version__,
        "uptime_seconds": round(uptime, 1),
        "pid": os.getpid(),
        "node_count": _get_node_count(),
    }


def create_server(name: str = "episodic"):
    """Create and configure the FastMCP server instance."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(name)

    # Register read-only tools
    from episodic.mcp.tools import register_tools
    register_tools(server)

    # Register health endpoint on the underlying Starlette app
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health_endpoint(request):
        return JSONResponse(_build_health_response())

    # Get the ASGI app and prepend health route
    app = server.sse_app()
    from episodic.mcp.middleware import (
        DEFAULT_DAILY_COST_LIMIT,
        create_auth_middleware,
    )
    db_path = os.environ.get("EPISODIC_DB_PATH", str(_get_data_dir() / "episodic.db"))
    daily_limit = float(
        os.environ.get("EPISODIC_MCP_DAILY_COST_LIMIT", DEFAULT_DAILY_COST_LIMIT)
    )
    app.add_middleware(create_auth_middleware(db_path, daily_limit))
    app.routes.insert(0, Route("/health", health_endpoint, methods=["GET"]))

    return server, app


def main():
    """Entry point for the MCP server process."""
    global _start_time

    parser = argparse.ArgumentParser(description="Episodic MCP Server")
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("EPISODIC_MCP_PORT", DEFAULT_MCP_PORT)),
        help=f"Port to listen on (default: {DEFAULT_MCP_PORT})",
    )
    parser.add_argument(
        "--host", type=str,
        default=os.environ.get("EPISODIC_MCP_HOST", DEFAULT_MCP_HOST),
        help=f"Host to bind to (default: {DEFAULT_MCP_HOST})",
    )
    args = parser.parse_args()

    _start_time = time.time()
    write_pidfile(args.port)

    def _cleanup(signum=None, frame=None):
        remove_pidfile()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    try:
        import uvicorn
        _server, app = create_server()
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        remove_pidfile()


if __name__ == "__main__":
    main()
