"""
Authentication middleware for the MCP server.

Extracts Bearer tokens from the Authorization header, validates them,
and attaches client_id + scopes to the request state.

Requires starlette (installed with the mcp package).
"""

import sqlite3
from typing import Optional

from episodic.mcp.auth import validate_token, get_daily_cost
from episodic.mcp.request_context import set_request_context, reset_request_context

# Default rate limits
DEFAULT_DAILY_COST_LIMIT = 10.0  # $10/day per client

# Paths that skip authentication. Only the health check is public; the SSE
# stream endpoint requires a token like every other endpoint, so an
# unauthenticated client cannot open (and exhaust) event streams. The MCP SSE
# client sends the Bearer token on the /sse GET in the standard auth flow.
PUBLIC_PATHS = {"/health"}


def create_auth_middleware(db_path: str, daily_cost_limit: float = DEFAULT_DAILY_COST_LIMIT):
    """Create and return an AuthMiddleware class.

    Defers starlette import to call time so the module can be imported
    without starlette installed (tests, CLI token management).
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class AuthMiddleware(BaseHTTPMiddleware):
        """Starlette middleware for MCP token authentication."""

        def __init__(self, app):
            super().__init__(app)
            self.db_path = db_path
            self.daily_cost_limit = daily_cost_limit

        def _get_connection(self) -> sqlite3.Connection:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            return conn

        async def dispatch(self, request, call_next):
            # Skip auth for public paths
            if request.url.path in PUBLIC_PATHS:
                return await call_next(request)

            # Extract token
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    {"error": "Missing or invalid Authorization header"},
                    status_code=401,
                )

            token = auth_header[7:]  # Strip "Bearer "

            # Validate token
            try:
                conn = self._get_connection()
                try:
                    result = validate_token(conn, token)
                finally:
                    conn.close()
            except Exception:
                return JSONResponse(
                    {"error": "Token validation failed"},
                    status_code=500,
                )

            if result is None:
                return JSONResponse(
                    {"error": "Invalid, expired, or revoked token"},
                    status_code=403,
                )

            # Check daily cost limit
            try:
                conn = self._get_connection()
                try:
                    daily = get_daily_cost(conn, result["client_id"])
                finally:
                    conn.close()
            except Exception:
                daily = 0.0

            if daily >= self.daily_cost_limit:
                return JSONResponse(
                    {"error": "Daily cost limit exceeded",
                     "daily_cost": daily,
                     "limit": self.daily_cost_limit},
                    status_code=429,
                )

            # Attach auth context to request state
            request.state.client_id = result["client_id"]
            request.state.token_id = result["token_id"]
            request.state.scopes = result["scopes"]

            tokens = set_request_context(
                client_id=result["client_id"],
                token_id=result["token_id"],
                scopes=result["scopes"],
            )
            try:
                return await call_next(request)
            finally:
                reset_request_context(tokens)

    return AuthMiddleware


def get_client_id(request) -> Optional[str]:
    """Get client_id from authenticated request state."""
    return getattr(getattr(request, "state", None), "client_id", None)


def get_scopes(request) -> list:
    """Get scopes from authenticated request state."""
    return getattr(getattr(request, "state", None), "scopes", [])


def has_scope(request, scope: str) -> bool:
    """Check if the request has a specific scope."""
    scopes = get_scopes(request)
    if not scopes:
        return True  # Empty scopes = all permissions
    return scope in scopes
