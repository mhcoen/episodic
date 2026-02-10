"""
MCP token authentication — generation, hashing, validation, CRUD.

Token format: epk_v1_<base64url(32 random bytes)>
Storage: only SHA-256 hash is persisted; plaintext shown once on creation.
"""

import hashlib
import json
import secrets
import sqlite3
import uuid
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


TOKEN_PREFIX = "epk_v1_"
TOKEN_BYTE_LENGTH = 32


def generate_token() -> Tuple[str, str]:
    """Generate a new capability token.

    Returns:
        (plaintext_token, token_id)
    """
    raw = secrets.token_bytes(TOKEN_BYTE_LENGTH)
    encoded = urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    token = f"{TOKEN_PREFIX}{encoded}"
    token_id = str(uuid.uuid4())
    return token, token_id


def hash_token(plaintext: str) -> str:
    """SHA-256 hash a plaintext token for storage."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Database helpers — operate on a connection passed in
# ---------------------------------------------------------------------------

def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create mcp_tokens and mcp_cost_accounting if they don't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mcp_tokens (
            token_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL UNIQUE,
            client_id TEXT NOT NULL,
            scopes TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at TEXT,
            revoked_at TEXT
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_tokens_client "
        "ON mcp_tokens(client_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_tokens_hash "
        "ON mcp_tokens(token_hash)"
    )
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mcp_cost_accounting (
            client_id TEXT NOT NULL,
            date TEXT NOT NULL,
            total_cost REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (client_id, date)
        )
    """)
    conn.commit()


def create_token(
    conn: sqlite3.Connection,
    client_id: str,
    scopes: Optional[List[str]] = None,
    expires_at: Optional[str] = None,
) -> Tuple[str, str]:
    """Create and store a new token.

    Returns:
        (plaintext_token, token_id) — plaintext is shown once, never stored.
    """
    _ensure_tables(conn)
    plaintext, token_id = generate_token()
    token_h = hash_token(plaintext)
    scopes_json = json.dumps(scopes or [])

    conn.execute(
        "INSERT INTO mcp_tokens "
        "(token_id, token_hash, client_id, scopes, expires_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (token_id, token_h, client_id, scopes_json, expires_at),
    )
    conn.commit()
    return plaintext, token_id


def validate_token(conn: sqlite3.Connection, plaintext: str) -> Optional[Dict]:
    """Validate a plaintext token.

    Returns:
        Dict with token_id, client_id, scopes if valid; None otherwise.
    """
    _ensure_tables(conn)
    token_h = hash_token(plaintext)

    row = conn.execute(
        "SELECT token_id, client_id, scopes, expires_at, revoked_at "
        "FROM mcp_tokens WHERE token_hash = ?",
        (token_h,),
    ).fetchone()

    if row is None:
        return None

    token_id, client_id, scopes_json, expires_at, revoked_at = row

    # Check revoked
    if revoked_at is not None:
        return None

    # Check expired
    if expires_at is not None:
        try:
            exp = datetime.fromisoformat(expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > exp:
                return None
        except ValueError:
            pass

    return {
        "token_id": token_id,
        "client_id": client_id,
        "scopes": json.loads(scopes_json),
    }


def revoke_token(conn: sqlite3.Connection, token_id: str) -> bool:
    """Revoke a token by setting revoked_at.

    Returns:
        True if token was found and revoked.
    """
    _ensure_tables(conn)
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE mcp_tokens SET revoked_at = ? "
        "WHERE token_id = ? AND revoked_at IS NULL",
        (now, token_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def rotate_token(
    conn: sqlite3.Connection,
    old_token_id: str,
    grace_seconds: int = 0,
) -> Optional[Tuple[str, str]]:
    """Rotate a token: create new one, schedule revocation of old.

    Args:
        old_token_id: ID of the token to rotate.
        grace_seconds: Seconds before old token is revoked (0 = immediate).

    Returns:
        (new_plaintext, new_token_id) or None if old token not found.
    """
    _ensure_tables(conn)

    # Find old token
    row = conn.execute(
        "SELECT client_id, scopes, revoked_at FROM mcp_tokens WHERE token_id = ?",
        (old_token_id,),
    ).fetchone()

    if row is None:
        return None

    client_id, scopes_json, revoked_at = row
    if revoked_at is not None:
        return None

    scopes = json.loads(scopes_json)

    # Create new token with same client_id and scopes
    new_plaintext, new_token_id = create_token(conn, client_id, scopes)

    # Revoke old token (immediately or with grace period)
    if grace_seconds <= 0:
        revoke_token(conn, old_token_id)
    else:
        from datetime import timedelta
        grace_time = (
            datetime.now(timezone.utc) + timedelta(seconds=grace_seconds)
        ).isoformat()
        conn.execute(
            "UPDATE mcp_tokens SET revoked_at = ? WHERE token_id = ?",
            (grace_time, old_token_id),
        )
        conn.commit()

    return new_plaintext, new_token_id


def list_tokens(conn: sqlite3.Connection) -> List[Dict]:
    """List all active (non-revoked) tokens.

    Returns:
        List of dicts with token_id, client_id, scopes, created_at.
    """
    _ensure_tables(conn)

    rows = conn.execute(
        "SELECT token_id, client_id, scopes, created_at, expires_at "
        "FROM mcp_tokens "
        "WHERE revoked_at IS NULL "
        "ORDER BY created_at DESC",
    ).fetchall()

    result = []
    for token_id, client_id, scopes_json, created_at, expires_at in rows:
        # Check if expired
        if expires_at is not None:
            try:
                exp = datetime.fromisoformat(expires_at)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > exp:
                    continue
            except ValueError:
                pass

        result.append({
            "token_id": token_id,
            "client_id": client_id,
            "scopes": json.loads(scopes_json),
            "created_at": created_at,
            "expires_at": expires_at,
        })

    return result


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

def record_cost(
    conn: sqlite3.Connection,
    client_id: str,
    cost: float,
) -> float:
    """Record a cost for a client. Returns new daily total."""
    _ensure_tables(conn)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO mcp_cost_accounting (client_id, date, total_cost) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(client_id, date) DO UPDATE SET total_cost = total_cost + ?",
        (client_id, today, cost, cost),
    )
    conn.commit()

    row = conn.execute(
        "SELECT total_cost FROM mcp_cost_accounting "
        "WHERE client_id = ? AND date = ?",
        (client_id, today),
    ).fetchone()
    return row[0] if row else 0.0


def get_daily_cost(conn: sqlite3.Connection, client_id: str) -> float:
    """Get today's total cost for a client."""
    _ensure_tables(conn)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT total_cost FROM mcp_cost_accounting "
        "WHERE client_id = ? AND date = ?",
        (client_id, today),
    ).fetchone()
    return row[0] if row else 0.0
