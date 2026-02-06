"""
MCP thread handle management — generation, hashing, validation, CRUD.

Handle format: eth_v1_<base64url(32 random bytes)>
Storage: only SHA-256 hash is persisted; plaintext shown once on creation.

Threads map to rows in the existing `conversations` table. MCP clients
access conversations only through validated handles with permissions.
The CLI's main conversation (thread 0) needs no handle.
"""

import hashlib
import json
import secrets
import sqlite3
import uuid
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


HANDLE_PREFIX = "eth_v1_"
HANDLE_BYTE_LENGTH = 32


def generate_thread_handle() -> Tuple[str, str]:
    """Generate a new thread handle.

    Returns:
        (plaintext_handle, handle_id)
    """
    raw = secrets.token_bytes(HANDLE_BYTE_LENGTH)
    encoded = urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    handle = f"{HANDLE_PREFIX}{encoded}"
    handle_id = str(uuid.uuid4())
    return handle, handle_id


def hash_handle(plaintext: str) -> str:
    """SHA-256 hash a plaintext handle for storage."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Database helpers — operate on a connection passed in
# ---------------------------------------------------------------------------

def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create mcp_thread_handles if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mcp_thread_handles (
            handle_id TEXT PRIMARY KEY,
            handle_hash TEXT NOT NULL UNIQUE,
            thread_id INTEGER NOT NULL,
            client_id TEXT NOT NULL,
            permissions TEXT NOT NULL DEFAULT '["read","write"]',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            revoked_at TEXT,
            FOREIGN KEY (thread_id) REFERENCES conversations(id)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_handles_thread "
        "ON mcp_thread_handles(thread_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_handles_client "
        "ON mcp_thread_handles(client_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_handles_hash "
        "ON mcp_thread_handles(handle_hash)"
    )
    conn.commit()


def create_thread(
    conn: sqlite3.Connection,
    client_id: str,
    background_influences_topics: bool = False,
    permissions: Optional[List[str]] = None,
) -> Dict:
    """Create a new conversation thread and return a handle.

    Creates a row in the conversations table and a corresponding handle
    in mcp_thread_handles.

    Args:
        client_id: ID of the MCP client creating the thread.
        background_influences_topics: Whether this thread's traffic
            affects topic segmentation (stored in conversations.metadata).
        permissions: Handle permissions (default: ["read", "write"]).

    Returns:
        Dict with thread_id, thread_handle (plaintext, shown once),
        handle_id, permissions.
    """
    _ensure_tables(conn)

    # Ensure conversations table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT UNIQUE NOT NULL,
            root_node_id TEXT,
            current_head_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
    """)

    # Create conversation row
    conversation_id = str(uuid.uuid4())
    metadata = json.dumps({
        "created_by": client_id,
        "background_influences_topics": background_influences_topics,
    })

    cursor = conn.execute(
        "INSERT INTO conversations (conversation_id, metadata) "
        "VALUES (?, ?)",
        (conversation_id, metadata),
    )
    thread_id = cursor.lastrowid

    # Generate handle
    perms = permissions or ["read", "write"]
    plaintext, handle_id = generate_thread_handle()
    handle_h = hash_handle(plaintext)
    perms_json = json.dumps(perms)

    conn.execute(
        "INSERT INTO mcp_thread_handles "
        "(handle_id, handle_hash, thread_id, client_id, permissions) "
        "VALUES (?, ?, ?, ?, ?)",
        (handle_id, handle_h, thread_id, client_id, perms_json),
    )
    conn.commit()

    return {
        "thread_id": thread_id,
        "thread_handle": plaintext,
        "handle_id": handle_id,
        "permissions": perms,
    }


def validate_thread_handle(
    conn: sqlite3.Connection,
    plaintext: str,
    required_permission: Optional[str] = None,
) -> Optional[Dict]:
    """Validate a plaintext thread handle.

    Args:
        plaintext: The thread handle to validate.
        required_permission: If set, check the handle has this permission
            (e.g. "read", "write", "admin").

    Returns:
        Dict with handle_id, thread_id, client_id, permissions if valid;
        None otherwise.
    """
    _ensure_tables(conn)
    handle_h = hash_handle(plaintext)

    row = conn.execute(
        "SELECT handle_id, thread_id, client_id, permissions, revoked_at "
        "FROM mcp_thread_handles WHERE handle_hash = ?",
        (handle_h,),
    ).fetchone()

    if row is None:
        return None

    handle_id, thread_id, client_id, perms_json, revoked_at = row

    if revoked_at is not None:
        return None

    perms = json.loads(perms_json)

    if required_permission and required_permission not in perms:
        return None

    return {
        "handle_id": handle_id,
        "thread_id": thread_id,
        "client_id": client_id,
        "permissions": perms,
    }


def get_thread_handles(
    conn: sqlite3.Connection,
    client_id: Optional[str] = None,
    thread_id: Optional[int] = None,
) -> List[Dict]:
    """List active (non-revoked) thread handles.

    Args:
        client_id: Filter by client ID.
        thread_id: Filter by thread ID.

    Returns:
        List of dicts with handle_id, thread_id, client_id,
        permissions, created_at.
    """
    _ensure_tables(conn)

    query = (
        "SELECT handle_id, thread_id, client_id, permissions, created_at "
        "FROM mcp_thread_handles "
        "WHERE revoked_at IS NULL"
    )
    params: List = []

    if client_id is not None:
        query += " AND client_id = ?"
        params.append(client_id)

    if thread_id is not None:
        query += " AND thread_id = ?"
        params.append(thread_id)

    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()

    return [
        {
            "handle_id": hid,
            "thread_id": tid,
            "client_id": cid,
            "permissions": json.loads(perms_json),
            "created_at": created_at,
        }
        for hid, tid, cid, perms_json, created_at in rows
    ]


def revoke_thread_handle(conn: sqlite3.Connection, handle_id: str) -> bool:
    """Revoke a thread handle by setting revoked_at.

    Returns:
        True if handle was found and revoked.
    """
    _ensure_tables(conn)
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE mcp_thread_handles SET revoked_at = ? "
        "WHERE handle_id = ? AND revoked_at IS NULL",
        (now, handle_id),
    )
    conn.commit()
    return cursor.rowcount > 0
