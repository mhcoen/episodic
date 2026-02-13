"""
MCP trace recording — full tool call tracing per spec section 12.

Records every tool invocation with timing, parameters (redacted),
input/output hashes, and error details. Supports retention eviction.
"""

import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Keys that should be redacted from parameters
_REDACT_PATTERNS = {"key", "token", "secret", "password", "credential", "auth"}

# Table auto-creation SQL (mirrors m022 migration)
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mcp_traces (
    trace_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL DEFAULT '1.0',
    timestamp_start TEXT NOT NULL,
    timestamp_end TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    direction TEXT NOT NULL,
    server_id TEXT,
    tool_name TEXT NOT NULL,
    client_id TEXT,
    thread_id TEXT,
    origin TEXT NOT NULL,
    purpose TEXT NOT NULL,
    request_id TEXT NOT NULL,
    parameter_schema_version TEXT,
    parameters_redacted TEXT,
    input_hash TEXT NOT NULL,
    input_size_bytes INTEGER NOT NULL,
    model_provider TEXT,
    model_id TEXT,
    token_in INTEGER,
    token_out INTEGER,
    cache_hit INTEGER,
    retries INTEGER DEFAULT 0,
    timeout_ms INTEGER,
    status TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    output_size_bytes INTEGER NOT NULL,
    error_code TEXT,
    message_safe TEXT,
    detail_debug TEXT
)
"""


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create mcp_traces table if it doesn't exist."""
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_traces_timestamp "
        "ON mcp_traces(timestamp_start)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_traces_tool "
        "ON mcp_traces(tool_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_traces_client "
        "ON mcp_traces(client_id)"
    )


def compute_hash(data: Any) -> str:
    """Canonical JSON SHA-256 hash.

    Canonical JSON: keys sorted, no extra whitespace, UTF-8.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def redact_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive keys from parameters dict.

    Any key whose lowercase name contains 'key', 'token', 'secret',
    'password', 'credential', or 'auth' is replaced with '[REDACTED]'.
    """
    if not params:
        return {}
    redacted = {}
    for k, v in params.items():
        k_lower = k.lower()
        if any(pat in k_lower for pat in _REDACT_PATTERNS):
            redacted[k] = "[REDACTED]"
        elif isinstance(v, dict):
            redacted[k] = redact_parameters(v)
        elif isinstance(v, list):
            redacted[k] = _redact_sequence(v)
        elif isinstance(v, tuple):
            redacted[k] = _redact_sequence(v)
        else:
            redacted[k] = v
    return redacted


def _redact_sequence(values: Any) -> list[Any]:
    """Redact nested dict/list elements inside a sequence."""
    redacted_items: list[Any] = []
    for item in values:
        if isinstance(item, dict):
            redacted_items.append(redact_parameters(item))
        elif isinstance(item, list) or isinstance(item, tuple):
            redacted_items.append(_redact_sequence(item))
        else:
            redacted_items.append(item)
    return redacted_items


def record_trace(conn: sqlite3.Connection, trace_data: Dict[str, Any]) -> str:
    """Insert a trace record into the database.

    Args:
        conn: Database connection.
        trace_data: Dict with trace fields. Must include at minimum:
            trace_id, timestamp_start, timestamp_end, duration_ms,
            direction, tool_name, origin, purpose, request_id,
            input_hash, input_size_bytes, status, output_hash,
            output_size_bytes.

    Returns:
        The trace_id.
    """
    _ensure_table(conn)

    cols = [
        "trace_id", "schema_version", "timestamp_start", "timestamp_end",
        "duration_ms", "direction", "server_id", "tool_name", "client_id",
        "thread_id", "origin", "purpose", "request_id",
        "parameter_schema_version", "parameters_redacted",
        "input_hash", "input_size_bytes", "model_provider", "model_id",
        "token_in", "token_out", "cache_hit", "retries", "timeout_ms",
        "status", "output_hash", "output_size_bytes",
        "error_code", "message_safe", "detail_debug",
    ]

    values = []
    for col in cols:
        val = trace_data.get(col)
        # Serialize dicts to JSON
        if isinstance(val, (dict, list)):
            val = json.dumps(val, sort_keys=True, separators=(",", ":"),
                             default=str)
        values.append(val)

    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)

    conn.execute(
        f"INSERT INTO mcp_traces ({col_names}) VALUES ({placeholders})",
        values,
    )
    conn.commit()

    return trace_data["trace_id"]


def get_traces(
    conn: sqlite3.Connection,
    limit: int = 100,
    tool_name: Optional[str] = None,
    client_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent traces.

    Args:
        conn: Database connection.
        limit: Max traces to return.
        tool_name: Filter by tool name.
        client_id: Filter by client ID.

    Returns:
        List of trace dicts, newest first.
    """
    _ensure_table(conn)

    query = "SELECT * FROM mcp_traces"
    conditions = []
    params = []

    if tool_name:
        conditions.append("tool_name = ?")
        params.append(tool_name)
    if client_id:
        conditions.append("client_id = ?")
        params.append(client_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp_start DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    traces = []
    for row in rows:
        trace = dict(zip(columns, row))
        # Parse JSON fields
        if trace.get("parameters_redacted"):
            try:
                trace["parameters_redacted"] = json.loads(
                    trace["parameters_redacted"]
                )
            except (json.JSONDecodeError, TypeError):
                pass
        traces.append(trace)

    return traces


def evict_old_traces(
    conn: sqlite3.Connection,
    max_age_days: int = 30,
    max_size_bytes: int = 2 * 1024 * 1024 * 1024,
) -> int:
    """Evict old traces per retention policy.

    Deletes traces older than max_age_days. Then if total size exceeds
    max_size_bytes, deletes oldest traces until under the limit.

    Args:
        conn: Database connection.
        max_age_days: Maximum age of traces in days.
        max_size_bytes: Maximum total size in bytes.

    Returns:
        Number of traces deleted.
    """
    _ensure_table(conn)
    total_deleted = 0

    # Phase 1: delete by age
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=max_age_days)
    ).isoformat()

    cursor = conn.execute(
        "DELETE FROM mcp_traces WHERE timestamp_start < ?", (cutoff,)
    )
    total_deleted += cursor.rowcount
    conn.commit()

    # Phase 2: delete by size (estimate using page_count * page_size
    # for the whole DB is too coarse; use row count + avg size instead)
    row = conn.execute(
        "SELECT COUNT(*), SUM(input_size_bytes + output_size_bytes) "
        "FROM mcp_traces"
    ).fetchone()

    if row and row[1] and row[1] > max_size_bytes:
        # Delete oldest traces until under limit
        excess = row[1] - max_size_bytes
        deleted_so_far = 0

        # Get traces oldest first with their sizes
        size_cursor = conn.execute(
            "SELECT trace_id, input_size_bytes + output_size_bytes as total "
            "FROM mcp_traces ORDER BY timestamp_start ASC"
        )
        ids_to_delete = []
        for trace_id, total in size_cursor:
            if deleted_so_far >= excess:
                break
            ids_to_delete.append(trace_id)
            deleted_so_far += (total or 0)

        if ids_to_delete:
            placeholders = ", ".join("?" for _ in ids_to_delete)
            cursor = conn.execute(
                f"DELETE FROM mcp_traces WHERE trace_id IN ({placeholders})",
                ids_to_delete,
            )
            total_deleted += cursor.rowcount
            conn.commit()

    return total_deleted


@contextmanager
def trace_tool_call(
    conn: sqlite3.Connection,
    tool_name: str,
    client_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    purpose: str = "interactive",
    direction: str = "server_tool_call",
):
    """Context manager that records a full trace for a tool call.

    Usage:
        with trace_tool_call(conn, "get_topics", client_id, params) as ctx:
            ctx["output"] = do_work()

    On exit (normal or exception), a trace record is written to the DB.
    """
    trace_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    start = datetime.now(timezone.utc)
    params = parameters or {}

    ctx = {
        "trace_id": trace_id,
        "status": "ok",
        "output": None,
        "error": None,
        "model_provider": None,
        "model_id": None,
        "token_in": None,
        "token_out": None,
        "cache_hit": None,
    }

    try:
        yield ctx
    except Exception as e:
        ctx["status"] = "error"
        ctx["error"] = e
        raise
    finally:
        end = datetime.now(timezone.utc)
        duration_ms = int((end - start).total_seconds() * 1000)

        # Compute hashes
        input_json = json.dumps(params, sort_keys=True, separators=(",", ":"),
                                default=str)
        output_data = ctx.get("output") or {}
        output_json = json.dumps(output_data, sort_keys=True,
                                 separators=(",", ":"), default=str)

        error_info = ctx.get("error")
        trace_data = {
            "trace_id": trace_id,
            "schema_version": "1.0",
            "timestamp_start": start.isoformat(),
            "timestamp_end": end.isoformat(),
            "duration_ms": duration_ms,
            "direction": direction,
            "server_id": None,
            "tool_name": tool_name,
            "client_id": client_id,
            "thread_id": None,
            "origin": "mcp_server",
            "purpose": purpose,
            "request_id": request_id,
            "parameter_schema_version": "1.0",
            "parameters_redacted": redact_parameters(params),
            "input_hash": compute_hash(params),
            "input_size_bytes": len(input_json.encode("utf-8")),
            "model_provider": ctx.get("model_provider"),
            "model_id": ctx.get("model_id"),
            "token_in": ctx.get("token_in"),
            "token_out": ctx.get("token_out"),
            "cache_hit": ctx.get("cache_hit"),
            "retries": 0,
            "timeout_ms": None,
            "status": ctx["status"],
            "output_hash": compute_hash(output_data),
            "output_size_bytes": len(output_json.encode("utf-8")),
            "error_code": type(error_info).__name__ if error_info else None,
            "message_safe": str(error_info)[:500] if error_info else None,
            "detail_debug": None,
        }

        try:
            record_trace(conn, trace_data)
        except Exception as rec_err:
            logger.warning("Failed to record trace: %s", rec_err)
