"""
Indexing status operations for Episodic database.

This module handles durable tracking of RAG indexing status,
enabling recovery from failures and visibility into indexing state.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional

from .db_connection import get_connection


def ensure_indexing_table():
    """Ensure the indexing_status table exists."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexing_status (
                node_id TEXT NOT NULL,
                index_type TEXT NOT NULL DEFAULT 'conversation',
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'ok', 'failed')),
                indexed_at TEXT,
                failed_at TEXT,
                last_error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_retry_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (node_id, index_type),
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_status_status ON indexing_status(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_status_type ON indexing_status(index_type)")
        conn.commit()  # Explicit commit for DDL


def update_indexing_status(node_id: str, index_type: str, status: str, error: Optional[str] = None):
    """Upsert indexing status for a node.

    Args:
        node_id: The node ID being indexed
        index_type: Type of index (e.g., 'conversation')
        status: 'ok' for success, 'failed' for failure
        error: Error message if status is 'failed' (truncated to 500 chars)
    """
    ensure_indexing_table()
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()

        if status == 'ok':
            cursor.execute("""
                INSERT INTO indexing_status (node_id, index_type, status, indexed_at, attempts)
                VALUES (?, ?, 'ok', ?, 1)
                ON CONFLICT(node_id, index_type) DO UPDATE SET
                    status = 'ok',
                    indexed_at = excluded.indexed_at,
                    failed_at = NULL,
                    last_error = NULL,
                    attempts = attempts + 1
            """, (node_id, index_type, now))
        else:  # failed
            error_msg = error[:500] if error else None
            cursor.execute("""
                INSERT INTO indexing_status (node_id, index_type, status, failed_at, last_error, attempts)
                VALUES (?, ?, 'failed', ?, ?, 1)
                ON CONFLICT(node_id, index_type) DO UPDATE SET
                    status = 'failed',
                    failed_at = excluded.failed_at,
                    last_error = excluded.last_error,
                    attempts = attempts + 1
            """, (node_id, index_type, now, error_msg))

        conn.commit()  # Explicit commit to ensure status is persisted


def get_indexing_status(node_id: str, index_type: str = 'conversation') -> Optional[Dict]:
    """Get indexing status for a specific node.

    Args:
        node_id: The node ID to check
        index_type: Type of index (default: 'conversation')

    Returns:
        Dict with status info or None if never attempted
    """
    ensure_indexing_table()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT status, indexed_at, failed_at, last_error, attempts
            FROM indexing_status
            WHERE node_id = ? AND index_type = ?
        """, (node_id, index_type))
        row = cursor.fetchone()

        if not row:
            return None

        return {
            'status': row[0],
            'indexed_at': row[1],
            'failed_at': row[2],
            'last_error': row[3],
            'attempts': row[4]
        }


def should_index(node_id: str, index_type: str = 'conversation') -> bool:
    """Check if a node needs indexing.

    Returns True if:
    - Never attempted (no status record)
    - Previous attempt failed (status = 'failed')

    Returns False if:
    - Successfully indexed (status = 'ok')
    """
    status = get_indexing_status(node_id, index_type)

    if status is None:
        return True  # Never attempted

    return status['status'] == 'failed'  # Retry failures


def get_unindexed_nodes(index_type: str = 'conversation', limit: int = 100) -> List[str]:
    """Get node IDs that haven't been successfully indexed.

    Returns nodes where:
    - No indexing_status record exists, OR
    - Status is 'failed'

    Args:
        index_type: Type of index (default: 'conversation')
        limit: Maximum number of nodes to return

    Returns:
        List of node IDs (user nodes only)
    """
    ensure_indexing_table()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT n.id FROM nodes n
            LEFT JOIN indexing_status s ON n.id = s.node_id AND s.index_type = ?
            WHERE n.role = 'user'
            AND (s.status IS NULL OR s.status = 'failed')
            ORDER BY n.created_at DESC
            LIMIT ?
        """, (index_type, limit))

        return [row[0] for row in cursor.fetchall()]


def get_failed_nodes(index_type: str = 'conversation', limit: int = 100) -> List[Dict]:
    """Get nodes that failed indexing.

    Args:
        index_type: Type of index (default: 'conversation')
        limit: Maximum number of nodes to return

    Returns:
        List of dicts with node_id, last_error, attempts, failed_at
    """
    ensure_indexing_table()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT node_id, last_error, attempts, failed_at
            FROM indexing_status
            WHERE index_type = ? AND status = 'failed'
            ORDER BY failed_at DESC
            LIMIT ?
        """, (index_type, limit))

        return [
            {
                'node_id': row[0],
                'last_error': row[1],
                'attempts': row[2],
                'failed_at': row[3]
            }
            for row in cursor.fetchall()
        ]


def get_indexing_stats(index_type: str = 'conversation') -> Dict[str, int]:
    """Get indexing statistics.

    Args:
        index_type: Type of index (default: 'conversation')

    Returns:
        Dict mapping status to count (e.g., {'ok': 100, 'failed': 5})
    """
    ensure_indexing_table()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                status,
                COUNT(*) as count
            FROM indexing_status
            WHERE index_type = ?
            GROUP BY status
        """, (index_type,))

        return {row[0]: row[1] for row in cursor.fetchall()}


def clear_indexing_status(index_type: Optional[str] = None, node_id: Optional[str] = None):
    """Clear indexing status records.

    Args:
        index_type: If provided, only clear for this index type
        node_id: If provided, only clear for this node
    """
    ensure_indexing_table()

    with get_connection() as conn:
        cursor = conn.cursor()

        if node_id and index_type:
            cursor.execute(
                "DELETE FROM indexing_status WHERE node_id = ? AND index_type = ?",
                (node_id, index_type)
            )
        elif index_type:
            cursor.execute(
                "DELETE FROM indexing_status WHERE index_type = ?",
                (index_type,)
            )
        elif node_id:
            cursor.execute(
                "DELETE FROM indexing_status WHERE node_id = ?",
                (node_id,)
            )
        else:
            cursor.execute("DELETE FROM indexing_status")
