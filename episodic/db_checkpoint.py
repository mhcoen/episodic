"""
Checkpoint operations for Episodic database.

This module handles checkpoint tracking for incremental operations like
embedding indexing. Checkpoints prevent O(n) full-table scans on enable
by tracking the last processed rowid.
"""

import logging
from typing import Optional

from .db_connection import get_connection

logger = logging.getLogger(__name__)


def get_embedding_checkpoint() -> int:
    """
    Get the embedding checkpoint (last indexed rowid).

    Returns:
        The rowid of the last indexed node, or 0 if no checkpoint exists.
    """
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT value FROM configuration
            WHERE key = 'embedding_checkpoint_rowid'
        """)
        row = cursor.fetchone()
        if row:
            try:
                return int(row[0])
            except (ValueError, TypeError):
                return 0
        return 0


def set_embedding_checkpoint(rowid: int) -> None:
    """
    Set the embedding checkpoint (last indexed rowid).

    Args:
        rowid: The rowid to set as the checkpoint
    """
    with get_connection() as conn:
        # Use REPLACE to upsert the configuration value
        conn.execute("""
            INSERT OR REPLACE INTO configuration (key, value)
            VALUES ('embedding_checkpoint_rowid', ?)
        """, (str(rowid),))
        conn.commit()
        logger.debug(f"Set embedding checkpoint to {rowid}")


def get_nodes_after_checkpoint(checkpoint: int, limit: Optional[int] = None) -> list:
    """
    Get conversation nodes after a checkpoint rowid.

    Args:
        checkpoint: The rowid to start after (exclusive)
        limit: Optional limit on number of nodes to return

    Returns:
        List of dicts with id, role, content, rowid for each node
    """
    with get_connection() as conn:
        if limit:
            cursor = conn.execute("""
                SELECT id, role, content, rowid
                FROM nodes
                WHERE role IN ('user', 'assistant')
                AND content IS NOT NULL AND content != ''
                AND rowid > ?
                ORDER BY rowid
                LIMIT ?
            """, (checkpoint, limit))
        else:
            cursor = conn.execute("""
                SELECT id, role, content, rowid
                FROM nodes
                WHERE role IN ('user', 'assistant')
                AND content IS NOT NULL AND content != ''
                AND rowid > ?
                ORDER BY rowid
            """, (checkpoint,))

        nodes = []
        for row in cursor.fetchall():
            nodes.append({
                'id': row[0],
                'role': row[1],
                'content': row[2],
                'rowid': row[3]
            })
        return nodes


def get_max_node_rowid() -> int:
    """
    Get the maximum rowid in the nodes table.

    Returns:
        The maximum rowid, or 0 if no nodes exist.
    """
    with get_connection() as conn:
        cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
        row = cursor.fetchone()
        if row and row[0] is not None:
            return row[0]
        return 0


def ensure_configuration_table() -> None:
    """
    Ensure the configuration table exists for storing checkpoints.

    This is idempotent and safe to call multiple times.
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS configuration (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()
