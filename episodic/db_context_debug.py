"""
Context assembly debug persistence for Episodic database.

This module manages the context_assembly_debug table which stores
debug information from each context assembly operation.
"""

import json
import logging
import sqlite3
from typing import Optional, Dict, Any, List

from .db_connection import get_connection

logger = logging.getLogger(__name__)


def persist_context_assembly_debug(
    user_node_id: str,
    debug_info: Dict[str, Any],
    reactivation_decision: Optional[Any] = None,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Persist context assembly debug info keyed by user_node_id.

    Args:
        user_node_id: The user node this context was built for
        debug_info: Debug dict from strategy.assemble()
        reactivation_decision: Optional reactivation decision for reason extraction
        conn: Optional existing connection

    Returns:
        True if persisted successfully, False otherwise
    """
    def _persist(c: sqlite3.Connection) -> bool:
        cursor = c.cursor()

        # Extract fields from debug_info
        mode = debug_info.get("mode", "unknown")
        active_topic_id = debug_info.get("topic_start_node_id")
        included_node_ids = debug_info.get("included_node_ids", [])
        token_counts = debug_info.get("token_counts", {})
        reactivation_fired = 1 if debug_info.get("reactivation_fired", False) else 0
        truncation_info = debug_info.get("truncation_info")

        # Extract reactivation reason if available
        reactivation_reason = None
        if reactivation_decision is not None:
            if hasattr(reactivation_decision, 'action'):
                reactivation_reason = reactivation_decision.action
                if hasattr(reactivation_decision, 'debug'):
                    reason = reactivation_decision.debug.get("reason")
                    if reason:
                        reactivation_reason = f"{reactivation_decision.action}:{reason}"

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO context_assembly_debug (
                    user_node_id,
                    mode,
                    active_topic_id,
                    included_node_ids_json,
                    token_counts_json,
                    reactivation_fired,
                    reactivation_reason,
                    truncation_info_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_node_id,
                mode,
                active_topic_id,
                json.dumps(included_node_ids),
                json.dumps(token_counts),
                reactivation_fired,
                reactivation_reason,
                json.dumps(truncation_info) if truncation_info else None
            ))
            c.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to persist context debug: {e}")
            return False

    if conn is not None:
        return _persist(conn)
    else:
        with get_connection() as c:
            return _persist(c)


def get_context_assembly_debug(
    user_node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve context assembly debug info for a user node.

    Args:
        user_node_id: The user node to look up
        conn: Optional existing connection

    Returns:
        Debug info dict or None if not found
    """
    def _get(c: sqlite3.Connection) -> Optional[Dict[str, Any]]:
        cursor = c.cursor()
        cursor.execute("""
            SELECT
                mode,
                active_topic_id,
                included_node_ids_json,
                token_counts_json,
                reactivation_fired,
                reactivation_reason,
                truncation_info_json,
                created_at
            FROM context_assembly_debug
            WHERE user_node_id = ?
        """, (user_node_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "mode": row[0],
            "active_topic_id": row[1],
            "included_node_ids": json.loads(row[2]) if row[2] else [],
            "token_counts": json.loads(row[3]) if row[3] else {},
            "reactivation_fired": bool(row[4]),
            "reactivation_reason": row[5],
            "truncation_info": json.loads(row[6]) if row[6] else None,
            "created_at": row[7],
        }

    if conn is not None:
        return _get(conn)
    else:
        with get_connection() as c:
            return _get(c)


def get_recent_context_debug(
    limit: int = 20,
    mode_filter: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get recent context assembly debug records.

    Args:
        limit: Max records to return
        mode_filter: Optional filter by mode
        conn: Optional existing connection

    Returns:
        List of debug info dicts
    """
    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.cursor()

        if mode_filter:
            cursor.execute("""
                SELECT
                    user_node_id,
                    mode,
                    active_topic_id,
                    included_node_ids_json,
                    token_counts_json,
                    reactivation_fired,
                    reactivation_reason,
                    created_at
                FROM context_assembly_debug
                WHERE mode = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (mode_filter, limit))
        else:
            cursor.execute("""
                SELECT
                    user_node_id,
                    mode,
                    active_topic_id,
                    included_node_ids_json,
                    token_counts_json,
                    reactivation_fired,
                    reactivation_reason,
                    created_at
                FROM context_assembly_debug
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "user_node_id": row[0],
                "mode": row[1],
                "active_topic_id": row[2],
                "included_node_ids": json.loads(row[3]) if row[3] else [],
                "token_counts": json.loads(row[4]) if row[4] else {},
                "reactivation_fired": bool(row[5]),
                "reactivation_reason": row[6],
                "created_at": row[7],
            })
        return results

    if conn is not None:
        return _get(conn)
    else:
        with get_connection() as c:
            return _get(c)
