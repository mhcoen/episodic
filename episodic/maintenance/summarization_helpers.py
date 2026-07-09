"""Summarization DB + hash helper functions.

Split out of summarization.py to keep it under the size limit. These are leaf
helpers: exchange fetching, turn-index lookups, stale-topic detection, node-id
hashing, and structured-summary id preservation. They do not call the
summarize_* functions in summarization.py, so there is no import cycle.
"""

import hashlib
import sqlite3
from typing import Any, Dict, List, Optional

from episodic.config import config
from episodic.db_connection import get_connection

from .summary_spec import (
    LastState,
    StructuredSummary,
)


def get_exchanges_since_turn(
    topic_start_node_id: str,
    since_turn_idx: Optional[int],
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get all exchanges in a topic since a specific turn index.

    Args:
        topic_start_node_id: The topic to get exchanges from
        since_turn_idx: Only get exchanges after this turn (None = all)
        conn: Optional database connection

    Returns:
        List of exchange dicts with user/assistant content
    """
    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.cursor()

        if since_turn_idx is not None:
            cursor.execute("""
                SELECT tn.node_id, tn.turn_idx, tn.role, n.content
                FROM topic_nodes tn
                JOIN nodes n ON tn.node_id = n.id
                WHERE tn.topic_start_node_id = ?
                AND tn.turn_idx > ?
                AND tn.role IN ('user', 'assistant')
                ORDER BY tn.turn_idx ASC
            """, (topic_start_node_id, since_turn_idx))
        else:
            cursor.execute("""
                SELECT tn.node_id, tn.turn_idx, tn.role, n.content
                FROM topic_nodes tn
                JOIN nodes n ON tn.node_id = n.id
                WHERE tn.topic_start_node_id = ?
                AND tn.role IN ('user', 'assistant')
                ORDER BY tn.turn_idx ASC
            """, (topic_start_node_id,))

        rows = cursor.fetchall()

        # Build exchange pairs
        exchanges = []
        i = 0
        while i < len(rows):
            if rows[i][2] == 'user':
                user_node_id, user_turn_idx, _, user_content = rows[i]
                # Look for following assistant message
                asst_content = None
                asst_node_id = None
                asst_turn_idx = None
                if i + 1 < len(rows) and rows[i + 1][2] == 'assistant':
                    asst_node_id, asst_turn_idx, _, asst_content = rows[i + 1]
                    i += 2
                else:
                    i += 1

                exchanges.append({
                    'user_node_id': user_node_id,
                    'user_content': user_content,
                    'user_turn_idx': user_turn_idx,
                    'assistant_node_id': asst_node_id,
                    'assistant_content': asst_content,
                    'assistant_turn_idx': asst_turn_idx,
                })
            else:
                i += 1

        return exchanges

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def get_max_turn_idx(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[int]:
    """Get the maximum turn_idx for a topic."""
    def _get(c: sqlite3.Connection) -> Optional[int]:
        cursor = c.cursor()
        cursor.execute("""
            SELECT MAX(turn_idx) FROM topic_nodes
            WHERE topic_start_node_id = ?
        """, (topic_start_node_id,))
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def get_stale_topics(
    min_new_exchanges: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Find topics that need summarization.

    A topic is "stale" if:
    - It has no summary yet (last_summarized_turn_idx is NULL)
    - It has grown by min_new_exchanges since last summary

    Args:
        min_new_exchanges: Minimum new exchanges to trigger re-summarization
        conn: Optional database connection

    Returns:
        List of topic dicts with start_node_id, name, exchange_count, etc.
    """
    if min_new_exchanges is None:
        min_new_exchanges = config.get("summary_min_new_exchanges", 4)

    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.cursor()

        # Get all topics with their working set state
        cursor.execute("""
            SELECT
                t.start_node_id,
                t.name,
                t.end_node_id,
                ws.last_summarized_turn_idx,
                ws.summary_md,
                (SELECT MAX(turn_idx) FROM topic_nodes WHERE topic_start_node_id = t.start_node_id) as max_turn,
                (SELECT COUNT(*) FROM topic_nodes WHERE topic_start_node_id = t.start_node_id) as node_count
            FROM topics t
            LEFT JOIN topic_working_set ws ON t.start_node_id = ws.topic_start_node_id
            ORDER BY t.rowid ASC
        """)

        stale = []
        for row in cursor.fetchall():
            start_node_id = row[0]
            name = row[1]
            end_node_id = row[2]
            last_summarized = row[3]
            existing_summary = row[4]
            max_turn = row[5]
            node_count = row[6]

            if max_turn is None:
                continue  # Empty topic

            # Calculate new exchanges since last summary
            if last_summarized is None:
                # Never summarized - count all exchanges
                new_exchanges = node_count // 2  # Approximate exchange count
            else:
                # Count turns since last summary
                new_turns = max_turn - last_summarized
                new_exchanges = new_turns // 2  # Approximate

            # Check if stale
            is_stale = (
                last_summarized is None or  # Never summarized
                new_exchanges >= min_new_exchanges  # Has grown enough
            )

            if is_stale:
                stale.append({
                    'start_node_id': start_node_id,
                    'name': name,
                    'end_node_id': end_node_id,
                    'last_summarized_turn_idx': last_summarized,
                    'existing_summary': existing_summary,
                    'max_turn_idx': max_turn,
                    'node_count': node_count,
                    'new_exchanges': new_exchanges,
                })

        return stale

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def compute_node_ids_hash(node_ids: List[str]) -> str:
    """Hash of ordered node IDs for provenance."""
    return hashlib.sha256("|".join(sorted(node_ids)).encode()).hexdigest()[:16]


def preserve_ids(
    old_summary: Optional[StructuredSummary],
    new_summary: StructuredSummary,
) -> StructuredSummary:
    """
    Preserve IDs from old summary when decisions/open_loops match.

    v1 strategy: replace context + last_state, preserve IDs by text matching.
    """
    if not old_summary:
        return new_summary

    # Build lookup by normalized text
    old_decisions = {d.decision.lower().strip(): d.id for d in old_summary.decisions}
    old_loops = {o.question.lower().strip(): o.id for o in old_summary.open_loops}

    # Assign IDs to new decisions
    for d in new_summary.decisions:
        key = d.decision.lower().strip()
        if key in old_decisions:
            d.id = old_decisions[key]

    # Assign IDs to new open_loops
    for o in new_summary.open_loops:
        key = o.question.lower().strip()
        if key in old_loops:
            o.id = old_loops[key]

    return new_summary

