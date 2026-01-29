"""
Database operations for reactivation decisions.

Stores and retrieves detailed reactivation probe decisions
for calibration and evaluation.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .db_connection import get_connection
from .recall.reactivation import ReactivationDecision

logger = logging.getLogger(__name__)


def persist_reactivation_decision(
    user_node_id: str,
    decision: ReactivationDecision,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Persist a reactivation probe decision.

    Args:
        user_node_id: ID of the user node that triggered the probe
        decision: The ReactivationDecision returned by probe_reactivation
        conn: Optional database connection

    Returns:
        True if successful, False otherwise
    """
    def _persist(c: sqlite3.Connection) -> bool:
        try:
            debug = decision.debug

            cursor = c.execute("""
                INSERT OR REPLACE INTO reactivation_decisions (
                    user_node_id,
                    decision,
                    reason,
                    confidence,
                    topic_name,
                    topic_start_node_id,
                    candidates_json,
                    support_counts_json,
                    gates_json,
                    best_similarity,
                    best_support_count,
                    dormancy_turns,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_node_id,
                decision.action,
                debug.get('exit_reason') or debug.get('best_topic'),
                debug.get('confidence', 0.0),
                decision.topic_name,
                decision.topic_start_node_id,
                json.dumps(debug.get('candidates', [])),
                json.dumps(debug.get('support_counts', {})),
                json.dumps({
                    'passed': debug.get('gates_passed', []),
                    'failed': debug.get('gates_failed', [])
                }),
                debug.get('best_similarity'),
                debug.get('best_support_count'),
                debug.get('dormancy_turns'),
                datetime.now().isoformat()
            ))

            c.commit()
            return True

        except Exception as e:
            logger.warning(f"Failed to persist reactivation decision: {e}")
            return False

    if conn is not None:
        return _persist(conn)

    with get_connection() as c:
        return _persist(c)


def get_reactivation_decision(
    user_node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a stored reactivation decision by user_node_id.

    Args:
        user_node_id: ID of the user node

    Returns:
        Dict with decision info or None if not found
    """
    def _get(c: sqlite3.Connection) -> Optional[Dict[str, Any]]:
        cursor = c.execute("""
            SELECT
                user_node_id,
                decision,
                reason,
                confidence,
                topic_name,
                topic_start_node_id,
                candidates_json,
                support_counts_json,
                gates_json,
                best_similarity,
                best_support_count,
                dormancy_turns,
                created_at
            FROM reactivation_decisions
            WHERE user_node_id = ?
        """, (user_node_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'user_node_id': row[0],
            'decision': row[1],
            'reason': row[2],
            'confidence': row[3],
            'topic_name': row[4],
            'topic_start_node_id': row[5],
            'candidates': json.loads(row[6]) if row[6] else [],
            'support_counts': json.loads(row[7]) if row[7] else {},
            'gates': json.loads(row[8]) if row[8] else {'passed': [], 'failed': []},
            'best_similarity': row[9],
            'best_support_count': row[10],
            'dormancy_turns': row[11],
            'created_at': row[12],
        }

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def get_recent_reactivation_decisions(
    limit: int = 100,
    decision_filter: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get recent reactivation decisions.

    Args:
        limit: Maximum number of decisions to return
        decision_filter: Optional filter by decision type (CONTINUE, REACTIVATE, DISAMBIGUATE)

    Returns:
        List of decision dicts
    """
    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        if decision_filter:
            cursor = c.execute("""
                SELECT
                    user_node_id,
                    decision,
                    reason,
                    confidence,
                    topic_name,
                    topic_start_node_id,
                    candidates_json,
                    support_counts_json,
                    gates_json,
                    best_similarity,
                    best_support_count,
                    dormancy_turns,
                    created_at
                FROM reactivation_decisions
                WHERE decision = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (decision_filter, limit))
        else:
            cursor = c.execute("""
                SELECT
                    user_node_id,
                    decision,
                    reason,
                    confidence,
                    topic_name,
                    topic_start_node_id,
                    candidates_json,
                    support_counts_json,
                    gates_json,
                    best_similarity,
                    best_support_count,
                    dormancy_turns,
                    created_at
                FROM reactivation_decisions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'user_node_id': row[0],
                'decision': row[1],
                'reason': row[2],
                'confidence': row[3],
                'topic_name': row[4],
                'topic_start_node_id': row[5],
                'candidates': json.loads(row[6]) if row[6] else [],
                'support_counts': json.loads(row[7]) if row[7] else {},
                'gates': json.loads(row[8]) if row[8] else {'passed': [], 'failed': []},
                'best_similarity': row[9],
                'best_support_count': row[10],
                'dormancy_turns': row[11],
                'created_at': row[12],
            })

        return results

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def store_reactivation_label(
    user_node_id: str,
    ground_truth: str,
    labeler: Optional[str] = None,
    notes: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Store a human-labeled ground truth for a reactivation decision.

    Args:
        user_node_id: ID of the user node
        ground_truth: Label - 'reactivate:{topic}', 'continue', or 'new_topic'
        labeler: Optional labeler identifier
        notes: Optional notes about the label

    Returns:
        True if successful
    """
    def _store(c: sqlite3.Connection) -> bool:
        try:
            c.execute("""
                INSERT OR REPLACE INTO reactivation_labels (
                    user_node_id,
                    ground_truth,
                    labeler,
                    notes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                user_node_id,
                ground_truth,
                labeler,
                notes,
                datetime.now().isoformat()
            ))
            c.commit()
            return True
        except Exception as e:
            logger.warning(f"Failed to store reactivation label: {e}")
            return False

    if conn is not None:
        return _store(conn)

    with get_connection() as c:
        return _store(c)


def get_labeled_decisions(
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get all decisions that have ground truth labels.

    Returns:
        List of dicts with both decision info and ground truth
    """
    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.execute("""
            SELECT
                rd.user_node_id,
                rd.decision,
                rd.reason,
                rd.confidence,
                rd.topic_name,
                rd.topic_start_node_id,
                rd.candidates_json,
                rd.support_counts_json,
                rd.gates_json,
                rd.best_similarity,
                rd.best_support_count,
                rd.dormancy_turns,
                rd.created_at,
                rl.ground_truth,
                rl.labeler,
                rl.notes
            FROM reactivation_decisions rd
            JOIN reactivation_labels rl ON rd.user_node_id = rl.user_node_id
            ORDER BY rd.created_at DESC
        """)

        results = []
        for row in cursor.fetchall():
            results.append({
                'user_node_id': row[0],
                'decision': row[1],
                'reason': row[2],
                'confidence': row[3],
                'topic_name': row[4],
                'topic_start_node_id': row[5],
                'candidates': json.loads(row[6]) if row[6] else [],
                'support_counts': json.loads(row[7]) if row[7] else {},
                'gates': json.loads(row[8]) if row[8] else {'passed': [], 'failed': []},
                'best_similarity': row[9],
                'best_support_count': row[10],
                'dormancy_turns': row[11],
                'created_at': row[12],
                'ground_truth': row[13],
                'labeler': row[14],
                'notes': row[15],
            })

        return results

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def get_unlabeled_decisions(
    limit: int = 50,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get decisions that don't have ground truth labels yet.

    Args:
        limit: Maximum number to return

    Returns:
        List of decision dicts
    """
    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.execute("""
            SELECT
                rd.user_node_id,
                rd.decision,
                rd.reason,
                rd.confidence,
                rd.topic_name,
                rd.topic_start_node_id,
                rd.candidates_json,
                rd.support_counts_json,
                rd.gates_json,
                rd.best_similarity,
                rd.best_support_count,
                rd.dormancy_turns,
                rd.created_at
            FROM reactivation_decisions rd
            LEFT JOIN reactivation_labels rl ON rd.user_node_id = rl.user_node_id
            WHERE rl.user_node_id IS NULL
            ORDER BY rd.created_at DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'user_node_id': row[0],
                'decision': row[1],
                'reason': row[2],
                'confidence': row[3],
                'topic_name': row[4],
                'topic_start_node_id': row[5],
                'candidates': json.loads(row[6]) if row[6] else [],
                'support_counts': json.loads(row[7]) if row[7] else {},
                'gates': json.loads(row[8]) if row[8] else {'passed': [], 'failed': []},
                'best_similarity': row[9],
                'best_support_count': row[10],
                'dormancy_turns': row[11],
                'created_at': row[12],
            })

        return results

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)
