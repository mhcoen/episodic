"""
Migration: Add reactivation_decisions table for probe calibration.

This stores detailed feature information for every reactivation probe decision,
enabling replay-based calibration and evaluation.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Create the reactivation_decisions table."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reactivation_decisions (
            user_node_id TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            reason TEXT,
            confidence REAL,
            topic_name TEXT,
            topic_start_node_id TEXT,
            candidates_json TEXT NOT NULL DEFAULT '[]',
            support_counts_json TEXT NOT NULL DEFAULT '{}',
            gates_json TEXT NOT NULL DEFAULT '{"passed": [], "failed": []}',
            best_similarity REAL,
            best_support_count INTEGER,
            dormancy_turns INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index for querying by decision type
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_reactivation_decisions_decision
        ON reactivation_decisions(decision)
    """)

    # Index for querying by timestamp
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_reactivation_decisions_created_at
        ON reactivation_decisions(created_at)
    """)

    # Index for querying by topic
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_reactivation_decisions_topic
        ON reactivation_decisions(topic_start_node_id)
    """)

    conn.commit()


def rollback(conn: sqlite3.Connection) -> None:
    """Remove the reactivation_decisions table."""
    cursor = conn.cursor()

    cursor.execute("DROP INDEX IF EXISTS idx_reactivation_decisions_topic")
    cursor.execute("DROP INDEX IF EXISTS idx_reactivation_decisions_created_at")
    cursor.execute("DROP INDEX IF EXISTS idx_reactivation_decisions_decision")
    cursor.execute("DROP TABLE IF EXISTS reactivation_decisions")

    conn.commit()
