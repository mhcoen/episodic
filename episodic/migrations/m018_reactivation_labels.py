"""
Migration: Add reactivation_labels table for calibration ground truth.

This stores human-labeled ground truth for reactivation decisions,
used for calibrating and evaluating the reactivation probe.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Create the reactivation_labels table."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reactivation_labels (
            user_node_id TEXT PRIMARY KEY,
            ground_truth TEXT NOT NULL,
            labeler TEXT,
            notes TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index for querying by ground truth type
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_reactivation_labels_ground_truth
        ON reactivation_labels(ground_truth)
    """)

    conn.commit()


def rollback(conn: sqlite3.Connection) -> None:
    """Remove the reactivation_labels table."""
    cursor = conn.cursor()

    cursor.execute("DROP INDEX IF EXISTS idx_reactivation_labels_ground_truth")
    cursor.execute("DROP TABLE IF EXISTS reactivation_labels")

    conn.commit()
