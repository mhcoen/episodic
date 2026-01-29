"""
Migration: Add context_assembly_debug table for instrumentation.

This table stores debug information from each context assembly operation,
keyed by user_node_id for later auditing.
"""

import sqlite3
from typing import Optional


def migrate(conn: sqlite3.Connection) -> None:
    """Create context_assembly_debug table."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_assembly_debug (
            user_node_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            active_topic_id TEXT,
            included_node_ids_json TEXT NOT NULL DEFAULT '[]',
            token_counts_json TEXT NOT NULL DEFAULT '{}',
            reactivation_fired INTEGER NOT NULL DEFAULT 0,
            reactivation_reason TEXT,
            truncation_info_json TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index for querying by mode or topic
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_context_debug_mode
        ON context_assembly_debug(mode)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_context_debug_topic
        ON context_assembly_debug(active_topic_id)
    """)

    conn.commit()


def rollback(conn: sqlite3.Connection) -> None:
    """Drop context_assembly_debug table."""
    cursor = conn.cursor()
    cursor.execute("DROP INDEX IF EXISTS idx_context_debug_mode")
    cursor.execute("DROP INDEX IF EXISTS idx_context_debug_topic")
    cursor.execute("DROP TABLE IF EXISTS context_assembly_debug")
    conn.commit()


def is_applied(conn: sqlite3.Connection) -> bool:
    """Check if migration has been applied."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='context_assembly_debug'
    """)
    return cursor.fetchone() is not None
