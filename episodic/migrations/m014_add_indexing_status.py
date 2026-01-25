"""Add indexing_status table for durable tracking of RAG indexing.

This table tracks which nodes have been indexed into ChromaDB,
allowing recovery from failures and visibility into indexing state.
"""

import sqlite3
from episodic.migrations import Migration


class AddIndexingStatus(Migration):
    """Add indexing_status table for durable indexing tracking."""

    def __init__(self):
        super().__init__(
            version=14,
            description="Add indexing_status table for durable RAG indexing tracking"
        )

    def up(self, conn: sqlite3.Connection) -> None:
        """Create the indexing_status table."""
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

        conn.commit()

    def down(self, conn: sqlite3.Connection) -> None:
        """Remove the indexing_status table."""
        cursor = conn.cursor()
        cursor.execute("DROP INDEX IF EXISTS idx_indexing_status_status")
        cursor.execute("DROP INDEX IF EXISTS idx_indexing_status_type")
        cursor.execute("DROP TABLE IF EXISTS indexing_status")
        conn.commit()


# Create migration instance
migration = AddIndexingStatus()
