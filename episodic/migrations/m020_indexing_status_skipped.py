"""Add 'skipped' status to indexing_status table.

Updates the CHECK constraint to allow 'skipped' status for nodes
that can't be indexed (e.g., user messages without assistant responses).
"""

import sqlite3
from episodic.migrations import Migration


class IndexingStatusSkipped(Migration):
    """Add 'skipped' status to indexing_status table."""

    def __init__(self):
        super().__init__(
            version=20,
            description="Add 'skipped' status to indexing_status CHECK constraint"
        )

    def up(self, conn: sqlite3.Connection) -> None:
        """Recreate table with updated CHECK constraint."""
        cursor = conn.cursor()

        # SQLite doesn't support ALTER CHECK constraint, so we recreate the table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexing_status_new (
                node_id TEXT NOT NULL,
                index_type TEXT NOT NULL DEFAULT 'conversation',
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'ok', 'failed', 'skipped')),
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

        # Copy existing data
        cursor.execute("""
            INSERT OR IGNORE INTO indexing_status_new
            SELECT * FROM indexing_status
        """)

        # Drop old table and indexes
        cursor.execute("DROP INDEX IF EXISTS idx_indexing_status_status")
        cursor.execute("DROP INDEX IF EXISTS idx_indexing_status_type")
        cursor.execute("DROP TABLE IF EXISTS indexing_status")

        # Rename new table
        cursor.execute("ALTER TABLE indexing_status_new RENAME TO indexing_status")

        # Recreate indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_status_status ON indexing_status(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_status_type ON indexing_status(index_type)")

        conn.commit()

    def down(self, conn: sqlite3.Connection) -> None:
        """Revert to original CHECK constraint (loses 'skipped' entries)."""
        cursor = conn.cursor()

        # Delete any 'skipped' entries first
        cursor.execute("DELETE FROM indexing_status WHERE status = 'skipped'")

        # Recreate with original constraint
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexing_status_old (
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

        cursor.execute("""
            INSERT OR IGNORE INTO indexing_status_old
            SELECT * FROM indexing_status WHERE status != 'skipped'
        """)

        cursor.execute("DROP INDEX IF EXISTS idx_indexing_status_status")
        cursor.execute("DROP INDEX IF EXISTS idx_indexing_status_type")
        cursor.execute("DROP TABLE IF EXISTS indexing_status")
        cursor.execute("ALTER TABLE indexing_status_old RENAME TO indexing_status")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_status_status ON indexing_status(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexing_status_type ON indexing_status(index_type)")

        conn.commit()


# Create migration instance
migration = IndexingStatusSkipped()
