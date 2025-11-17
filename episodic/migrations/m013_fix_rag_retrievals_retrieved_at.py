"""Fix rag_retrievals table to match m011 schema.

This migration handles the case where rag_retrievals was created by
db_rag.py:create_rag_tables() with the old schema instead of using
the migration system. It updates the table to match the m011 schema.
"""

import sqlite3
from episodic.migrations import Migration


class FixRagRetrievalsSchema(Migration):
    """Fix rag_retrievals table schema to include retrieved_at column."""

    def __init__(self):
        super().__init__(
            version=13,
            description="Fix rag_retrievals schema to include retrieved_at column"
        )

    def up(self, conn: sqlite3.Connection) -> None:
        """Update rag_retrievals table to include retrieved_at column."""
        cursor = conn.cursor()

        # Check if the table exists and what columns it has
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='rag_retrievals'")
        result = cursor.fetchone()

        if result is None:
            # Table doesn't exist, will be created by db_rag.py with correct schema
            return

        table_sql = result[0]

        # Check if retrieved_at column already exists
        if 'retrieved_at' in table_sql:
            # Already has the correct schema
            return

        # Need to recreate table with new schema
        # SQLite doesn't support ALTER TABLE to change schema significantly

        # 1. Rename old table
        cursor.execute('ALTER TABLE rag_retrievals RENAME TO rag_retrievals_old')

        # 2. Create new table with correct schema (matching m011)
        cursor.execute('''
            CREATE TABLE rag_retrievals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                retrieved_doc_ids TEXT,
                chunk_texts TEXT,
                retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 3. Migrate data if old table had any
        # Try to map old columns to new ones where possible
        cursor.execute("SELECT COUNT(*) FROM rag_retrievals_old")
        if cursor.fetchone()[0] > 0:
            # Old schema had: node_id, document_id, relevance_score, created_at, was_helpful
            # New schema needs: message, retrieved_doc_ids, chunk_texts, retrieved_at
            # We can't perfectly map these, so we'll just preserve timestamps
            cursor.execute('''
                INSERT INTO rag_retrievals (id, message, retrieved_at)
                SELECT id, COALESCE(node_id, ''), COALESCE(created_at, CURRENT_TIMESTAMP)
                FROM rag_retrievals_old
            ''')

        # 4. Drop old table
        cursor.execute('DROP TABLE rag_retrievals_old')

        # 5. Create index
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_retrieved_at ON rag_retrievals(retrieved_at)')

        conn.commit()

    def down(self, conn: sqlite3.Connection) -> None:
        """Revert to old schema (not recommended)."""
        cursor = conn.cursor()

        # Rename current table
        cursor.execute('ALTER TABLE rag_retrievals RENAME TO rag_retrievals_new')

        # Recreate old schema
        cursor.execute('''
            CREATE TABLE rag_retrievals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT REFERENCES nodes(id),
                document_id TEXT,
                relevance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                was_helpful BOOLEAN DEFAULT NULL
            )
        ''')

        # Try to migrate data back
        cursor.execute('''
            INSERT INTO rag_retrievals (id, node_id, created_at)
            SELECT id, message, retrieved_at
            FROM rag_retrievals_new
        ''')

        # Drop new table
        cursor.execute('DROP TABLE rag_retrievals_new')

        conn.commit()


# Create migration instance
migration = FixRagRetrievalsSchema()
