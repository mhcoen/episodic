"""Add MCP thread handles table for conversation thread access control.

Per spec section 5.7: thread handles with permissions for MCP clients.
"""

import sqlite3
from episodic.migrations import Migration


class AddThreadHandles(Migration):
    """Add mcp_thread_handles table for thread access control."""

    def __init__(self):
        super().__init__(
            version=23,
            description="Add MCP thread handles table"
        )

    def up(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_thread_handles (
                handle_id TEXT PRIMARY KEY,
                handle_hash TEXT NOT NULL UNIQUE,
                thread_id INTEGER NOT NULL,
                client_id TEXT NOT NULL,
                permissions TEXT NOT NULL DEFAULT '["read","write"]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                revoked_at TEXT,
                FOREIGN KEY (thread_id) REFERENCES conversations(id)
            )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_handles_thread "
            "ON mcp_thread_handles(thread_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_handles_client "
            "ON mcp_thread_handles(client_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_handles_hash "
            "ON mcp_thread_handles(handle_hash)"
        )

        conn.commit()

    def down(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute("DROP INDEX IF EXISTS idx_handles_thread")
        cursor.execute("DROP INDEX IF EXISTS idx_handles_client")
        cursor.execute("DROP INDEX IF EXISTS idx_handles_hash")
        cursor.execute("DROP TABLE IF EXISTS mcp_thread_handles")
        conn.commit()


migration = AddThreadHandles()
