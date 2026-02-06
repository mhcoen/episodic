"""Add MCP authentication tables.

Creates mcp_tokens for capability-based auth and mcp_cost_accounting
for per-client daily cost tracking.
"""

import sqlite3
from episodic.migrations import Migration


class AddMcpAuthTables(Migration):
    """Add mcp_tokens and mcp_cost_accounting tables."""

    def __init__(self):
        super().__init__(
            version=21,
            description="Add MCP auth tokens and cost accounting tables"
        )

    def up(self, conn: sqlite3.Connection) -> None:
        """Create MCP auth tables."""
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_tokens (
                token_id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL UNIQUE,
                client_id TEXT NOT NULL,
                scopes TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT,
                revoked_at TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcp_tokens_client "
            "ON mcp_tokens(client_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcp_tokens_hash "
            "ON mcp_tokens(token_hash)"
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_cost_accounting (
                client_id TEXT NOT NULL,
                date TEXT NOT NULL,
                total_cost REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (client_id, date)
            )
        """)

        conn.commit()

    def down(self, conn: sqlite3.Connection) -> None:
        """Remove MCP auth tables."""
        cursor = conn.cursor()
        cursor.execute("DROP INDEX IF EXISTS idx_mcp_tokens_client")
        cursor.execute("DROP INDEX IF EXISTS idx_mcp_tokens_hash")
        cursor.execute("DROP TABLE IF EXISTS mcp_tokens")
        cursor.execute("DROP TABLE IF EXISTS mcp_cost_accounting")
        conn.commit()


migration = AddMcpAuthTables()
