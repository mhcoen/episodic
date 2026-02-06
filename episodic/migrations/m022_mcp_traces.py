"""Add MCP traces table for tool call tracing.

Per spec section 12: full trace recording with retention/eviction.
"""

import sqlite3
from episodic.migrations import Migration


class AddMcpTraces(Migration):
    """Add mcp_traces table for tool call tracing."""

    def __init__(self):
        super().__init__(
            version=22,
            description="Add MCP traces table for tool call tracing"
        )

    def up(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_traces (
                trace_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL DEFAULT '1.0',
                timestamp_start TEXT NOT NULL,
                timestamp_end TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                direction TEXT NOT NULL,
                server_id TEXT,
                tool_name TEXT NOT NULL,
                client_id TEXT,
                thread_id TEXT,
                origin TEXT NOT NULL,
                purpose TEXT NOT NULL,
                request_id TEXT NOT NULL,
                parameter_schema_version TEXT,
                parameters_redacted TEXT,
                input_hash TEXT NOT NULL,
                input_size_bytes INTEGER NOT NULL,
                model_provider TEXT,
                model_id TEXT,
                token_in INTEGER,
                token_out INTEGER,
                cache_hit INTEGER,
                retries INTEGER DEFAULT 0,
                timeout_ms INTEGER,
                status TEXT NOT NULL,
                output_hash TEXT NOT NULL,
                output_size_bytes INTEGER NOT NULL,
                error_code TEXT,
                message_safe TEXT,
                detail_debug TEXT
            )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_timestamp "
            "ON mcp_traces(timestamp_start)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_tool "
            "ON mcp_traces(tool_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_client "
            "ON mcp_traces(client_id)"
        )

        conn.commit()

    def down(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute("DROP INDEX IF EXISTS idx_traces_timestamp")
        cursor.execute("DROP INDEX IF EXISTS idx_traces_tool")
        cursor.execute("DROP INDEX IF EXISTS idx_traces_client")
        cursor.execute("DROP TABLE IF EXISTS mcp_traces")
        conn.commit()


migration = AddMcpTraces()
