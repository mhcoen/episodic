"""Tests for MCP migrations m021, m022, and m023."""

import sqlite3

import pytest

from episodic.migrations.m021_mcp_auth_tables import migration
from episodic.migrations.m022_mcp_traces import migration as m022_migration
from episodic.migrations.m023_thread_handles import migration as m023_migration


class TestMcpAuthMigration:
    """Tests for m021_mcp_auth_tables migration."""

    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        yield conn
        conn.close()

    def test_migration_version(self):
        assert migration.version == 21

    def test_up_creates_mcp_tokens(self, db):
        migration.up(db)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_tokens'"
        )
        assert cursor.fetchone() is not None

    def test_up_creates_mcp_cost_accounting(self, db):
        migration.up(db)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_cost_accounting'"
        )
        assert cursor.fetchone() is not None

    def test_up_is_idempotent(self, db):
        migration.up(db)
        migration.up(db)  # Should not raise

    def test_down_removes_tables(self, db):
        migration.up(db)
        migration.down(db)
        for table in ("mcp_tokens", "mcp_cost_accounting"):
            cursor = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            assert cursor.fetchone() is None

    def test_token_table_schema(self, db):
        migration.up(db)
        # Insert a token row
        db.execute(
            "INSERT INTO mcp_tokens (token_id, token_hash, client_id, scopes) "
            "VALUES ('tid', 'hash', 'client', '[]')"
        )
        row = db.execute("SELECT * FROM mcp_tokens WHERE token_id='tid'").fetchone()
        assert row is not None

    def test_cost_table_composite_key(self, db):
        migration.up(db)
        db.execute(
            "INSERT INTO mcp_cost_accounting (client_id, date, total_cost) "
            "VALUES ('c1', '2025-01-01', 1.5)"
        )
        # Same key should conflict
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO mcp_cost_accounting (client_id, date, total_cost) "
                "VALUES ('c1', '2025-01-01', 2.0)"
            )


class TestMcpTracesMigration:
    """Tests for m022_mcp_traces migration."""

    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        yield conn
        conn.close()

    def test_migration_version(self):
        assert m022_migration.version == 22

    def test_up_creates_mcp_traces(self, db):
        m022_migration.up(db)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_traces'"
        )
        assert cursor.fetchone() is not None

    def test_up_creates_indices(self, db):
        m022_migration.up(db)
        indices = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='mcp_traces'"
        ).fetchall()
        index_names = [i[0] for i in indices]
        assert "idx_traces_timestamp" in index_names
        assert "idx_traces_tool" in index_names
        assert "idx_traces_client" in index_names

    def test_up_is_idempotent(self, db):
        m022_migration.up(db)
        m022_migration.up(db)  # Should not raise

    def test_down_removes_table(self, db):
        m022_migration.up(db)
        m022_migration.down(db)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_traces'"
        )
        assert cursor.fetchone() is None

    def test_traces_table_schema(self, db):
        m022_migration.up(db)
        cursor = db.execute("PRAGMA table_info(mcp_traces)")
        columns = {row[1] for row in cursor.fetchall()}
        required = {
            "trace_id", "schema_version", "timestamp_start", "timestamp_end",
            "duration_ms", "direction", "tool_name", "status",
            "input_hash", "input_size_bytes", "output_hash", "output_size_bytes",
            "origin", "purpose", "request_id",
        }
        assert required.issubset(columns)

    def test_insert_trace_row(self, db):
        m022_migration.up(db)
        db.execute(
            "INSERT INTO mcp_traces "
            "(trace_id, timestamp_start, timestamp_end, duration_ms, direction, "
            "tool_name, origin, purpose, request_id, input_hash, input_size_bytes, "
            "status, output_hash, output_size_bytes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("t1", "2024-01-01T00:00:00", "2024-01-01T00:00:01", 1000,
             "server_tool_call", "get_topics", "mcp_server", "interactive",
             "req-1", "abc", 100, "ok", "def", 200),
        )
        row = db.execute("SELECT * FROM mcp_traces WHERE trace_id='t1'").fetchone()
        assert row is not None


class TestThreadHandlesMigration:
    """Tests for m023_thread_handles migration."""

    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        # Create conversations table (FK target)
        conn.execute("""
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                root_node_id TEXT,
                current_head_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
        """)
        yield conn
        conn.close()

    def test_migration_version(self):
        assert m023_migration.version == 23

    def test_up_creates_table(self, db):
        m023_migration.up(db)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='mcp_thread_handles'"
        )
        assert cursor.fetchone() is not None

    def test_up_creates_indices(self, db):
        m023_migration.up(db)
        indices = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='mcp_thread_handles'"
        ).fetchall()
        index_names = [i[0] for i in indices]
        assert "idx_handles_thread" in index_names
        assert "idx_handles_client" in index_names
        assert "idx_handles_hash" in index_names

    def test_up_is_idempotent(self, db):
        m023_migration.up(db)
        m023_migration.up(db)  # Should not raise

    def test_down_removes_table(self, db):
        m023_migration.up(db)
        m023_migration.down(db)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='mcp_thread_handles'"
        )
        assert cursor.fetchone() is None

    def test_table_schema(self, db):
        m023_migration.up(db)
        cursor = db.execute("PRAGMA table_info(mcp_thread_handles)")
        columns = {row[1] for row in cursor.fetchall()}
        required = {
            "handle_id", "handle_hash", "thread_id", "client_id",
            "permissions", "created_at", "revoked_at",
        }
        assert required.issubset(columns)

    def test_insert_handle_row(self, db):
        m023_migration.up(db)
        # Create a conversation first (FK target)
        db.execute(
            "INSERT INTO conversations (conversation_id) VALUES ('conv-1')"
        )
        db.execute(
            "INSERT INTO mcp_thread_handles "
            "(handle_id, handle_hash, thread_id, client_id, permissions) "
            "VALUES ('h1', 'hash1', 1, 'client1', '[\"read\",\"write\"]')"
        )
        row = db.execute(
            "SELECT * FROM mcp_thread_handles WHERE handle_id='h1'"
        ).fetchone()
        assert row is not None

    def test_unique_handle_hash(self, db):
        m023_migration.up(db)
        db.execute(
            "INSERT INTO conversations (conversation_id) VALUES ('conv-1')"
        )
        db.execute(
            "INSERT INTO mcp_thread_handles "
            "(handle_id, handle_hash, thread_id, client_id, permissions) "
            "VALUES ('h1', 'hash1', 1, 'client1', '[]')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO mcp_thread_handles "
                "(handle_id, handle_hash, thread_id, client_id, permissions) "
                "VALUES ('h2', 'hash1', 1, 'client2', '[]')"
            )
