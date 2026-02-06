"""Tests for MCP auth migration m021."""

import sqlite3

import pytest

from episodic.migrations.m021_mcp_auth_tables import migration


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
