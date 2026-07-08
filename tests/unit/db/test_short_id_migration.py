"""Regression test for migrate_to_short_ids assigning unique short IDs.

The migration used to call generate_short_id(), which probes a *different*
pooled connection whose snapshot can't see the migration's uncommitted updates,
so every node received the same candidate and the UNIQUE index creation failed.
"""

import os
import sqlite3
import tempfile

import pytest


def test_migration_assigns_unique_short_ids(monkeypatch):
    db_path = os.path.join(tempfile.mkdtemp(), "legacy.db")
    monkeypatch.setenv("EPISODIC_DB_PATH", db_path)

    # Build a legacy nodes table WITHOUT a short_id column.
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, parent_id TEXT, content TEXT, role TEXT
        )
    """)
    for i in range(50):
        conn.execute(
            "INSERT INTO nodes (id, content, role) VALUES (?, ?, ?)",
            (f"node-{i}", f"content {i}", "user"),
        )
    conn.commit()
    conn.close()

    # Reset the pool so it opens the new DB path.
    from episodic import db_connection
    db_connection.close_pool()
    db_connection._resolved_db_path = None

    from episodic.db_migrations import migrate_to_short_ids
    migrate_to_short_ids()

    check = sqlite3.connect(db_path)
    rows = check.execute("SELECT short_id FROM nodes").fetchall()
    short_ids = [r[0] for r in rows]
    check.close()
    db_connection.close_pool()
    db_connection._resolved_db_path = None

    assert len(short_ids) == 50
    assert all(s is not None for s in short_ids), "every node got a short_id"
    assert len(set(short_ids)) == 50, "short_ids are unique"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
