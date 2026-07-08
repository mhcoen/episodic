"""
Unit tests for db_ids.generate_short_id.

Focus: the scan-free hot path — generate_short_id must probe the unique index
for a free candidate rather than loading every existing short_id, still return
unique IDs, escalate length when a shorter space is saturated, and preserve the
legacy "no short_id column → None" contract.
"""

import sqlite3
import contextlib

import pytest

from episodic import db_ids
from episodic.db_ids import generate_short_id
from episodic.db_connection import get_connection


def _existing_short_ids():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT short_id FROM nodes WHERE short_id IS NOT NULL"
        ).fetchall()
    return {r[0] for r in rows}


class TestGenerateShortId:
    def test_fresh_db_returns_two_char_free_id(self, temp_database):
        sid = generate_short_id()
        assert sid is not None
        assert len(sid) == 2
        assert sid not in _existing_short_ids()

    def test_returned_id_is_currently_free(self, temp_database):
        from episodic.db_nodes import insert_node

        for i in range(30):
            insert_node(f"msg {i}")

        existing = _existing_short_ids()
        sid = generate_short_id()
        assert sid not in existing

    def test_hot_path_ids_are_unique(self, temp_database):
        from episodic.db_nodes import insert_node

        seen = set()
        for i in range(200):
            _, sid = insert_node(f"msg {i}")
            assert sid not in seen, f"duplicate short_id {sid} at insert {i}"
            seen.add(sid)

    def test_escalates_to_three_chars_when_two_char_saturated(self, temp_database):
        # Fill the entire 2-char letter space so no 2-char candidate is free.
        letters = "abcdefghijklmnopqrstuvwxyz"
        with get_connection() as conn:
            for a in letters:
                for b in letters:
                    conn.execute(
                        "INSERT INTO nodes (id, short_id, content, role) "
                        "VALUES (?, ?, ?, ?)",
                        (f"id-{a}{b}", a + b, "x", "user"),
                    )
            conn.commit()

        sid = generate_short_id()
        assert sid is not None
        assert len(sid) == 3, f"expected escalation to 3 chars, got {sid!r}"
        assert sid not in _existing_short_ids()

    def test_missing_short_id_column_returns_none(self, monkeypatch):
        # A nodes table without a short_id column must yield None, not raise.
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT)")
        conn.commit()

        @contextlib.contextmanager
        def fake_get_connection():
            yield conn

        monkeypatch.setattr(db_ids, "get_connection", fake_get_connection)

        assert generate_short_id() is None
        conn.close()

    def test_no_full_table_scan_or_pragma(self, monkeypatch):
        # Regression guard for the O(N) hot-path scan: generate_short_id must
        # probe the index (WHERE short_id = ?), never load every short_id and
        # never run a per-call PRAGMA table_info.
        executed = []

        real = sqlite3.connect(":memory:")
        real.execute(
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, short_id TEXT UNIQUE, "
            "content TEXT, role TEXT)"
        )
        real.commit()

        class RecordingCursor:
            def __init__(self, cur):
                self._cur = cur

            def execute(self, sql, *args):
                executed.append(sql)
                return self._cur.execute(sql, *args)

            def fetchone(self):
                return self._cur.fetchone()

            def fetchall(self):
                return self._cur.fetchall()

        class RecordingConn:
            def cursor(self):
                return RecordingCursor(real.cursor())

        @contextlib.contextmanager
        def fake_get_connection():
            yield RecordingConn()

        monkeypatch.setattr(db_ids, "get_connection", fake_get_connection)

        sid = generate_short_id()
        assert sid is not None

        joined = " ".join(executed).lower()
        assert "pragma table_info" not in joined
        # The old scan selected every short_id; the new probe filters by value.
        assert "select short_id from nodes" not in joined
        assert any("where short_id = ?" in sql.lower() for sql in executed)
        real.close()

    def test_real_error_propagates(self, monkeypatch):
        # A genuine sqlite error (not a missing column) must NOT be swallowed.
        class BoomCursor:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("database is locked")

        class BoomConn:
            def cursor(self):
                return BoomCursor()

        @contextlib.contextmanager
        def fake_get_connection():
            yield BoomConn()

        monkeypatch.setattr(db_ids, "get_connection", fake_get_connection)

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            generate_short_id()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
