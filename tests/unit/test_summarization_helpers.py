"""Smoke tests for maintenance/summarization DB + hash helpers.

Establishes a safety net (this module was previously untested) before splitting
the helpers out. The DB helpers accept an explicit conn, so no mocking needed.
"""

import sqlite3

import pytest

from episodic.maintenance.summarization import (
    compute_node_ids_hash,
    get_max_turn_idx,
    get_stale_topics,
)


class TestComputeNodeIdsHash:
    def test_deterministic(self):
        h1 = compute_node_ids_hash(["a", "b", "c"])
        h2 = compute_node_ids_hash(["a", "b", "c"])
        assert h1 == h2
        assert len(h1) == 16

    def test_order_independent(self):
        assert compute_node_ids_hash(["a", "b", "c"]) == compute_node_ids_hash(["c", "a", "b"])

    def test_distinct_inputs_differ(self):
        assert compute_node_ids_hash(["a", "b"]) != compute_node_ids_hash(["a", "c"])


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE topics (
            rowid_ INTEGER, start_node_id TEXT, name TEXT, end_node_id TEXT
        );
        CREATE TABLE topic_nodes (
            topic_start_node_id TEXT, node_id TEXT, turn_idx INTEGER, role TEXT
        );
        CREATE TABLE topic_working_set (
            topic_start_node_id TEXT PRIMARY KEY,
            last_summarized_turn_idx INTEGER, summary_md TEXT
        );
    """)
    # Topic A: 6 nodes, never summarized -> stale.
    for i in range(1, 7):
        conn.execute("INSERT INTO topic_nodes VALUES ('A', ?, ?, 'user')", (f"a{i}", i))
    # Topic B: 4 nodes, summarized recently -> not stale (few new exchanges).
    for i in range(10, 14):
        conn.execute("INSERT INTO topic_nodes VALUES ('B', ?, ?, 'user')", (f"b{i}", i))
    conn.execute("INSERT INTO topics VALUES (1, 'A', 'Topic A', NULL)")
    conn.execute("INSERT INTO topics VALUES (2, 'B', 'Topic B', NULL)")
    conn.execute("INSERT INTO topic_working_set VALUES ('B', 13, 'summary')")
    conn.commit()
    yield conn
    conn.close()


class TestDbHelpers:
    def test_get_max_turn_idx(self, db):
        assert get_max_turn_idx("A", conn=db) == 6
        assert get_max_turn_idx("B", conn=db) == 13
        assert get_max_turn_idx("nonexistent", conn=db) is None

    def test_get_stale_topics_flags_unsummarized(self, db):
        stale = get_stale_topics(min_new_exchanges=4, conn=db)
        names = {t["name"] for t in stale}
        # A never summarized -> stale; B summarized at its max turn -> not stale.
        assert "Topic A" in names
        assert "Topic B" not in names

    def test_stale_topic_shape(self, db):
        stale = get_stale_topics(min_new_exchanges=4, conn=db)
        a = next(t for t in stale if t["name"] == "Topic A")
        assert a["start_node_id"] == "A"
        assert a["max_turn_idx"] == 6
        assert a["last_summarized_turn_idx"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
