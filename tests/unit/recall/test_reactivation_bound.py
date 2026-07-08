"""The reactivation candidate query must be bounded and dormancy-filtered."""

import sqlite3

import pytest

from episodic.recall.reactivation import _get_dormant_topic_centroids


@pytest.fixture
def centroid_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
            start_node_id TEXT NOT NULL, end_node_id TEXT
        );
        CREATE TABLE topic_centroids (
            start_node_id TEXT PRIMARY KEY,
            centroid_medoid_exchange_id TEXT,
            exchange_count INTEGER,
            last_active_turn_idx INTEGER
        );
    """)
    # 100 topics, last_active_turn_idx = 1..100
    for i in range(1, 101):
        conn.execute("INSERT INTO topics (name, start_node_id) VALUES (?, ?)",
                     (f"t{i}", f"start_{i}"))
        conn.execute(
            "INSERT INTO topic_centroids (start_node_id, centroid_medoid_exchange_id, "
            "exchange_count, last_active_turn_idx) VALUES (?, ?, ?, ?)",
            (f"start_{i}", f"medoid_{i}", 3, i),
        )
    conn.commit()
    yield conn
    conn.close()


def test_limit_bounds_result(centroid_db):
    # current_turn = 200 → all 100 topics are dormant (dormancy >= 2).
    rows = _get_dormant_topic_centroids(
        centroid_db, current_turn_idx=200,
        active_topic_start_node_id=None, dormancy_min=2, limit=10,
    )
    assert len(rows) == 10
    # Most-recently-active first: turn_idx 100, 99, ... 91
    assert rows[0]['last_active_turn_idx'] == 100
    assert rows[-1]['last_active_turn_idx'] == 91


def test_dormancy_filter_excludes_recent(centroid_db):
    # current_turn = 100, dormancy_min = 5 → topics with last_active > 95 excluded.
    rows = _get_dormant_topic_centroids(
        centroid_db, current_turn_idx=100,
        active_topic_start_node_id=None, dormancy_min=5, limit=100,
    )
    assert all(r['last_active_turn_idx'] <= 95 for r in rows)
    assert rows[0]['last_active_turn_idx'] == 95


def test_active_topic_excluded(centroid_db):
    rows = _get_dormant_topic_centroids(
        centroid_db, current_turn_idx=200,
        active_topic_start_node_id="start_100", dormancy_min=2, limit=100,
    )
    assert all(r['start_node_id'] != "start_100" for r in rows)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
