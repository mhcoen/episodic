"""
Tests for retrieval/segment.py ancestry bounding.

The segment computation must stop walking ancestry at the topic's start node
instead of continuing to the DAG root — that is what keeps promotion off an
O(topics x conversation) hot path — while producing exactly the same segment
membership it did with the unbounded walk.
"""

import pytest

from episodic.retrieval.segment import (
    build_ancestry_map,
    compute_segment_nodes,
    get_cached_segment_nodes,
)
from . import create_test_db


class TestBoundedAncestryMap:
    def test_bounded_map_stops_at_start(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        # Topic 2 spans node_5..node_8. Bounding at node_5 must NOT include any
        # earlier node (node_1..node_4) — i.e. it does not walk to the root.
        amap = build_ancestry_map(conn, "node_8", stop_at="node_5")
        assert set(amap) == {"node_5", "node_6", "node_7", "node_8"}
        assert "node_4" not in amap
        assert "node_1" not in amap

    def test_unbounded_map_reaches_root(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        # Without stop_at, the walk still reaches the root (display path relies
        # on this).
        amap = build_ancestry_map(conn, "node_8")
        assert {"node_1", "node_4", "node_5", "node_8"} <= set(amap)

    def test_bounded_matches_unbounded_within_segment(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        bounded = build_ancestry_map(conn, "node_8", stop_at="node_5")
        unbounded = build_ancestry_map(conn, "node_8")
        # Every node in the bounded map maps to the same parent as unbounded.
        for nid, parent in bounded.items():
            assert unbounded[nid] == parent

    def test_single_node_segment(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        amap = build_ancestry_map(conn, "node_5", stop_at="node_5")
        assert set(amap) == {"node_5"}


class TestComputeSegmentNodes:
    def test_closed_topic_segment(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        # Topic 2 (id=2): node_5..node_8, closed.
        ordered, members = compute_segment_nodes(conn, 2, "node_8")
        assert ordered == ["node_5", "node_6", "node_7", "node_8"]
        assert members == set(ordered)

    def test_ongoing_topic_to_head(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        # Topic 3 (id=3): node_9..(ongoing); effective_end is the head node_12.
        ordered, members = get_cached_segment_nodes(conn, 3)
        assert ordered == ["node_9", "node_10", "node_11", "node_12"]
        assert members == set(ordered)

    def test_first_topic_segment(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        ordered, members = compute_segment_nodes(conn, 1, "node_4")
        assert ordered == ["node_1", "node_2", "node_3", "node_4"]

    def test_start_not_ancestor_returns_empty(self, tmp_path):
        conn, _ = create_test_db(tmp_path)
        # Topic 3 starts at node_9, but here effective_end is node_4 (earlier in
        # the chain), so the topic's start is never reached. The walk falls back
        # to root and the segment is empty — same as the old unbounded behavior.
        ordered, members = compute_segment_nodes(conn, 3, "node_4")
        assert ordered == []
        assert members == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
