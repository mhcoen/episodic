"""
Tests for recall/promotion.py

Tests exchange-to-topic promotion with SQLite membership.
"""

import pytest
from . import create_test_db, create_overlap_db, create_hits


class TestPromotionFirstMatchIdAsc:
    """Test that first matching topic (by id ASC) wins for overlapping membership."""
    
    def test_promote_first_match_id_asc(self, tmp_path):
        """Overlapping exchange assigned to topic with smaller id; audit logs both."""
        from episodic.recall.promotion import promote_hits_to_topics
        
        conn = create_overlap_db(tmp_path)
        
        # node_2 is in both topic 1 (id=1) and topic 2 (id=2)
        hits = create_hits(['node_2'], [0.9])
        
        result = promote_hits_to_topics(conn, hits)
        
        # Should be assigned to topic 1 (lower id)
        assert 1 in result.by_topic
        assert result.by_topic[1][0].exchange_id == 'node_2'
        assert result.by_topic[1][0].topic_id == 1
        
        # Audit should contain overlap entry
        assert len(result.audit_entries) == 1
        assert 'OVERLAP' in result.audit_entries[0]
        assert 'topic_id=1' in result.audit_entries[0]
        assert 'topic_id=2' in result.audit_entries[0]
    
    def test_overlapping_node_3_also_goes_to_topic_1(self, tmp_path):
        """node_3 is also in both topics; should go to topic 1."""
        from episodic.recall.promotion import promote_hits_to_topics
        
        conn = create_overlap_db(tmp_path)
        hits = create_hits(['node_3'], [0.85])
        
        result = promote_hits_to_topics(conn, hits)
        
        assert 1 in result.by_topic
        assert result.by_topic[1][0].topic_id == 1
        assert len(result.audit_entries) == 1  # One overlap logged


class TestPromotionNoTopic:
    """Test that hits not in any topic get topic_id=None."""
    
    def test_promote_no_topic_returns_none(self, tmp_path):
        """Hit exchange_id not in any topic node_set → topic_id None."""
        from episodic.recall.promotion import promote_hits_to_topics, get_unassigned_hits
        
        conn = create_overlap_db(tmp_path)
        
        # node_5 is not in any topic (both topics end at node_3 and node_4)
        hits = create_hits(['node_5'], [0.8])
        
        result = promote_hits_to_topics(conn, hits)
        
        # Should be in None bucket
        assert None in result.by_topic
        assert len(result.by_topic[None]) == 1
        assert result.by_topic[None][0].topic_id is None
        
        # Helper function should return same
        unassigned = get_unassigned_hits(result)
        assert len(unassigned) == 1
        assert unassigned[0].exchange_id == 'node_5'
    
    def test_nonexistent_node_returns_none(self, tmp_path):
        """Hit with exchange_id not in nodes table → topic_id None."""
        from episodic.recall.promotion import promote_hits_to_topics
        
        conn = create_overlap_db(tmp_path)
        hits = create_hits(['nonexistent_node'], [0.7])
        
        result = promote_hits_to_topics(conn, hits)
        
        assert None in result.by_topic
        assert result.by_topic[None][0].topic_id is None


class TestPromotionStableOrdering:
    """Test that promotion is deterministic regardless of topic insertion order."""
    
    def test_promote_stable_under_topic_ordering(self, tmp_path):
        """Topics inserted in different order still use ORDER BY id ASC."""
        from episodic.recall.promotion import promote_hits_to_topics
        
        # Create DB with topics inserted in reverse order
        import sqlite3
        db_path = str(tmp_path / "test_reverse.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        conn.executescript("""
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                short_id TEXT,
                content TEXT NOT NULL,
                parent_id TEXT,
                role TEXT
            );
            CREATE TABLE topics (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                start_node_id TEXT NOT NULL,
                end_node_id TEXT
            );
            CREATE TABLE state (name TEXT PRIMARY KEY, head_id TEXT);
            CREATE INDEX idx_nodes_parent ON nodes(parent_id);
        """)
        
        # Insert nodes
        nodes = [
            ("node_1", "n1", "Content 1", None, "user"),
            ("node_2", "n2", "Content 2", "node_1", "assistant"),
            ("node_3", "n3", "Content 3", "node_2", "user"),
        ]
        conn.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?)", nodes)
        
        # Insert topics with explicit IDs in "wrong" order
        # Topic 2 inserted first but has higher id
        conn.execute("INSERT INTO topics (id, name, start_node_id, end_node_id) VALUES (2, 'topic-two', 'node_2', 'node_3')")
        conn.execute("INSERT INTO topics (id, name, start_node_id, end_node_id) VALUES (1, 'topic-one', 'node_1', 'node_2')")
        
        conn.execute("INSERT INTO state VALUES ('head', 'node_3')")
        conn.commit()
        
        # node_2 is in both topics
        hits = create_hits(['node_2'], [0.9])
        
        result = promote_hits_to_topics(conn, hits)
        
        # Should be assigned to topic 1 (lower id) despite insertion order
        assert 1 in result.by_topic
        assert result.by_topic[1][0].topic_id == 1


class TestPromotionMultipleHits:
    """Test promotion with multiple hits across topics."""
    
    def test_multiple_hits_grouped_correctly(self, tmp_path):
        """Multiple hits get grouped by their respective topics."""
        from episodic.recall.promotion import promote_hits_to_topics
        
        conn, _ = create_test_db(tmp_path)
        
        # Hits from different topics
        hits = create_hits(
            ['node_1', 'node_2', 'node_5', 'node_6', 'node_10'],
            [0.9, 0.85, 0.8, 0.75, 0.7]
        )
        
        result = promote_hits_to_topics(conn, hits)
        
        # Topic 1 (nodes 1-4): node_1, node_2
        assert 1 in result.by_topic
        assert len(result.by_topic[1]) == 2
        
        # Topic 2 (nodes 5-8): node_5, node_6
        assert 2 in result.by_topic
        assert len(result.by_topic[2]) == 2
        
        # Topic 3 (nodes 9-12, ongoing): node_10
        assert 3 in result.by_topic
        assert len(result.by_topic[3]) == 1
    
    def test_hits_sorted_by_similarity_within_topic(self, tmp_path):
        """Hits within a topic are sorted by similarity descending."""
        from episodic.recall.promotion import promote_hits_to_topics
        
        conn, _ = create_test_db(tmp_path)
        
        # Hits from topic 1, not in similarity order
        hits = create_hits(
            ['node_3', 'node_1', 'node_2'],
            [0.7, 0.9, 0.8]
        )
        
        result = promote_hits_to_topics(conn, hits)
        
        topic_hits = result.by_topic[1]
        similarities = [h.similarity for h in topic_hits]
        
        # Should be sorted descending
        assert similarities == sorted(similarities, reverse=True)
        assert similarities == [0.9, 0.8, 0.7]
