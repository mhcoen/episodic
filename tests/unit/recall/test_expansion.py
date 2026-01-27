"""
Tests for recall/expansion.py

Tests tier-based anchor expansion using real test DB fixtures.
"""

import pytest
from . import create_test_db, create_hits


class TestAnchorDiversity:
    """Test anchor diversity selection."""
    
    def test_anchor_diversity_skip_within_d(self, tmp_path):
        """
        Given ordered exchanges 1..N and hit anchors at positions 5,6,7;
        with d=1 (anchor_diversity_distance), only 5 and 7 selected (skip 6).
        """
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        
        # Hits at consecutive nodes in topic 2 (nodes 5-8)
        # Positions in topic: node_5=0, node_6=1, node_7=2, node_8=3
        hits = create_hits(['node_5', 'node_6', 'node_7'], [0.9, 0.85, 0.8])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        # Get topic 2
        topic_2 = next(t for t in ranking.ranked_topics if t.topic_id == 2)
        
        # Expand with diversity distance = 1, max 3 anchors
        config = ExpansionConfig(
            anchor_diversity_distance=1,
            tier_c_max_anchors=3,
            tier_c_window=0,  # No window, just anchors
            tier_c_max_exchanges=10
        )
        
        expansion = expand_topic(conn, topic_2, Tier.C, config)
        
        # Should select node_5 (pos 0, best score), skip node_6 (pos 1, within d=1 of 0)
        # Then select node_7 (pos 2, outside d=1 of 0)
        anchor_ids = [e.node_id for e in expansion.exchanges if e.is_anchor]
        
        assert 'node_5' in anchor_ids
        assert 'node_6' not in anchor_ids  # Skipped due to diversity
        assert 'node_7' in anchor_ids
    
    def test_anchor_diversity_all_too_close(self, tmp_path):
        """When all candidates are within d of first, only first selected."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        
        # Hits at adjacent positions with large diversity distance
        hits = create_hits(['node_5', 'node_6'], [0.9, 0.85])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        topic_2 = next(t for t in ranking.ranked_topics if t.topic_id == 2)
        
        config = ExpansionConfig(
            anchor_diversity_distance=5,  # Very large
            tier_c_max_anchors=3,
            tier_c_window=0,
            tier_c_max_exchanges=10
        )
        
        expansion = expand_topic(conn, topic_2, Tier.C, config)
        
        # Only node_5 should be anchor (node_6 within d=5)
        anchor_ids = [e.node_id for e in expansion.exchanges if e.is_anchor]
        assert anchor_ids == ['node_5']


class TestWindowClamping:
    """Test that windows are clamped to topic bounds."""
    
    def test_window_clamped_to_topic_start(self, tmp_path):
        """Anchor near segment start; window does not include outside nodes."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        
        # Hit at first node of topic 1
        hits = create_hits(['node_1'], [0.9])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        topic_1 = next(t for t in ranking.ranked_topics if t.topic_id == 1)
        
        config = ExpansionConfig(
            tier_b_max_anchors=1,
            tier_b_window=3,  # Would go before topic start
            tier_b_max_exchanges=10
        )
        
        expansion = expand_topic(conn, topic_1, Tier.B, config)
        
        # All exchanges should be from topic 1 (nodes 1-4)
        exchange_ids = [e.node_id for e in expansion.exchanges]
        
        for eid in exchange_ids:
            assert eid in ['node_1', 'node_2', 'node_3', 'node_4']
    
    def test_window_clamped_to_topic_end(self, tmp_path):
        """Anchor near segment end; window does not include outside nodes."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        
        # Hit at last node of topic 1
        hits = create_hits(['node_4'], [0.9])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        topic_1 = next(t for t in ranking.ranked_topics if t.topic_id == 1)
        
        config = ExpansionConfig(
            tier_b_max_anchors=1,
            tier_b_window=3,
            tier_b_max_exchanges=10
        )
        
        expansion = expand_topic(conn, topic_1, Tier.B, config)
        
        # All exchanges should be from topic 1 (nodes 1-4)
        exchange_ids = [e.node_id for e in expansion.exchanges]
        
        for eid in exchange_ids:
            assert eid in ['node_1', 'node_2', 'node_3', 'node_4']
        
        # node_5 (topic 2) should NOT be included
        assert 'node_5' not in exchange_ids


class TestMergeWindows:
    """Test window merging and deduplication."""
    
    def test_merge_windows_dedup_order(self, tmp_path):
        """Two overlapping windows; merged output has no duplicates and preserves order."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        
        # Hits at positions 1 and 3 in topic 2 (nodes 5-8)
        # Position 1 = node_6, position 3 = node_8
        # With window ±1, pos 1 covers {0,1,2}, pos 3 covers {2,3}
        # Overlap at position 2 (node_7)
        hits = create_hits(['node_6', 'node_8'], [0.9, 0.85])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        topic_2 = next(t for t in ranking.ranked_topics if t.topic_id == 2)
        
        config = ExpansionConfig(
            anchor_diversity_distance=0,  # Allow adjacent
            tier_c_max_anchors=2,
            tier_c_window=1,
            tier_c_max_exchanges=10
        )
        
        expansion = expand_topic(conn, topic_2, Tier.C, config)
        
        exchange_ids = [e.node_id for e in expansion.exchanges]
        
        # No duplicates
        assert len(exchange_ids) == len(set(exchange_ids))
        
        # Should cover most of topic 2
        assert 'node_5' in exchange_ids or 'node_6' in exchange_ids
        assert 'node_7' in exchange_ids  # Overlap node
        assert 'node_8' in exchange_ids
        
        # Chronological order (by position)
        positions = [e.position for e in expansion.exchanges]
        assert positions == sorted(positions)


class TestTierBehavior:
    """Test different expansion tiers."""
    
    def test_tier_a_anchors_only(self, tmp_path):
        """Tier A: minimal expansion, just anchors."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        hits = create_hits(['node_5', 'node_7'], [0.9, 0.8])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        topic_2 = next(t for t in ranking.ranked_topics if t.topic_id == 2)
        
        config = ExpansionConfig(
            tier_a_max_anchors=2,
            tier_a_window=0,  # No window
        )
        
        expansion = expand_topic(conn, topic_2, Tier.A, config)
        
        # Should only include anchor nodes
        exchange_ids = [e.node_id for e in expansion.exchanges]
        assert set(exchange_ids).issubset({'node_5', 'node_7'})
        
        # All should be marked as anchors
        assert all(e.is_anchor for e in expansion.exchanges)
    
    def test_tier_b_adds_window(self, tmp_path):
        """Tier B: anchors plus small window."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        hits = create_hits(['node_6'], [0.9])  # Middle of topic 2
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        topic_2 = next(t for t in ranking.ranked_topics if t.topic_id == 2)
        
        config = ExpansionConfig(
            tier_b_max_anchors=1,
            tier_b_window=1,
            tier_b_max_exchanges=6
        )
        
        expansion = expand_topic(conn, topic_2, Tier.B, config)
        
        exchange_ids = [e.node_id for e in expansion.exchanges]
        
        # Should include anchor plus neighbors
        assert 'node_6' in exchange_ids  # Anchor
        assert 'node_5' in exchange_ids or 'node_7' in exchange_ids  # Window
    
    def test_tier_c_larger_window(self, tmp_path):
        """Tier C: diverse anchors plus larger window."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        
        conn, _ = create_test_db(tmp_path)
        hits = create_hits(['node_1', 'node_3'], [0.9, 0.8])  # Topic 1
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        topic_1 = next(t for t in ranking.ranked_topics if t.topic_id == 1)
        
        config = ExpansionConfig(
            anchor_diversity_distance=1,
            tier_c_max_anchors=2,
            tier_c_window=2,
            tier_c_max_exchanges=12
        )
        
        expansion = expand_topic(conn, topic_1, Tier.C, config)
        
        # Should cover most/all of topic 1
        exchange_ids = set(e.node_id for e in expansion.exchanges)
        topic_1_nodes = {'node_1', 'node_2', 'node_3', 'node_4'}
        
        # At least anchors and some window
        assert len(exchange_ids.intersection(topic_1_nodes)) >= 2
