"""
Tests for recall/ranking.py

Tests topic ranking by concentration.
"""

import math
import pytest
from . import create_test_db, create_hits


class TestRankConcentration:
    """Test that concentrated hits outrank single high hits."""
    
    def test_rank_concentration_beats_single_high_hit(self, tmp_path):
        """
        Topic A: 3 hits (0.8, 0.75, 0.7) - concentrated
        Topic B: 1 hit (0.9) - single high
        
        With default weights (w_best=0.5, w_mass=0.3, w_count=0.2):
        Topic A: 0.5*0.8 + 0.3*(0.8+0.75+0.7) + 0.2*log(4) = 0.4 + 0.675 + 0.277 = 1.352
        Topic B: 0.5*0.9 + 0.3*0.9 + 0.2*log(2) = 0.45 + 0.27 + 0.139 = 0.859
        
        Topic A should outrank Topic B.
        """
        from episodic.recall.promotion import promote_hits_to_topics, PromotionResult, PromotedHit
        from episodic.recall.ranking import rank_topics
        
        conn, _ = create_test_db(tmp_path)
        
        # Create hits - topic 1 gets 3 hits, topic 2 gets 1 hit
        hits = create_hits(
            ['node_1', 'node_2', 'node_3', 'node_5'],  # topic 1: nodes 1-3, topic 2: node 5
            [0.8, 0.75, 0.7, 0.9]
        )
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        # Topic 1 should be ranked first
        assert len(ranking.ranked_topics) >= 2
        assert ranking.ranked_topics[0].topic_id == 1
        assert ranking.ranked_topics[1].topic_id == 2
        
        # Verify scores match expected calculation
        topic_a = ranking.ranked_topics[0]
        topic_b = ranking.ranked_topics[1]
        
        assert topic_a.hit_count == 3
        assert topic_b.hit_count == 1
        assert topic_a.score > topic_b.score
    
    def test_single_hit_with_very_high_score_can_win(self, tmp_path):
        """Edge case: single hit with very high score can beat weak concentration."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        
        conn, _ = create_test_db(tmp_path)
        
        # Topic 1: 2 weak hits, Topic 2: 1 very strong hit
        hits = create_hits(
            ['node_1', 'node_2', 'node_5'],
            [0.3, 0.25, 0.99]
        )
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        # With very weak topic 1 hits, topic 2's single strong hit should win
        # Topic 1: 0.5*0.3 + 0.3*(0.3+0.25) + 0.2*log(3) = 0.15 + 0.165 + 0.22 = 0.535
        # Topic 2: 0.5*0.99 + 0.3*0.99 + 0.2*log(2) = 0.495 + 0.297 + 0.139 = 0.931
        assert ranking.ranked_topics[0].topic_id == 2


class TestRankDeterministicTiebreak:
    """Test deterministic tiebreaking when scores are equal."""
    
    def test_rank_deterministic_tiebreak(self, tmp_path):
        """Two topics with equal scores; sort falls back to topic_id ascending."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        
        conn, _ = create_test_db(tmp_path)
        
        # Give both topics identical hits to produce identical scores
        hits = create_hits(
            ['node_1', 'node_5'],  # One hit each in topic 1 and topic 2
            [0.8, 0.8]  # Identical similarities
        )
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        # Both should have identical scores
        assert len(ranking.ranked_topics) == 2
        topic_1_score = ranking.ranked_topics[0].score if ranking.ranked_topics[0].topic_id == 1 else ranking.ranked_topics[1].score
        topic_2_score = ranking.ranked_topics[0].score if ranking.ranked_topics[0].topic_id == 2 else ranking.ranked_topics[1].score
        assert topic_1_score == topic_2_score
        
        # Tiebreak by topic_id ascending, so topic 1 should be first
        assert ranking.ranked_topics[0].topic_id == 1
        assert ranking.ranked_topics[1].topic_id == 2
    
    def test_tiebreak_is_stable_across_runs(self, tmp_path):
        """Same input produces same ranking order every time."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        
        conn, _ = create_test_db(tmp_path)
        hits = create_hits(['node_1', 'node_5'], [0.8, 0.8])
        
        # Run multiple times
        results = []
        for _ in range(5):
            promotion = promote_hits_to_topics(conn, hits)
            ranking = rank_topics(promotion)
            results.append([t.topic_id for t in ranking.ranked_topics])
        
        # All results should be identical
        assert all(r == results[0] for r in results)


class TestRankUnassignedHits:
    """Test handling of unassigned (statement-only) hits."""
    
    def test_unassigned_hits_in_ranking_result(self, tmp_path):
        """Hits not in any topic appear in unassigned_hits."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics
        
        conn, _ = create_test_db(tmp_path)
        
        # node_99 doesn't exist in any topic
        hits = create_hits(['node_1', 'nonexistent'], [0.9, 0.7])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        # Should have unassigned hit
        assert len(ranking.unassigned_hits) == 1
        assert ranking.unassigned_hits[0].exchange_id == 'nonexistent'


class TestGetTopTopicsAndStatements:
    """Test helper functions for getting top topics and statements."""
    
    def test_get_top_topics(self, tmp_path):
        """get_top_topics returns correct number of topics."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics, get_top_topics
        
        conn, _ = create_test_db(tmp_path)
        hits = create_hits(['node_1', 'node_5', 'node_9'], [0.9, 0.8, 0.7])
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        top_1 = get_top_topics(ranking, 1)
        top_2 = get_top_topics(ranking, 2)
        
        assert len(top_1) == 1
        assert len(top_2) == 2
    
    def test_get_top_statements_excludes_top_topics(self, tmp_path):
        """get_top_statements excludes hits from specified topics."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics, get_top_topics, get_top_statements
        
        conn, _ = create_test_db(tmp_path)
        hits = create_hits(
            ['node_1', 'node_2', 'node_5', 'node_6'],
            [0.9, 0.85, 0.8, 0.75]
        )
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        # Exclude topic 1
        top_topics = get_top_topics(ranking, 1)
        statements = get_top_statements(ranking, n=10, exclude_topic_ids=[1])
        
        # Statements should only include hits from topic 2 (not topic 1)
        statement_ids = [s.exchange_id for s in statements]
        assert 'node_1' not in statement_ids
        assert 'node_2' not in statement_ids
        assert 'node_5' in statement_ids
        assert 'node_6' in statement_ids
