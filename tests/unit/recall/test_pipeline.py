"""
Tests for recall/pipeline.py

Integration tests with golden fixture for determinism.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from . import create_test_db, create_hits, FakeChroma

from episodic.recall.pipeline import SemanticHit


def make_semantic_hit(exchange_id, relevance_score, metadata, text=""):
    """Helper to create SemanticHit objects for testing."""
    return SemanticHit(
        exchange_id=exchange_id,
        relevance_score=relevance_score,
        metadata=metadata,
        text=text,
        embedding=None,  # No embeddings needed for these tests
    )


# Golden fixture: expected output for deterministic input
GOLDEN_QUERY_TARGET = "database indexing"
GOLDEN_HITS = [
    # Topic 1 hits (nodes 1-4)
    make_semantic_hit('node_1', 0.85, {'user_id': 'node_1', 'timestamp': '2026-01-15T10:01:00Z'}),
    make_semantic_hit('node_2', 0.80, {'user_id': 'node_2', 'timestamp': '2026-01-15T10:02:00Z'}),
    # Topic 2 hits (nodes 5-8)
    make_semantic_hit('node_5', 0.75, {'user_id': 'node_5', 'timestamp': '2026-01-16T10:05:00Z'}),
    # Unassigned hit
    make_semantic_hit('node_99', 0.60, {'user_id': 'node_99', 'timestamp': '2026-01-17T10:00:00Z'}),
]

# Expected deterministic output structure
GOLDEN_EXPECTED = {
    'budget': {
        'intent': 'CONVERSATION_RECALL_LOCATE',
        'max_topics': 2,
        'max_statements': 1,
        'topic_tier': 'B',
        'broad_horizon': False,
    },
    'promotion': {
        'topic_1_hit_count': 2,  # node_1, node_2
        'topic_2_hit_count': 1,  # node_5
        'unassigned_count': 1,   # node_99
        'audit_count': 0,        # No overlaps
    },
    'ranking': {
        'top_topic_id': 1,       # Topic 1 has more hits, higher concentration
        'topic_count': 2,        # Topics 1 and 2 have hits
    },
    'formatted': {
        'conversation_block_count': 2,
        'statement_block_count': 0,  # node_99 doesn't exist in test DB, so no statement block
        'first_topic_name': 'topic-alpha',  # Topic 1's name
    },
}


def make_resolved_query(target, temporal=None, speaker=None, has_broadness_cue=False, mode="answer"):
    """Create a ResolvedQuery with correct field signature."""
    from episodic.query.types import ResolvedQuery
    
    return ResolvedQuery(
        mode=mode,
        target=target,
        segment_explicit=False,
        segment_query=None,
        segment_resolved_ids=None,
        segment_ambiguous=False,
        segment_candidates=None,
        temporal=temporal,
        speaker=speaker,
        deictic=None,
        has_broadness_cue=has_broadness_cue,
        audit_trace="{}",
    )


class TestEndToEndDeterminism:
    """Test that pipeline produces deterministic results."""
    
    def test_end_to_end_determinism_golden(self, tmp_path):
        """
        Frozen fixture: input query + fixed hits + fixed topics + fixed nodes
        → exact serialized RecallResult matches golden structure.
        """
        from episodic.recall.pipeline import recall, _get_semantic_hits
        
        conn, _ = create_test_db(tmp_path)
        
        # Create resolved query with correct signature
        query = make_resolved_query(target="database indexing")
        
        # Mock semantic search to return fixed hits
        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return list(GOLDEN_HITS)
        
        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")
        
        # Verify budget
        assert result.budget.intent.name == GOLDEN_EXPECTED['budget']['intent']
        assert result.budget.max_topics == GOLDEN_EXPECTED['budget']['max_topics']
        assert result.budget.max_statements == GOLDEN_EXPECTED['budget']['max_statements']
        assert result.budget.topic_tier.name == GOLDEN_EXPECTED['budget']['topic_tier']
        assert result.budget.broad_horizon == GOLDEN_EXPECTED['budget']['broad_horizon']
        
        # Verify promotion
        assert len(result.promotion.by_topic.get(1, [])) == GOLDEN_EXPECTED['promotion']['topic_1_hit_count']
        assert len(result.promotion.by_topic.get(2, [])) == GOLDEN_EXPECTED['promotion']['topic_2_hit_count']
        assert len(result.promotion.by_topic.get(None, [])) == GOLDEN_EXPECTED['promotion']['unassigned_count']
        assert len(result.promotion.audit_entries) == GOLDEN_EXPECTED['promotion']['audit_count']
        
        # Verify ranking
        assert result.ranking.ranked_topics[0].topic_id == GOLDEN_EXPECTED['ranking']['top_topic_id']
        assert len(result.ranking.ranked_topics) == GOLDEN_EXPECTED['ranking']['topic_count']
        
        # Verify formatted output
        assert len(result.formatted.conversation_blocks) == GOLDEN_EXPECTED['formatted']['conversation_block_count']
        assert len(result.formatted.statement_blocks) == GOLDEN_EXPECTED['formatted']['statement_block_count']
        assert result.formatted.conversation_blocks[0].topic_name == GOLDEN_EXPECTED['formatted']['first_topic_name']
    
    def test_determinism_across_runs(self, tmp_path):
        """Same input produces identical output across multiple runs."""
        from episodic.recall.pipeline import recall
        
        conn, _ = create_test_db(tmp_path)
        
        query = make_resolved_query(target="testing")
        
        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return list(GOLDEN_HITS)

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            results = [recall(conn, query, query_form="when_we") for _ in range(3)]
        
        # All results should be identical
        for i in range(1, 3):
            assert results[i].ranking.ranked_topics[0].topic_id == results[0].ranking.ranked_topics[0].topic_id
            assert len(results[i].formatted.conversation_blocks) == len(results[0].formatted.conversation_blocks)
            assert len(results[i].formatted.statement_blocks) == len(results[0].formatted.statement_blocks)
            
            # Context strings should be byte-identical
            assert results[i].to_context_string() == results[0].to_context_string()


class TestPipelineEmptyResults:
    """Test pipeline behavior with no hits."""
    
    def test_empty_hits_returns_empty_result(self, tmp_path):
        """No semantic hits produces empty result."""
        from episodic.recall.pipeline import recall
        
        conn, _ = create_test_db(tmp_path)
        
        query = make_resolved_query(target="xyzzy_nonexistent_topic")
        
        def mock_empty_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return []

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_empty_hits):
            result = recall(conn, query, query_form="when_we")
        
        assert result.is_empty()
        assert result.to_context_string() == ""
    
    def test_no_target_returns_empty_result(self, tmp_path):
        """Query with no target produces empty result."""
        from episodic.recall.pipeline import recall
        
        conn, _ = create_test_db(tmp_path)
        
        query = make_resolved_query(target=None)
        
        result = recall(conn, query, query_form="when_we")
        
        assert result.is_empty()


class TestPipelineBudgetApplication:
    """Test that budget correctly limits output."""
    
    def test_max_topics_respected(self, tmp_path):
        """Number of conversation blocks <= max_topics."""
        from episodic.recall.pipeline import recall
        
        conn, _ = create_test_db(tmp_path)
        
        # Hits spread across all 3 topics
        many_hits = [
            make_semantic_hit('node_1', 0.9, {'user_id': 'node_1'}),
            make_semantic_hit('node_5', 0.8, {'user_id': 'node_5'}),
            make_semantic_hit('node_9', 0.7, {'user_id': 'node_9'}),
        ]

        query = make_resolved_query(target="test")

        def mock_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return list(many_hits)

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_hits):
            # when_we has max_topics=2
            result = recall(conn, query, query_form="when_we")
        
        assert len(result.formatted.conversation_blocks) <= result.budget.max_topics
        assert len(result.formatted.conversation_blocks) == 2  # Exactly 2 for when_we
    
    def test_existence_check_minimal_output(self, tmp_path):
        """have_we produces minimal output (Tier A)."""
        from episodic.recall.pipeline import recall
        
        conn, _ = create_test_db(tmp_path)
        
        hits = [
            make_semantic_hit('node_1', 0.9, {'user_id': 'node_1'}),
        ]

        query = make_resolved_query(target="this")

        def mock_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return list(hits)

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_hits):
            result = recall(conn, query, query_form="have_we")
        
        # have_we has max_topics=1
        assert len(result.formatted.conversation_blocks) <= 1
        # Tier A means minimal expansion
        assert result.budget.topic_tier.name == 'A'


class TestPipelineTierEscalation:
    """Test automatic tier escalation based on evidence."""
    
    def test_escalate_to_tier_c_with_strong_evidence(self, tmp_path):
        """Strong evidence (3+ hits) escalates from Tier B to Tier C."""
        from episodic.recall.pipeline import _select_tier
        from episodic.recall.ranking import RankedTopic
        from episodic.recall.budget import map_parser_output_to_budget
        from episodic.recall.expansion import Tier
        
        # Create a ranked topic with 3+ hits
        class MockHit:
            similarity = 0.8
        
        ranked_topic = RankedTopic(
            topic_id=1,
            score=2.5,
            best_hit=0.9,
            top_k_mass=2.4,
            hit_count=4,  # Strong evidence
            hits=[MockHit() for _ in range(4)],
            topic_info={'name': 'test'}
        )
        
        budget = map_parser_output_to_budget("when_we", False, None)
        assert budget.topic_tier == Tier.B
        
        selected_tier = _select_tier(ranked_topic, budget)
        
        # Should escalate to Tier C
        assert selected_tier == Tier.C
    
    def test_no_escalation_with_weak_evidence(self, tmp_path):
        """Weak evidence (< 3 hits) stays at budget tier."""
        from episodic.recall.pipeline import _select_tier
        from episodic.recall.ranking import RankedTopic
        from episodic.recall.budget import map_parser_output_to_budget
        from episodic.recall.expansion import Tier
        
        class MockHit:
            similarity = 0.8
        
        ranked_topic = RankedTopic(
            topic_id=1,
            score=1.5,
            best_hit=0.9,
            top_k_mass=1.6,
            hit_count=2,  # Weak evidence
            hits=[MockHit() for _ in range(2)],
            topic_info={'name': 'test'}
        )
        
        budget = map_parser_output_to_budget("when_we", False, None)
        selected_tier = _select_tier(ranked_topic, budget)
        
        # Should stay at Tier B
        assert selected_tier == Tier.B
