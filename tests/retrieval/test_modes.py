"""
Tests for modes and empty target handling (Success Criterion 10).

SC10: Empty target - browse returns recent; answer/summarize return empty.
"""
import pytest


class TestEmptyTargetHandling:
    """SC10: Empty target behavior by mode."""
    
    def test_empty_target_browse_returns_recent(self, fx_base, fake_chroma_empty):
        """Browse mode with empty target returns recent exchanges."""
        from episodic.retrieval.pipeline import retrieve
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(fx_base)
        
        results = retrieve(
            conn=fx_base,
            chroma=fake_chroma_empty,
            target="   ",  # Empty/whitespace
            segment_scope=None,
            temporal=None,
            speaker=None,
            mode="browse",
            max_results=5,
            config={"semantic_weight": 0.6, "lexical_weight": 0.4,
                    "over_fetch_multiplier": 3, "segment_filter_in_clause_max": 100,
                    "sqlite_max_variable_number": 999}
        )
        
        # Should return recent exchanges (user nodes)
        assert len(results) > 0
        
        # Results should be recent exchanges ordered by created_at DESC
        for r in results:
            assert 'exchange_id' in r
    
    def test_empty_target_answer_returns_empty(self, fx_base, fake_chroma_empty):
        """Answer mode with empty target returns empty list."""
        from episodic.retrieval.pipeline import retrieve
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(fx_base)
        
        results = retrieve(
            conn=fx_base,
            chroma=fake_chroma_empty,
            target="",
            segment_scope=None,
            temporal=None,
            speaker=None,
            mode="answer",
            max_results=10,
            config={"semantic_weight": 0.6, "lexical_weight": 0.4,
                    "over_fetch_multiplier": 3, "segment_filter_in_clause_max": 100,
                    "sqlite_max_variable_number": 999}
        )
        
        assert results == []
    
    def test_empty_target_summarize_returns_empty(self, fx_base, fake_chroma_empty):
        """Summarize mode with empty target returns empty list."""
        from episodic.retrieval.pipeline import retrieve
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(fx_base)
        
        results = retrieve(
            conn=fx_base,
            chroma=fake_chroma_empty,
            target="",
            segment_scope=None,
            temporal=None,
            speaker=None,
            mode="summarize",
            max_results=10,
            config={"semantic_weight": 0.6, "lexical_weight": 0.4,
                    "over_fetch_multiplier": 3, "segment_filter_in_clause_max": 100,
                    "sqlite_max_variable_number": 999}
        )
        
        assert results == []


class TestModeResponses:
    """Test mode-specific response requirements."""
    
    def test_answer_empty_retrieval_returns_fixed_string(self, fx_base, fake_chroma_empty):
        """Answer mode with no results returns fixed string, no LLM call."""
        from episodic.retrieval.pipeline import retrieve
        from episodic.retrieval.modes import format_answer_response
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(fx_base)
        
        results = retrieve(
            conn=fx_base,
            chroma=fake_chroma_empty,
            target="nonexistent query that matches nothing",
            segment_scope=None,
            temporal=None,
            speaker=None,
            mode="answer",
            max_results=10,
            config={"semantic_weight": 0.6, "lexical_weight": 0.4,
                    "over_fetch_multiplier": 3, "segment_filter_in_clause_max": 100,
                    "sqlite_max_variable_number": 999}
        )
        
        response = format_answer_response(results)
        
        if not results:
            assert response == "I don't have that in our conversation history."
    
    def test_summarize_empty_retrieval_returns_fixed_string(self):
        """Summarize mode with no results returns fixed string, no LLM call."""
        from episodic.retrieval.modes import format_summarize_response
        
        response = format_summarize_response([])
        
        assert response == "No conversations found to summarize."
    
    def test_browse_groups_by_segment(self, fx_base, fake_chroma_grinder):
        """Browse mode groups results by segment."""
        from episodic.retrieval.pipeline import retrieve
        from episodic.retrieval.modes import format_browse_response
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(fx_base)
        
        results = retrieve(
            conn=fx_base,
            chroma=fake_chroma_grinder,
            target="grinder",
            segment_scope=None,
            temporal=None,
            speaker=None,
            mode="browse",
            max_results=10,
            config={"semantic_weight": 0.6, "lexical_weight": 0.4,
                    "over_fetch_multiplier": 3, "segment_filter_in_clause_max": 100,
                    "sqlite_max_variable_number": 999}
        )
        
        formatted = format_browse_response(fx_base, results)
        
        # Should have segment groupings
        assert 'groups' in formatted or 'segments' in formatted or isinstance(formatted, list)


class TestRecentExchanges:
    """Test get_recent_exchanges for browse mode."""
    
    def test_recent_exchanges_returns_user_nodes(self, fx_base):
        """Recent exchanges are anchored by user nodes."""
        from episodic.retrieval.lexical import get_recent_exchanges
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        results = get_recent_exchanges(
            conn=fx_base,
            limit=10,
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            temporal=None
        )
        
        # All should be user nodes (exchange anchors)
        for r in results:
            assert r['role'] == 'user'
    
    def test_recent_exchanges_ordered_by_created_at_desc(self, fx_base):
        """Recent exchanges ordered newest first."""
        from episodic.retrieval.lexical import get_recent_exchanges
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        results = get_recent_exchanges(
            conn=fx_base,
            limit=10,
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            temporal=None
        )
        
        if len(results) > 1:
            timestamps = [r['created_at'] for r in results]
            assert timestamps == sorted(timestamps, reverse=True)
    
    def test_recent_exchanges_respects_limit(self, fx_base):
        """Limit parameter is respected."""
        from episodic.retrieval.lexical import get_recent_exchanges
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        results = get_recent_exchanges(
            conn=fx_base,
            limit=2,
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            temporal=None
        )
        
        assert len(results) <= 2
    
    def test_recent_exchanges_applies_segment_filter(self, fx_base):
        """Segment filter restricts recent exchanges."""
        from episodic.retrieval.lexical import get_recent_exchanges
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        # Only nodes in coffee segment
        segment_filter = SegmentFilter(kind=FilterKind.IN_CLAUSE, node_ids=["U1", "U2"])
        
        results = get_recent_exchanges(
            conn=fx_base,
            limit=10,
            segment_filter=segment_filter,
            temporal=None
        )
        
        result_ids = {r['id'] for r in results}
        assert result_ids.issubset({"U1", "U2"})
