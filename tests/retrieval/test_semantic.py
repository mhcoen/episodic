"""
Tests for semantic retrieval (Success Criteria 6, 7, 8).

SC6: Speaker scope routing (semantic disabled)
SC7: Display consistency with assistant_id
SC8: Temporal filtering half-open boundaries
"""
import pytest
from datetime import datetime, timezone


class TestSpeakerRouting:
    """SC6: Speaker scope disables semantic."""
    
    def test_speaker_scope_disables_semantic(self, fx_base, fake_chroma_grinder):
        """When speaker scope set, semantic retrieval is skipped."""
        from episodic.retrieval.pipeline import retrieve
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(fx_base)
        
        results = retrieve(
            conn=fx_base,
            chroma=fake_chroma_grinder,
            target="grinder",
            segment_scope=None,
            temporal=None,
            speaker="user",  # Speaker scope set
            mode="browse",
            max_results=10,
            config={"semantic_weight": 0.6, "lexical_weight": 0.4,
                    "over_fetch_multiplier": 3, "segment_filter_in_clause_max": 100,
                    "sqlite_max_variable_number": 999}
        )
        
        # Chroma should NOT have been queried
        assert fake_chroma_grinder.query_count == 0
    
    def test_speaker_scope_filters_lexical(self, fx_base):
        """Speaker scope filters lexical results to that role."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        migrate_fts5(fx_base)
        
        results = execute_lexical_search(
            conn=fx_base,
            target="reply",  # Matches assistant messages
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            speaker="assistant",
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # All results should be assistant role
        for r in results:
            assert r['role'] == 'assistant'
    
    def test_browse_shows_full_exchange_with_speaker_scope(self, fx_base, fake_chroma_empty):
        """Browse mode shows full exchange even when speaker scoped."""
        from episodic.retrieval.pipeline import retrieve
        from episodic.retrieval.display import get_exchange_for_display
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(fx_base)
        
        results = retrieve(
            conn=fx_base,
            chroma=fake_chroma_empty,
            target="coffee",
            segment_scope=None,
            temporal=None,
            speaker="user",
            mode="browse",
            max_results=10,
            config={"semantic_weight": 0.6, "lexical_weight": 0.4,
                    "over_fetch_multiplier": 3, "segment_filter_in_clause_max": 100,
                    "sqlite_max_variable_number": 999}
        )
        
        # Even with speaker="user", display should show both turns
        for r in results:
            exchange = get_exchange_for_display(fx_base, r['exchange_id'], metadata=None)
            assert 'user_content' in exchange
            assert 'assistant_content' in exchange or 'assistant_id' in exchange


class TestDisplayConsistency:
    """SC7: Display uses metadata.assistant_id when valid."""
    
    def test_display_uses_metadata_assistant_id_when_valid(self, fx_base):
        """Valid metadata.assistant_id is used even if off-ancestry."""
        from episodic.retrieval.display import get_exchange_for_display
        
        # U2's semantic result has assistant_id = A2b (off-ancestry but valid)
        metadata = {"assistant_id": "A2b"}
        
        exchange = get_exchange_for_display(fx_base, "U2", metadata)
        
        # Should use A2b, not A2 (which is on ancestry)
        assert exchange['assistant_id'] == "A2b"
    
    def test_display_falls_back_on_invalid_assistant_id(self, fx_base, audit_capture):
        """Invalid assistant_id triggers fallback and AUDIT log."""
        from episodic.retrieval.display import get_exchange_for_display
        from unittest.mock import patch
        
        # A1 is not a child of U2, so invalid for exchange U2
        metadata = {"assistant_id": "A1"}
        
        with patch('episodic.retrieval.display.logger', audit_capture):
            exchange = get_exchange_for_display(fx_base, "U2", metadata)
        
        # Should fall back to A2 (valid child on ancestry)
        assert exchange['assistant_id'] == "A2"
        
        # Should log AUDIT
        assert audit_capture.contains("Invalid assistant_id") or audit_capture.contains("AUDIT")
    
    def test_display_falls_back_when_no_metadata(self, fx_base):
        """No metadata triggers fallback selection."""
        from episodic.retrieval.display import get_exchange_for_display
        
        exchange = get_exchange_for_display(fx_base, "U2", metadata=None)
        
        # Should select A2 (on ancestry, earlier than A2b)
        assert exchange['assistant_id'] == "A2"
    
    def test_fallback_prefers_ancestry_assistant(self, fx_base):
        """Fallback prefers assistant on current head ancestry."""
        from episodic.retrieval.display import get_exchange_for_display
        
        # U2 has two children: A2 (on ancestry) and A2b (off ancestry)
        # A2 created_at < A2b created_at, but A2 is on ancestry
        
        exchange = get_exchange_for_display(fx_base, "U2", metadata=None)
        
        # A2 is on ancestry (path to head A3), so preferred
        assert exchange['assistant_id'] == "A2"


class TestTemporalFiltering:
    """SC8: Half-open temporal boundaries."""
    
    def test_temporal_half_open_includes_start(self, fx_base):
        """Temporal filter includes start boundary."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        migrate_fts5(fx_base)
        
        # U1 created_at is exactly 2026-01-02T10:00:00.000000Z
        temporal = ("2026-01-02T10:00:00.000000Z", "2026-01-02T12:00:00.000000Z")
        
        results = execute_lexical_search(
            conn=fx_base,
            target="coffee",
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            speaker=None,
            temporal=temporal,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # U1 should be included (at start boundary)
        result_ids = {r['id'] for r in results}
        assert "U1" in result_ids
    
    def test_temporal_half_open_excludes_end(self, fx_base):
        """Temporal filter excludes end boundary."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        migrate_fts5(fx_base)
        
        # End exactly at U3's timestamp
        temporal = ("2026-01-02T00:00:00.000000Z", "2026-01-03T09:00:00.000000Z")
        
        results = execute_lexical_search(
            conn=fx_base,
            target="topic",  # Matches U3's content
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            speaker=None,
            temporal=temporal,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # U3 should NOT be included (at end boundary)
        result_ids = {r['id'] for r in results}
        assert "U3" not in result_ids
    
    def test_semantic_temporal_drops_missing_timestamp(self, audit_capture):
        """Semantic candidate without timestamp is dropped when temporal active."""
        from episodic.retrieval.semantic import filter_semantic_by_temporal
        from unittest.mock import patch
        from datetime import datetime, timezone
        
        candidates = [
            {"exchange_id": "U1", "distance": 0.1, "metadata": {}},  # No timestamp
            {"exchange_id": "U2", "distance": 0.2, "metadata": {"timestamp": "2026-01-02T11:00:00.000000Z"}},
        ]
        
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 5, tzinfo=timezone.utc)
        
        with patch('episodic.retrieval.semantic.logger', audit_capture):
            filtered = filter_semantic_by_temporal(candidates, start, end)
        
        # U1 should be dropped
        assert len(filtered) == 1
        assert filtered[0]['exchange_id'] == "U2"
        
        # AUDIT log for missing timestamp
        assert audit_capture.contains("missing timestamp") or audit_capture.contains("AUDIT")
    
    def test_semantic_temporal_drops_unparseable_timestamp(self, audit_capture):
        """Unparseable timestamp is dropped with AUDIT."""
        from episodic.retrieval.semantic import filter_semantic_by_temporal
        from unittest.mock import patch
        from datetime import datetime, timezone
        
        candidates = [
            {"exchange_id": "U1", "distance": 0.1, "metadata": {"timestamp": "not-a-date"}},
        ]
        
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 5, tzinfo=timezone.utc)
        
        with patch('episodic.retrieval.semantic.logger', audit_capture):
            filtered = filter_semantic_by_temporal(candidates, start, end)
        
        assert len(filtered) == 0
        assert audit_capture.contains("unparseable") or audit_capture.contains("AUDIT")
