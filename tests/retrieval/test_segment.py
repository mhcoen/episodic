"""
Tests for segment mechanics (Success Criteria 4, 5).

SC4: Segment scoping tri-state semantics
SC5: Ongoing segment cache invalidation
"""
import pytest
import sqlite3


class TestSegmentTriState:
    """SC4: None searches all; [] returns empty."""
    
    def test_segment_scope_none_searches_all(self, fx_base):
        """segment_scope=None means no restriction."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import build_segment_filter, FilterKind
        
        migrate_fts5(fx_base)
        
        # None = no segment scope requested
        segment_filter = build_segment_filter(None)
        assert segment_filter.kind == FilterKind.NONE
        
        results = execute_lexical_search(
            conn=fx_base,
            target="reply",  # Matches both segments
            segment_filter=segment_filter,
            speaker=None,
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # Should return results from all segments
        assert len(results) > 0
    
    def test_segment_scope_empty_returns_empty(self, fx_base):
        """segment_scope=[] means scope requested but failed."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import build_segment_filter, FilterKind
        
        migrate_fts5(fx_base)
        
        # [] = scope requested but resolver found nothing
        segment_filter = build_segment_filter([])
        assert segment_filter.kind == FilterKind.EMPTY
        
        results = execute_lexical_search(
            conn=fx_base,
            target="coffee",
            segment_filter=segment_filter,
            speaker=None,
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # Must return empty without executing query
        assert results == []
    
    def test_segment_scope_with_ids_filters(self, fx_base):
        """segment_scope=[ids] filters to those nodes."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import build_segment_filter
        
        migrate_fts5(fx_base)
        
        # Scope to coffee segment nodes only
        segment_filter = build_segment_filter(["U1", "A1", "U2", "A2"])
        
        results = execute_lexical_search(
            conn=fx_base,
            target="reply",
            segment_filter=segment_filter,
            speaker=None,
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # Should only return nodes from coffee segment
        result_ids = {r['id'] for r in results}
        assert result_ids.issubset({"U1", "A1", "U2", "A2"})
    
    def test_segment_filter_deduplicates_ids(self):
        """Building filter deduplicates with stable order."""
        from episodic.retrieval.segment_filter import build_segment_filter, FilterKind
        
        segment_filter = build_segment_filter(["A", "B", "A", "C", "B"])
        
        assert segment_filter.kind == FilterKind.PENDING_IDS
        assert segment_filter.node_ids == ["A", "B", "C"]


class TestSegmentCache:
    """SC5: Ongoing segment cache invalidation."""
    
    def test_segment_cache_returns_correct_nodes(self, fx_base):
        """Cache returns ordered list and set of segment nodes."""
        from episodic.retrieval.segment import get_cached_segment_nodes
        
        # Topic 1: coffee (closed, end_node_id = U2)
        nodes_list, nodes_set = get_cached_segment_nodes(fx_base, segment_id=1)
        
        # Should include nodes from U1 to U2 on ancestry
        assert "U1" in nodes_set
        assert "U2" in nodes_set
        assert nodes_list[0] == "U1"  # Start
        assert nodes_list[-1] in ["U2", "A2"]  # End area
    
    def test_ongoing_segment_uses_head(self, fx_base):
        """Ongoing segment (end_node_id=NULL) uses head as effective_end."""
        from episodic.retrieval.segment import get_cached_segment_nodes
        
        # Topic 2: legal (ongoing, end_node_id = NULL)
        nodes_list, nodes_set = get_cached_segment_nodes(fx_base, segment_id=2)
        
        # Should include U3, A3 (current head ancestry from start)
        assert "U3" in nodes_set
        assert "A3" in nodes_set
    
    def test_cache_invalidated_on_head_change(self, fx_base):
        """Cache must invalidate when head changes for ongoing segment."""
        from episodic.retrieval.segment import get_cached_segment_nodes, _segment_cache
        
        # Clear cache
        _segment_cache.clear()
        
        # First call - caches with current head (A3)
        nodes_list_1, _ = get_cached_segment_nodes(fx_base, segment_id=2)
        initial_count = len(nodes_list_1)
        
        # Add new nodes to extend the ongoing segment
        cursor = fx_base.cursor()
        cursor.execute("""
            INSERT INTO nodes (id, content, parent_id, role, created_at) 
            VALUES ('U4', 'new user msg', 'A3', 'user', '2026-01-04T00:00:00.000000Z')
        """)
        cursor.execute("""
            INSERT INTO nodes (id, content, parent_id, role, created_at) 
            VALUES ('A4', 'new assistant msg', 'U4', 'assistant', '2026-01-04T00:00:01.000000Z')
        """)
        cursor.execute("UPDATE state SET head_id = 'A4' WHERE name = 'head'")
        fx_base.commit()
        
        # Second call - should detect head change and recompute
        nodes_list_2, nodes_set_2 = get_cached_segment_nodes(fx_base, segment_id=2)
        
        # New nodes should be included
        assert "U4" in nodes_set_2
        assert "A4" in nodes_set_2
        assert len(nodes_list_2) > initial_count
    
    def test_closed_segment_cache_stable(self, fx_base):
        """Closed segment cache doesn't change with head."""
        from episodic.retrieval.segment import get_cached_segment_nodes, _segment_cache
        
        _segment_cache.clear()
        
        # Topic 1 is closed (end_node_id = U2)
        nodes_list_1, nodes_set_1 = get_cached_segment_nodes(fx_base, segment_id=1)
        
        # Change head
        cursor = fx_base.cursor()
        cursor.execute("""
            INSERT INTO nodes (id, content, parent_id, role, created_at) 
            VALUES ('U99', 'extra', 'A3', 'user', '2026-01-05T00:00:00.000000Z')
        """)
        cursor.execute("UPDATE state SET head_id = 'U99' WHERE name = 'head'")
        fx_base.commit()
        
        # Closed segment should return same nodes
        nodes_list_2, nodes_set_2 = get_cached_segment_nodes(fx_base, segment_id=1)
        
        assert nodes_set_1 == nodes_set_2


class TestAncestryMap:
    """Test build_ancestry_map functionality."""
    
    def test_ancestry_map_single_query(self, fx_base):
        """Ancestry map built in single recursive CTE."""
        from episodic.retrieval.segment import build_ancestry_map
        
        # Build from head (A3)
        ancestry = build_ancestry_map(fx_base, "A3")
        
        # Should have all ancestors
        assert "A3" in ancestry
        assert "U3" in ancestry
        assert "A2" in ancestry
        assert "U2" in ancestry
        assert "A1" in ancestry
        assert "U1" in ancestry
        assert "S0" in ancestry
        
        # Parent relationships correct
        assert ancestry["A3"] == "U3"
        assert ancestry["U3"] == "A2"
        assert ancestry["S0"] is None  # Root
    
    def test_ancestry_map_excludes_branches(self, fx_base):
        """Ancestry map only includes direct ancestors, not siblings."""
        from episodic.retrieval.segment import build_ancestry_map
        
        ancestry = build_ancestry_map(fx_base, "A3")
        
        # A2b is a sibling of A2, not an ancestor of A3
        assert "A2b" not in ancestry
