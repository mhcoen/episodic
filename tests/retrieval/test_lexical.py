"""
Tests for lexical retrieval (Success Criteria 2, 3).

SC2: Connection passing + temp table cleanup
SC3: BM25 orientation correct
"""
import pytest
import sqlite3
from unittest.mock import patch


class TestConnectionPassing:
    """SC2: No internal get_connection(); temp tables cleaned up."""
    
    def test_lexical_search_uses_provided_connection(self, fx_base):
        """Lexical search must use the provided connection, not get_connection()."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        import episodic.retrieval.lexical as lexical_module

        migrate_fts5(fx_base)

        # Verify module doesn't import get_connection
        assert not hasattr(lexical_module, 'get_connection'), \
            "lexical module should not import get_connection"

        # Execute search - if it worked, it used the provided conn
        results = execute_lexical_search(
            conn=fx_base,
            target="coffee",
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            speaker=None,
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
    
    def test_temp_table_created_for_large_segment(self, fx_base):
        """Large segment filter uses temp table."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        migrate_fts5(fx_base)
        
        # Create segment filter that exceeds IN clause limit
        large_ids = [f"node_{i}" for i in range(200)]
        segment_filter = SegmentFilter(kind=FilterKind.PENDING_IDS, node_ids=large_ids)
        
        config = {"segment_filter_in_clause_max": 10, "sqlite_max_variable_number": 50}
        
        # Execute - should use temp table internally
        results = execute_lexical_search(
            conn=fx_base,
            target="coffee",
            segment_filter=segment_filter,
            speaker=None,
            temporal=None,
            limit=10,
            config=config
        )
        
        # Temp table should be cleaned up
        cursor = fx_base.cursor()
        cursor.execute("SELECT name FROM sqlite_temp_master WHERE name LIKE 'seg_filter_%'")
        assert cursor.fetchone() is None, "Temp table not cleaned up"
    
    def test_temp_table_cleaned_on_exception(self, fx_base):
        """Temp table is cleaned up even when query fails."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        migrate_fts5(fx_base)
        
        large_ids = [f"node_{i}" for i in range(200)]
        segment_filter = SegmentFilter(kind=FilterKind.PENDING_IDS, node_ids=large_ids)
        
        config = {"segment_filter_in_clause_max": 10, "sqlite_max_variable_number": 50}
        
        # Force an error by using invalid FTS syntax
        try:
            execute_lexical_search(
                conn=fx_base,
                target="INVALID:SYNTAX:",  # May cause FTS error
                segment_filter=segment_filter,
                speaker=None,
                temporal=None,
                limit=10,
                config=config
            )
        except Exception:
            pass  # Expected to fail
        
        # Temp table should still be cleaned up
        cursor = fx_base.cursor()
        cursor.execute("SELECT name FROM sqlite_temp_master WHERE name LIKE 'seg_filter_%'")
        assert cursor.fetchone() is None, "Temp table not cleaned up after exception"


class TestBM25Orientation:
    """SC3: BM25 negation and ordering correct."""
    
    def test_bm25_negated_higher_is_better(self, fx_base):
        """After negation, higher bm25_score = more relevant."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        migrate_fts5(fx_base)
        
        results = execute_lexical_search(
            conn=fx_base,
            target="coffee",
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            speaker=None,
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # Results should have bm25_score and be ordered DESC
        if len(results) > 1:
            scores = [r['bm25_score'] for r in results]
            assert scores == sorted(scores, reverse=True), "Results not sorted by bm25_score DESC"
    
    def test_more_relevant_doc_ranks_higher(self, retrieval_conn):
        """Document with more term occurrences ranks higher."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        from tests.retrieval.conftest import create_test_schema, set_head
        
        create_test_schema(retrieval_conn)
        
        cursor = retrieval_conn.cursor()
        # Node with single occurrence
        cursor.execute("""
            INSERT INTO nodes (id, content, role, created_at) 
            VALUES ('n1', 'I like coffee', 'user', '2026-01-01T00:00:00.000000Z')
        """)
        # Node with multiple occurrences - should rank higher
        cursor.execute("""
            INSERT INTO nodes (id, content, role, created_at) 
            VALUES ('n2', 'coffee coffee coffee beans', 'user', '2026-01-01T00:00:01.000000Z')
        """)
        set_head(retrieval_conn, 'n2')
        retrieval_conn.commit()
        
        migrate_fts5(retrieval_conn)
        
        results = execute_lexical_search(
            conn=retrieval_conn,
            target="coffee",
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            speaker=None,
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        assert len(results) == 2
        # More occurrences should rank first
        assert results[0]['id'] == 'n2', "More relevant doc should rank first"
    
    def test_lexical_returns_parent_role(self, fx_base):
        """Lexical results include parent_role for exchange mapping."""
        from episodic.retrieval.lexical import execute_lexical_search
        from episodic.retrieval.migration import migrate_fts5
        from episodic.retrieval.segment_filter import SegmentFilter, FilterKind
        
        migrate_fts5(fx_base)
        
        results = execute_lexical_search(
            conn=fx_base,
            target="grinder",
            segment_filter=SegmentFilter(kind=FilterKind.NONE),
            speaker=None,
            temporal=None,
            limit=10,
            config={"segment_filter_in_clause_max": 100, "sqlite_max_variable_number": 999}
        )
        
        # Results should have parent_role
        for r in results:
            assert 'parent_role' in r, "Results must include parent_role"
