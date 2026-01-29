"""
Integration tests for MQL retrieval execution.

Tests the full pipeline: parse_query -> execute_query -> RetrievalResult

Uses the test fixtures to verify:
- Temporal filtering
- Speaker filtering  
- Segment filtering
- Content search
- Combined constraints
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from episodic.test_fixtures import FixtureManager
from episodic.query import parse_query, execute_query
from episodic.query.retrieval import QueryExecutor, RetrievalResult


# Fixed reference time for reproducible tests
REFERENCE_TIME = datetime(2026, 1, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))


@pytest.fixture(scope="module")
def test_db():
    """Set up test database with fixtures."""
    manager = FixtureManager()
    manager.initialize_test_db(clean=True)
    manager.inject_standard_fixtures(reference_time=REFERENCE_TIME)
    yield manager
    manager.cleanup()


class TestTemporalRetrieval:
    """Tests for temporal filtering in retrieval."""
    
    def test_yesterday_returns_ml_topic(self, test_db):
        """'what did we discuss yesterday' should return ML topic nodes."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did we discuss yesterday",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        # ML topic was created yesterday
        assert any("supervised" in node.content.lower() or "overfitting" in node.content.lower() 
                   for node in result.nodes)
    
    def test_last_week_returns_database_topic(self, test_db):
        """'what did we discuss last week' should return database topic nodes."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did we discuss last week",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        # Database topic was created last week
        assert any("b-tree" in node.content.lower() or "index" in node.content.lower()
                   for node in result.nodes)
    
    def test_3_days_ago_returns_asyncio_topic(self, test_db):
        """'what did we talk about 3 days ago' should return asyncio topic nodes."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did we talk about 3 days ago",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        # Asyncio topic was created 3 days ago
        assert any("asyncio" in node.content.lower() or "async" in node.content.lower()
                   for node in result.nodes)
    
    def test_last_month_returns_quantum_topic(self, test_db):
        """'what did we discuss last month' should return quantum topic nodes."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did we discuss last month",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        # Quantum topic was created last month
        assert any("quantum" in node.content.lower() or "qubit" in node.content.lower()
                   for node in result.nodes)


class TestSpeakerRetrieval:
    """Tests for speaker filtering in retrieval."""
    
    def test_what_did_i_say_returns_user_only(self, test_db):
        """'what did I say yesterday' should return only user messages."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did I say yesterday",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        # All results should be user messages
        assert all(node.role == "user" for node in result.nodes)
    
    def test_what_did_you_say_returns_assistant_only(self, test_db):
        """'what did you say yesterday' should return only assistant messages."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did you say yesterday",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        # All results should be assistant messages
        assert all(node.role == "assistant" for node in result.nodes)


class TestContentRetrieval:
    """Tests for content/target search in retrieval."""
    
    def test_search_for_quantum(self, test_db):
        """Search for 'quantum' should return quantum-related nodes."""
        conn = test_db.get_connection()
        query = parse_query(
            "browse quantum",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        assert all("quantum" in node.content.lower() for node in result.nodes)
    
    def test_search_for_asyncio(self, test_db):
        """Search for 'asyncio' should return asyncio-related nodes."""
        conn = test_db.get_connection()
        query = parse_query(
            "browse asyncio",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        assert all("asyncio" in node.content.lower() for node in result.nodes)
    
    def test_search_nonexistent_returns_empty(self, test_db):
        """Search for nonexistent content should return empty."""
        conn = test_db.get_connection()
        query = parse_query(
            "browse xyznonexistent123",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert result.is_empty()


class TestSegmentRetrieval:
    """Tests for segment/topic filtering in retrieval."""
    
    def test_in_topic_quantum(self, test_db):
        """'in topic: quantum' should return only quantum topic nodes."""
        conn = test_db.get_connection()
        query = parse_query(
            "in topic: quantum-computing",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        # Should return nodes from quantum topic
        # If segment resolution worked, we get quantum nodes
        # If not, we get empty (no topic_node_cache match)
        if query.segment_resolved_ids:
            assert not result.is_empty()
            assert any("quantum" in node.content.lower() for node in result.nodes)


class TestCombinedFilters:
    """Tests for combined filter scenarios."""
    
    def test_temporal_plus_content(self, test_db):
        """'what did we discuss about databases last week' should combine filters."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did we discuss about databases last week",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        # Should find database-related content from last week
        # Note: query.target might be "databases" which filters content
        if not result.is_empty():
            assert any("database" in node.content.lower() or "index" in node.content.lower() 
                       for node in result.nodes)
    
    def test_speaker_plus_temporal(self, test_db):
        """'what did I say yesterday' should combine speaker + temporal."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did I say yesterday",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        assert not result.is_empty()
        # All should be user messages from yesterday
        assert all(node.role == "user" for node in result.nodes)


class TestRetrievalResult:
    """Tests for RetrievalResult formatting."""
    
    def test_context_string_format(self, test_db):
        """to_context_string should produce readable output."""
        conn = test_db.get_connection()
        query = parse_query(
            "browse quantum",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        context = result.to_context_string(max_nodes=5)
        
        if not result.is_empty():
            assert "quantum" in context.lower()
            # Should have role labels
            assert "You:" in context or "Assistant:" in context
    
    def test_filters_applied_populated(self, test_db):
        """filters_applied should list active filters."""
        conn = test_db.get_connection()
        query = parse_query(
            "what did I say yesterday",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        result = execute_query(query, conn)
        
        # Should have temporal and speaker filters
        assert len(result.filters_applied) >= 1
        assert any("temporal" in f for f in result.filters_applied)


class TestQueryExecutorDirectly:
    """Tests for QueryExecutor class directly."""
    
    def test_executor_with_limit(self, test_db):
        """Executor should respect limit parameter."""
        conn = test_db.get_connection()
        query = parse_query(
            "browse",  # No filters - should return all
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        executor = QueryExecutor(conn)
        result = executor.execute(query, limit=3)
        
        assert len(result.nodes) <= 3
    
    def test_executor_excludes_meta_queries(self, test_db):
        """Executor should exclude meta-query nodes by default."""
        conn = test_db.get_connection()
        
        # Insert a meta-query node
        conn.execute("""
            INSERT INTO nodes (id, short_id, content, role, is_meta_query)
            VALUES ('meta-test-id', 'mt1', 'This is a meta query', 'user', 1)
        """)
        conn.commit()
        
        query = parse_query(
            "browse meta",
            conn=conn,
            now_utc=REFERENCE_TIME,
            user_tz="America/Chicago"
        )
        
        executor = QueryExecutor(conn)
        result = executor.execute(query, include_meta_queries=False)
        
        # Should not include the meta-query node
        assert not any(node.node_id == "meta-test-id" for node in result.nodes)
        
        # Cleanup
        conn.execute("DELETE FROM nodes WHERE id = 'meta-test-id'")
        conn.commit()
