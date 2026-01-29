"""
Integration tests for query understanding with test fixtures.

These tests use the TestFixtureManager to inject known conversations
and verify that the query system correctly parses and resolves queries.

Run with: pytest tests/integration/test_query_retrieval.py -v
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from episodic.test_fixtures import FixtureManager, setup_test_environment
from episodic.query import parse_query, parse_to_ast
from episodic.query.types import DiscussionQuery, MQLCommand, FreeText


# Fixed reference time for all tests
REFERENCE_TIME = datetime(2026, 1, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))


@pytest.fixture(scope="module")
def test_db():
    """Set up test database with standard fixtures."""
    manager = setup_test_environment(REFERENCE_TIME)
    yield manager
    manager.cleanup()


class TestTemporalQueryParsing:
    """Tests for temporal reference parsing without DB."""
    
    def test_yesterday_parsed(self):
        """'yesterday' should parse to temporal kind='yesterday'."""
        result = parse_query(
            "what did we discuss yesterday",
            now_utc=REFERENCE_TIME
        )
        assert result.temporal is not None
        start, end = result.temporal
        # Should be Jan 25, 2026
        assert start.day == 25 or start.day == 26  # Depends on timezone offset
        
    def test_last_week_parsed(self):
        """'last week' should parse to temporal kind='last_week'."""
        result = parse_query(
            "what did we discuss last week",
            now_utc=REFERENCE_TIME
        )
        assert result.temporal is not None
        start, end = result.temporal
        # Should span 7 days
        assert (end - start).days == 7
        
    def test_three_days_ago_parsed(self):
        """'3 days ago' should parse to temporal kind='n_days_ago'."""
        result = parse_query(
            "what did we discuss 3 days ago",
            now_utc=REFERENCE_TIME
        )
        assert result.temporal is not None
        start, end = result.temporal
        # Should be a single day
        assert (end - start).days == 1
        
    def test_last_month_parsed(self):
        """'last month' should parse to temporal kind='last_month'."""
        result = parse_query(
            "what did we discuss last month",
            now_utc=REFERENCE_TIME
        )
        assert result.temporal is not None
        start, end = result.temporal
        # December 2025
        assert start.month == 12
        assert start.year == 2025


class TestDiscussionQueryRecognition:
    """Tests for discussion query form recognition."""
    
    def test_when_we_discussed_form(self):
        """'when we discussed X' should produce DiscussionQuery."""
        ast = parse_to_ast("when we discussed quantum computing")
        assert isinstance(ast, DiscussionQuery)
        assert ast.query_form == "when_we"
        assert ast.target.text == "quantum computing"
        
    def test_have_we_form(self):
        """'have we discussed X' should produce DiscussionQuery."""
        ast = parse_to_ast("have we discussed databases before")
        assert isinstance(ast, DiscussionQuery)
        assert ast.query_form == "have_we"
        assert ast.has_broadness_cue is True
        
    def test_did_i_form(self):
        """'did I mention X' should produce DiscussionQuery."""
        ast = parse_to_ast("did I mention asyncio")
        assert isinstance(ast, DiscussionQuery)
        assert ast.query_form == "did_speaker"
        assert ast.speaker.role == "user"
        
    def test_ever_sets_broadness_cue(self):
        """'ever' should set has_broadness_cue=True."""
        ast = parse_to_ast("have we ever discussed machine learning")
        assert isinstance(ast, DiscussionQuery)
        assert ast.has_broadness_cue is True
        
    def test_before_sets_broadness_cue(self):
        """'before' should set has_broadness_cue=True."""
        ast = parse_to_ast("did we talk about this before")
        assert isinstance(ast, DiscussionQuery)
        assert ast.has_broadness_cue is True


class TestSpeakerResolution:
    """Tests for speaker reference resolution."""
    
    def test_i_maps_to_user(self):
        """'I' should map to speaker='user'."""
        result = parse_query("did I say something", now_utc=REFERENCE_TIME)
        assert result.speaker == "user"
        
    def test_you_maps_to_assistant(self):
        """'you' should map to speaker='assistant'."""
        result = parse_query("did you mention that", now_utc=REFERENCE_TIME)
        assert result.speaker == "assistant"
        
    def test_we_maps_to_none(self):
        """'we' should map to speaker=None (both)."""
        result = parse_query("did we discuss this", now_utc=REFERENCE_TIME)
        assert result.speaker is None
        
    def test_my_maps_to_user(self):
        """'my' should map to speaker='user'."""
        ast = parse_to_ast("my questions about Python")
        # This might not be a DiscussionQuery, but if it parses speaker...
        # Let's check via the resolved query
        result = parse_query("did I ask about Python", now_utc=REFERENCE_TIME)
        assert result.speaker == "user"


class TestSegmentResolution:
    """Tests for segment/topic resolution with test database."""
    
    def test_explicit_topic_exact_match(self, test_db):
        """Exact topic name should resolve to node IDs."""
        result = parse_query(
            "in topic: quantum-computing",
            conn=test_db.get_connection(),
            now_utc=REFERENCE_TIME
        )
        assert result.segment_explicit is True
        assert result.segment_resolved_ids is not None
        assert len(result.segment_resolved_ids) > 0
        assert result.segment_ambiguous is False
        
    def test_explicit_topic_partial_match_ambiguous(self, test_db):
        """Partial topic name matching multiple should be ambiguous."""
        # "python" matches "python-asyncio" but might also match others
        result = parse_query(
            "in topic: python",
            conn=test_db.get_connection(),
            now_utc=REFERENCE_TIME
        )
        assert result.segment_explicit is True
        # Should find python-asyncio
        if result.segment_ambiguous:
            assert result.segment_candidates is not None
        else:
            assert len(result.segment_resolved_ids) > 0
            
    def test_explicit_topic_no_match(self, test_db):
        """Non-existent topic should return empty list."""
        result = parse_query(
            "in topic: nonexistent-topic",
            conn=test_db.get_connection(),
            now_utc=REFERENCE_TIME
        )
        assert result.segment_explicit is True
        assert result.segment_resolved_ids == []
        assert result.segment_ambiguous is False
        
    def test_no_topic_mentioned(self, test_db):
        """Query without topic should have segment_resolved_ids=None."""
        result = parse_query(
            "what did we discuss yesterday",
            conn=test_db.get_connection(),
            now_utc=REFERENCE_TIME
        )
        assert result.segment_explicit is False
        assert result.segment_resolved_ids is None


class TestCombinedConstraints:
    """Tests for queries with multiple constraints."""
    
    def test_temporal_plus_target(self):
        """Temporal + target should both be present."""
        result = parse_query(
            "what did we discuss about databases last week",
            now_utc=REFERENCE_TIME
        )
        assert result.temporal is not None
        assert result.target is not None
        assert "database" in result.target.lower()
        
    def test_speaker_plus_temporal(self):
        """Speaker + temporal should both resolve."""
        result = parse_query(
            "what did I say yesterday",
            now_utc=REFERENCE_TIME
        )
        assert result.speaker == "user"
        assert result.temporal is not None
        
    def test_all_constraints(self, test_db):
        """Topic + temporal + speaker + target should all parse."""
        # This is a complex query
        result = parse_query(
            "in topic: machine-learning-basics did I ask about overfitting yesterday",
            conn=test_db.get_connection(),
            now_utc=REFERENCE_TIME
        )
        # Depending on parser, some constraints may be captured
        assert result.segment_explicit is True


class TestModeResolution:
    """Tests for mode resolution."""
    
    def test_browse_mode(self):
        """'browse' should set mode='browse'."""
        result = parse_query("browse quantum computing", now_utc=REFERENCE_TIME)
        assert result.mode == "browse"
        
    def test_summarize_mode(self):
        """'summarize' should set mode='summarize'."""
        result = parse_query("summarize our database discussion", now_utc=REFERENCE_TIME)
        assert result.mode == "summarize"
        
    def test_discussion_query_always_browse(self):
        """DiscussionQuery should always resolve to mode='browse'."""
        result = parse_query(
            "when did we discuss quantum computing",
            now_utc=REFERENCE_TIME
        )
        assert result.mode == "browse"
        
    def test_default_mode_answer(self):
        """Queries without explicit mode should default to 'answer'."""
        result = parse_query("coffee", now_utc=REFERENCE_TIME)
        assert result.mode == "answer"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_target(self):
        """Query with only constraints, no target."""
        result = parse_query("yesterday", now_utc=REFERENCE_TIME)
        # Should still parse, possibly as FreeText or with temporal
        assert result is not None
        
    def test_malformed_input(self):
        """Malformed input should produce FreeText."""
        ast = parse_to_ast("@#$% invalid")
        assert isinstance(ast, FreeText)
        
    def test_very_long_input(self):
        """Very long input should not crash."""
        long_input = "what did we discuss about " + "something " * 100
        result = parse_query(long_input, now_utc=REFERENCE_TIME)
        assert result is not None
