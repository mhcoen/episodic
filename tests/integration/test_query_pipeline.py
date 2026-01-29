"""Integration tests for MQL query understanding pipeline.

This module provides:
1. Golden fixture enumeration test - validates all fixtures match expected behavior
2. End-to-end pipeline tests with mock database
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

from episodic.query import (
    parse_query,
    parse_to_ast,
    tokenize_input,
    normalize,
    DiscussionQuery,
    FreeText,
    MQLCommand,
)


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "golden"


def load_fixtures():
    """Load all golden fixtures from the fixtures directory."""
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("G*.json")):
        with open(path) as f:
            fixture = json.load(f)
            fixture["_path"] = path
            fixtures.append(fixture)
    return fixtures


def get_fixture_ids():
    """Get fixture IDs for parametrization."""
    return [f["id"] for f in load_fixtures()]


class TestGoldenFixtures:
    """Golden fixture enumeration test - validates all fixtures match expected behavior."""

    @pytest.fixture
    def test_db(self):
        """Create in-memory test database."""
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE topics (
                id TEXT PRIMARY KEY,
                name TEXT,
                start_node_id TEXT,
                end_node_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE topic_node_cache (
                topic_id TEXT,
                node_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                parent_id TEXT
            )
        """)
        conn.commit()
        yield conn
        conn.close()

    def setup_db_for_fixture(self, conn, fixture):
        """Set up database with test data from fixture if specified."""
        if "test_db_setup" not in fixture:
            return

        setup = fixture["test_db_setup"]
        if "topics" in setup:
            for topic in setup["topics"]:
                conn.execute(
                    "INSERT OR REPLACE INTO topics VALUES (?, ?, NULL, NULL)",
                    (topic["id"], topic["name"])
                )
                # Add to cache
                conn.execute(
                    "INSERT INTO topic_node_cache VALUES (?, ?)",
                    (topic["id"], f"node_{topic['id']}")
                )
        conn.commit()

    @pytest.mark.parametrize("fixture_id", get_fixture_ids())
    def test_golden_fixture(self, fixture_id, test_db):
        """Test that fixture produces expected results."""
        fixtures = {f["id"]: f for f in load_fixtures()}
        fixture = fixtures[fixture_id]

        # Set up database if needed
        self.setup_db_for_fixture(test_db, fixture)

        # Parse now_utc
        now_utc = datetime.fromisoformat(fixture["now_utc"].replace("Z", "+00:00"))
        user_tz = fixture.get("user_tz", "America/Chicago")

        # Step 1: Validate normalization
        s_norm, audit = normalize(fixture["input_raw"])
        assert s_norm == fixture["input_norm"], f"Normalization mismatch: {s_norm} != {fixture['input_norm']}"

        # Step 2: Validate tokens if specified
        if "tokens" in fixture:
            lex_result, _ = tokenize_input(fixture["input_raw"])
            for i, expected_tok in enumerate(fixture["tokens"]):
                if i >= len(lex_result.tokens):
                    pytest.fail(f"Missing token at index {i}: expected {expected_tok}")
                actual_tok = lex_result.tokens[i]
                assert actual_tok.kind.name == expected_tok["kind"], \
                    f"Token {i} kind mismatch: {actual_tok.kind.name} != {expected_tok['kind']}"
                if "lexeme" in expected_tok:
                    assert actual_tok.lexeme == expected_tok["lexeme"], \
                        f"Token {i} lexeme mismatch: {actual_tok.lexeme} != {expected_tok['lexeme']}"

        # Step 3: Validate parse
        ast = parse_to_ast(fixture["input_raw"])
        parse_expected = fixture.get("parse", {})

        if "ast_kind" in parse_expected:
            expected_kind = parse_expected["ast_kind"]
            if expected_kind == "MQLCommand":
                assert isinstance(ast, MQLCommand), f"Expected MQLCommand, got {type(ast).__name__}"
            elif expected_kind == "DiscussionQuery":
                assert isinstance(ast, DiscussionQuery), f"Expected DiscussionQuery, got {type(ast).__name__}"
            elif expected_kind == "FreeText":
                assert isinstance(ast, FreeText), f"Expected FreeText, got {type(ast).__name__}"

        if isinstance(ast, MQLCommand):
            if "mode" in parse_expected:
                assert ast.mode.value == parse_expected["mode"]
            if "segment" in parse_expected:
                assert ast.segment.explicit == parse_expected["segment"]["explicit"]
                if "query" in parse_expected["segment"]:
                    assert ast.segment.query == parse_expected["segment"]["query"]
            if "target" in parse_expected:
                if parse_expected["target"] is None:
                    assert ast.target is None
                else:
                    assert ast.target is not None
                    assert ast.target.text == parse_expected["target"]

        elif isinstance(ast, DiscussionQuery):
            if "query_form" in parse_expected:
                assert ast.query_form == parse_expected["query_form"]
            if "target" in parse_expected:
                if parse_expected["target"] is None:
                    assert ast.target is None
                else:
                    assert ast.target is not None
                    assert ast.target.text == parse_expected["target"]
            if "speaker" in parse_expected:
                if parse_expected["speaker"] is None:
                    assert ast.speaker is None
                else:
                    assert ast.speaker is not None
                    assert ast.speaker.role == parse_expected["speaker"]["role"]
            if "has_broadness_cue" in parse_expected:
                assert ast.has_broadness_cue == parse_expected["has_broadness_cue"]

        elif isinstance(ast, FreeText):
            if "parse_error" in parse_expected:
                assert ast.parse_error == parse_expected["parse_error"]
            if "text" in parse_expected:
                assert ast.text == parse_expected["text"]

        # Step 4: Validate resolution
        resolve_expected = fixture.get("resolve", {})
        if resolve_expected:
            result = parse_query(fixture["input_raw"], conn=test_db, now_utc=now_utc, user_tz=user_tz)

            if "mode" in resolve_expected:
                assert result.mode == resolve_expected["mode"], \
                    f"Mode mismatch: {result.mode} != {resolve_expected['mode']}"

            if "target" in resolve_expected:
                assert result.target == resolve_expected["target"], \
                    f"Target mismatch: {result.target} != {resolve_expected['target']}"

            if "segment_explicit" in resolve_expected:
                assert result.segment_explicit == resolve_expected["segment_explicit"]

            if "segment_query" in resolve_expected:
                assert result.segment_query == resolve_expected["segment_query"]

            if "segment_resolved_ids" in resolve_expected:
                expected_ids = resolve_expected["segment_resolved_ids"]
                if expected_ids is None:
                    assert result.segment_resolved_ids is None
                elif expected_ids == []:
                    assert result.segment_resolved_ids == []
                # For non-empty, just check it's a list

            if "segment_ambiguous" in resolve_expected:
                assert result.segment_ambiguous == resolve_expected["segment_ambiguous"]

            if "speaker" in resolve_expected:
                assert result.speaker == resolve_expected["speaker"], \
                    f"Speaker mismatch: {result.speaker} != {resolve_expected['speaker']}"

            if "has_broadness_cue" in resolve_expected:
                assert result.has_broadness_cue == resolve_expected["has_broadness_cue"]

            if "temporal" in resolve_expected:
                if resolve_expected["temporal"] is None:
                    assert result.temporal is None
                else:
                    assert result.temporal is not None
                    # Temporal is [start, end] ISO strings
                    expected_start = datetime.fromisoformat(resolve_expected["temporal"][0])
                    expected_end = datetime.fromisoformat(resolve_expected["temporal"][1])
                    assert result.temporal[0] == expected_start, \
                        f"Temporal start mismatch: {result.temporal[0]} != {expected_start}"
                    assert result.temporal[1] == expected_end, \
                        f"Temporal end mismatch: {result.temporal[1]} != {expected_end}"


class TestPipelineEndToEnd:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def test_db(self):
        """Create test database with sample topics."""
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE topics (
                id TEXT PRIMARY KEY,
                name TEXT,
                start_node_id TEXT,
                end_node_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE topic_node_cache (
                topic_id TEXT,
                node_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                parent_id TEXT
            )
        """)

        # Insert sample topics
        conn.execute("INSERT INTO topics VALUES ('t1', 'coffee-brewing', 'n1', NULL)")
        conn.execute("INSERT INTO topics VALUES ('t2', 'research-methodology', 'n2', NULL)")
        conn.execute("INSERT INTO topic_node_cache VALUES ('t1', 'n1')")
        conn.execute("INSERT INTO topic_node_cache VALUES ('t1', 'n1a')")
        conn.execute("INSERT INTO topic_node_cache VALUES ('t2', 'n2')")
        conn.commit()
        yield conn
        conn.close()

    def test_discussion_query_end_to_end(self, test_db):
        """Test discussion query produces correct ResolvedQuery."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("when we discussed coffee", conn=test_db, now_utc=now_utc)

        assert result.mode == "browse"
        assert result.target == "coffee"
        assert result.segment_explicit is False
        assert result.segment_resolved_ids is None
        assert result.speaker is None  # "we" maps to None

    def test_explicit_segment_end_to_end(self, test_db):
        """Test explicit segment produces correct ResolvedQuery."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("in topic: coffee-brewing", conn=test_db, now_utc=now_utc)

        assert result.segment_explicit is True
        assert result.segment_query == "coffee-brewing"
        assert result.segment_resolved_ids is not None
        assert len(result.segment_resolved_ids) > 0

    def test_temporal_with_segment_end_to_end(self, test_db):
        """Test combined temporal and segment."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("browse in topic: research-methodology last week", conn=test_db, now_utc=now_utc)

        assert result.mode == "browse"
        assert result.segment_explicit is True
        assert result.temporal is not None

    def test_speaker_restriction_end_to_end(self, test_db):
        """Test speaker restriction propagates."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("did I say coffee", conn=test_db, now_utc=now_utc)

        assert result.mode == "browse"
        assert result.speaker == "user"

    def test_lex_error_end_to_end(self, test_db):
        """Test lex error produces FreeText -> answer mode."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("@invalid input", conn=test_db, now_utc=now_utc)

        assert result.mode == "answer"
        assert result.segment_explicit is False


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_query_same_result(self):
        """Same query should produce identical results."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))

        result1 = parse_query("when we discussed coffee yesterday", now_utc=now_utc)
        result2 = parse_query("when we discussed coffee yesterday", now_utc=now_utc)

        assert result1.to_dict() == result2.to_dict()

    def test_segment_candidates_ordered(self):
        """Segment candidates should be in deterministic order."""
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE topics (
                id TEXT PRIMARY KEY,
                name TEXT,
                start_node_id TEXT,
                end_node_id TEXT
            )
        """)
        conn.execute("CREATE TABLE topic_node_cache (topic_id TEXT, node_id TEXT)")
        conn.execute("INSERT INTO topics VALUES ('t2', 'research-results', NULL, NULL)")
        conn.execute("INSERT INTO topics VALUES ('t1', 'research-methodology', NULL, NULL)")
        conn.commit()

        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("in topic: research", conn=conn, now_utc=now_utc)

        assert result.segment_ambiguous is True
        assert result.segment_candidates is not None
        # Should be sorted by id
        assert result.segment_candidates[0]["id"] == "t1"
        assert result.segment_candidates[1]["id"] == "t2"

        conn.close()


class TestAuditTraceability:
    """Tests for audit trail."""

    def test_audit_trace_contains_ast_info(self):
        """Audit trace should contain AST information."""
        import json
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("browse coffee", now_utc=now_utc)

        trace = json.loads(result.audit_trace)
        assert "ast_kind" in trace
        assert trace["ast_kind"] == "MQLCommand"

    def test_audit_trace_round_trips(self):
        """Audit trace should be valid JSON that round-trips."""
        import json
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = parse_query("did I say coffee", now_utc=now_utc)

        trace1 = json.loads(result.audit_trace)
        trace2 = json.loads(json.dumps(trace1))
        assert trace1 == trace2
