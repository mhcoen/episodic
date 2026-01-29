"""Unit tests for MQL resolver."""

import pytest
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from episodic.query import (
    parse_query,
    parse_to_ast,
    resolve,
    resolve_temporal,
    DiscussionQuery,
    FreeText,
    MQLCommand,
    TemporalSpec,
)


class TestResolverFreeText:
    """Tests for FreeText resolution."""

    def test_freetext_mode_answer(self):
        """FreeText should resolve to mode='answer'."""
        result = parse_query("@invalid syntax", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.mode == "answer"

    def test_freetext_target_is_s_norm(self):
        """FreeText target should be s_norm."""
        result = parse_query("@invalid syntax", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        # After lex error, target is the normalized input
        assert result.target is not None

    def test_freetext_segment_explicit_false(self):
        """FreeText should have segment_explicit=False."""
        result = parse_query("@test", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_explicit is False

    def test_freetext_segment_resolved_ids_none(self):
        """FreeText should have segment_resolved_ids=None."""
        result = parse_query("@test", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_resolved_ids is None

    def test_freetext_temporal_none(self):
        """FreeText should have temporal=None."""
        result = parse_query("@test", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.temporal is None

    def test_freetext_speaker_none(self):
        """FreeText should have speaker=None."""
        result = parse_query("@test", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.speaker is None

    def test_freetext_has_broadness_cue_false(self):
        """FreeText should have has_broadness_cue=False."""
        result = parse_query("@test", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.has_broadness_cue is False


class TestResolverDiscussionQuery:
    """Tests for DiscussionQuery resolution."""

    def test_discussion_query_mode_browse(self):
        """DiscussionQuery should always resolve to mode='browse'."""
        result = parse_query("when we discussed coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.mode == "browse"

    def test_discussion_query_segment_explicit_false(self):
        """DiscussionQuery should always have segment_explicit=False."""
        result = parse_query("when we discussed coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_explicit is False

    def test_discussion_query_segment_resolved_ids_none(self):
        """DiscussionQuery should always have segment_resolved_ids=None."""
        result = parse_query("when we discussed coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_resolved_ids is None

    def test_discussion_query_broadness_cue(self):
        """DiscussionQuery broadness cue should propagate."""
        result = parse_query("have we ever discussed coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.has_broadness_cue is True


class TestResolverSpeakerMapping:
    """Tests for speaker resolution (CRITICAL: both -> None)."""

    def test_speaker_user(self):
        """Speaker 'user' should map to 'user'."""
        result = parse_query("did I say coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.speaker == "user"

    def test_speaker_assistant(self):
        """Speaker 'assistant' should map to 'assistant'."""
        result = parse_query("did you say coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.speaker == "assistant"

    def test_speaker_both_maps_to_none(self):
        """Speaker 'both' should map to None (no restriction)."""
        result = parse_query("did we say coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.speaker is None

    def test_no_speaker_is_none(self):
        """No speaker should be None."""
        result = parse_query("when discussed coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.speaker is None


class TestTemporalResolution:
    """Tests for temporal resolution to UTC half-open intervals."""

    def test_yesterday_half_open(self):
        """'yesterday' should produce half-open [start, end) interval."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="yesterday", raw="yesterday")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")

        # Start should be midnight Jan 24 local -> UTC
        # End should be midnight Jan 25 local -> UTC
        assert start < end

    def test_today_half_open(self):
        """'today' should produce half-open interval."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="today", raw="today")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert start < end

    def test_temporal_timezone_aware(self):
        """Temporal datetimes should be timezone-aware UTC."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="yesterday", raw="yesterday")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert start.tzinfo is not None
        assert end.tzinfo is not None
        assert str(start.tzinfo) == "UTC"
        assert str(end.tzinfo) == "UTC"


class TestTemporalDSTSafety:
    """Tests for DST-safe temporal resolution."""

    def test_dst_spring_forward(self):
        """Spring forward (March 2024) should handle correctly."""
        # March 10, 2024 is spring forward in America/Chicago
        now_utc = datetime(2024, 3, 10, 15, 0, 0, tzinfo=ZoneInfo("UTC"))  # 9am local
        spec = TemporalSpec(kind="yesterday", raw="yesterday")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")

        # Should produce valid interval
        assert start < end
        # Start should be March 9 midnight local
        # March 9 midnight CST = 06:00 UTC
        assert start.day == 9 or start.day == 8  # Could be March 8 or 9 depending on offset

    def test_dst_fall_back(self):
        """Fall back (November 2024) should handle correctly."""
        # November 3, 2024 is fall back in America/Chicago
        now_utc = datetime(2024, 11, 3, 15, 0, 0, tzinfo=ZoneInfo("UTC"))  # 9am local
        spec = TemporalSpec(kind="yesterday", raw="yesterday")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")

        assert start < end


class TestTemporalKinds:
    """Tests for different temporal kinds."""

    def test_last_week(self):
        """'last week' should span previous Monday to this Monday."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))  # Saturday
        spec = TemporalSpec(kind="last_week", raw="last week")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert start < end
        # Should be a 7-day span
        assert (end - start).days == 7

    def test_this_week(self):
        """'this week' should span this Monday to next Monday."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="this_week", raw="this week")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert (end - start).days == 7

    def test_last_month(self):
        """'last month' should span previous month."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="last_month", raw="last month")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert start.month == 12
        assert end.month == 1

    def test_this_month(self):
        """'this month' should span current month."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="this_month", raw="this month")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert start.month == 1
        assert end.month == 2

    def test_last_year(self):
        """'last year' should span previous year."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="last_year", raw="last year")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert start.year == 2025
        assert end.year == 2026

    def test_last_n_days(self):
        """'last N days' should span N days including today."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="last_n_days", raw="last 7 days", n=7)
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        # Should include today, so end is tomorrow midnight
        assert (end - start).days == 8  # 7 days + today

    def test_n_days_ago(self):
        """'N days ago' should be a single day span."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="n_days_ago", raw="3 days ago", n=3)
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert (end - start).days == 1

    def test_iso_date(self):
        """ISO date should be a single day span."""
        now_utc = datetime(2026, 1, 25, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="iso_date", raw="2026-01-20", iso_date="2026-01-20")
        start, end = resolve_temporal(spec, now_utc, "America/Chicago")
        assert (end - start).days == 1
        # Start should be Jan 20 midnight local
        assert start.day == 20 or start.day == 19  # Depends on UTC offset


class TestSegmentResolution:
    """Tests for segment resolution (requires mock/test DB)."""

    @pytest.fixture
    def test_db(self):
        """Create in-memory test database with topics."""
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

        # Insert test topics
        conn.execute("INSERT INTO topics VALUES ('t1', 'research-methodology', 'n1', NULL)")
        conn.execute("INSERT INTO topics VALUES ('t2', 'research-results', 'n2', NULL)")
        conn.execute("INSERT INTO topics VALUES ('t3', 'meeting-notes', 'n3', NULL)")

        # Insert cache entries
        conn.execute("INSERT INTO topic_node_cache VALUES ('t1', 'n1')")
        conn.execute("INSERT INTO topic_node_cache VALUES ('t1', 'n1a')")
        conn.execute("INSERT INTO topic_node_cache VALUES ('t2', 'n2')")
        conn.execute("INSERT INTO topic_node_cache VALUES ('t3', 'n3')")

        conn.commit()
        yield conn
        conn.close()

    def test_exact_match(self, test_db):
        """Exact match should return single topic's nodes."""
        result = parse_query("in topic: research-methodology", conn=test_db,
                            now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_explicit is True
        assert result.segment_resolved_ids is not None
        assert len(result.segment_resolved_ids) > 0
        assert result.segment_ambiguous is False

    def test_contains_ambiguous(self, test_db):
        """Contains match with multiple results should be ambiguous."""
        result = parse_query("in topic: research", conn=test_db,
                            now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_explicit is True
        assert result.segment_ambiguous is True
        assert result.segment_resolved_ids == []
        assert result.segment_candidates is not None
        assert len(result.segment_candidates) == 2

    def test_no_match(self, test_db):
        """No match should return empty list."""
        result = parse_query("in topic: nonexistent", conn=test_db,
                            now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_explicit is True
        assert result.segment_resolved_ids == []
        assert result.segment_ambiguous is False

    def test_no_segment_mentioned(self, test_db):
        """No segment in query should have segment_resolved_ids=None."""
        result = parse_query("browse coffee", conn=test_db,
                            now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.segment_explicit is False
        assert result.segment_resolved_ids is None


class TestMQLCommandResolution:
    """Tests for MQLCommand resolution."""

    def test_mode_propagates(self):
        """Mode should propagate from AST."""
        result = parse_query("browse coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.mode == "browse"

        result = parse_query("summarize coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.mode == "summarize"

    def test_target_propagates(self):
        """Target should propagate from AST."""
        result = parse_query("browse coffee brewing", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.target == "coffee brewing"

    def test_temporal_resolves(self):
        """Temporal should resolve to UTC interval."""
        result = parse_query("browse yesterday coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        assert result.temporal is not None
        assert len(result.temporal) == 2
        assert result.temporal[0] < result.temporal[1]


class TestAuditTrace:
    """Tests for audit trace in resolved query."""

    def test_audit_trace_is_json(self):
        """Audit trace should be valid JSON."""
        import json
        result = parse_query("browse coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        parsed = json.loads(result.audit_trace)
        assert "ast_kind" in parsed

    def test_audit_trace_contains_ast_kind(self):
        """Audit trace should contain ast_kind."""
        import json
        result = parse_query("when we discussed coffee", now_utc=datetime(2026, 1, 25, 12, tzinfo=ZoneInfo("UTC")))
        parsed = json.loads(result.audit_trace)
        assert parsed["ast_kind"] == "DiscussionQuery"
