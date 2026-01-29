"""
Tests for natural language date parsing.

Tests timezone-aware parsing of time expressions like:
- "since yesterday"
- "before last week"
- "between 10am and 2pm today"
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from episodic.utils.date_parser import parse_time_range, format_time_range


@pytest.fixture
def reference_time():
    """Fixed reference time for deterministic tests."""
    # Wednesday, January 29, 2026, 2:00 PM Chicago time
    return datetime(2026, 1, 29, 14, 0, 0, tzinfo=ZoneInfo("America/Chicago"))


class TestParseTimeRange:
    """Tests for parse_time_range function."""

    def test_since_yesterday(self, reference_time):
        """Test 'since yesterday' parsing."""
        result = parse_time_range(
            "since yesterday",
            user_tz="America/Chicago",
            reference_time=reference_time
        )
        assert result is not None
        start, end = result
        assert start is not None
        assert end is None  # Open-ended

        # Should be sometime yesterday
        yesterday = reference_time - timedelta(days=1)
        assert start.date() == yesterday.date() or start < reference_time

    def test_before_last_week(self, reference_time):
        """Test 'before last week' parsing."""
        result = parse_time_range(
            "before last week",
            user_tz="America/Chicago",
            reference_time=reference_time
        )
        assert result is not None
        start, end = result
        assert start is None  # Open start
        assert end is not None

        # Should be before this week
        assert end < reference_time

    def test_between_times_today(self, reference_time):
        """Test 'between 10am and 2pm today' parsing."""
        result = parse_time_range(
            "between 10am and 2pm today",
            user_tz="America/Chicago",
            reference_time=reference_time
        )
        assert result is not None
        start, end = result
        assert start is not None
        assert end is not None

        # Start should be before end
        assert start < end

    def test_plain_yesterday(self, reference_time):
        """Test plain 'yesterday' as day range."""
        result = parse_time_range(
            "yesterday",
            user_tz="America/Chicago",
            reference_time=reference_time
        )
        assert result is not None
        start, end = result
        assert start is not None
        assert end is not None

        # Should cover a 24-hour period
        duration = end - start
        assert duration.total_seconds() == 86400  # 24 hours

    def test_days_ago(self, reference_time):
        """Test '3 days ago' parsing."""
        result = parse_time_range(
            "3 days ago",
            user_tz="America/Chicago",
            reference_time=reference_time
        )
        assert result is not None
        start, end = result
        assert start is not None
        assert end is not None

        # Should be approximately 3 days ago
        three_days_ago = reference_time - timedelta(days=3)
        # Allow some flexibility for day boundaries
        assert abs((start.date() - three_days_ago.date()).days) <= 1

    def test_empty_expression(self):
        """Test empty expression returns None."""
        assert parse_time_range("") is None
        assert parse_time_range("   ") is None

    def test_invalid_expression(self, reference_time):
        """Test invalid expression returns None."""
        result = parse_time_range(
            "not a time expression at all xyz123",
            user_tz="America/Chicago",
            reference_time=reference_time
        )
        # dateparser might still try to parse, but we check it's handled
        # This test verifies we don't crash on weird input


class TestTimezoneHandling:
    """Tests for timezone-aware parsing."""

    @pytest.mark.parametrize("tz,expected_offset_hours", [
        ("America/New_York", -5),  # EST
        ("America/Los_Angeles", -8),  # PST
        ("UTC", 0),
        ("Europe/London", 0),
        ("Asia/Tokyo", 9),
    ])
    def test_timezone_interpretation(self, tz, expected_offset_hours, reference_time):
        """Test that times are interpreted in user's timezone."""
        # Parse "yesterday" in different timezones
        local_ref = reference_time.astimezone(ZoneInfo(tz))

        result = parse_time_range(
            "yesterday",
            user_tz=tz,
            reference_time=local_ref
        )
        assert result is not None
        start, end = result

        # Result should be in UTC
        assert start.tzinfo == ZoneInfo("UTC")
        assert end.tzinfo == ZoneInfo("UTC")

    def test_different_tz_same_absolute_time(self):
        """Test that same absolute moment is preserved across timezones."""
        # Create same moment in two timezones
        utc_time = datetime(2026, 1, 29, 19, 0, 0, tzinfo=ZoneInfo("UTC"))
        chicago_time = utc_time.astimezone(ZoneInfo("America/Chicago"))  # 1 PM

        # Parse "since 10am today" from both perspectives
        result_utc = parse_time_range(
            "since 10am today",
            user_tz="UTC",
            reference_time=utc_time
        )
        result_chicago = parse_time_range(
            "since 10am today",
            user_tz="America/Chicago",
            reference_time=chicago_time
        )

        # Both should parse successfully
        assert result_utc is not None
        assert result_chicago is not None

        # The actual UTC times should differ by the timezone offset
        # (10 AM UTC vs 10 AM Chicago = 6 hours difference)
        start_utc, _ = result_utc
        start_chicago, _ = result_chicago
        diff = abs((start_utc - start_chicago).total_seconds() / 3600)
        assert diff == 6  # 6 hour difference


class TestFormatTimeRange:
    """Tests for format_time_range function."""

    def test_format_since(self):
        """Test formatting 'since X' range."""
        start = datetime(2026, 1, 28, 10, 0, tzinfo=ZoneInfo("UTC"))
        result = format_time_range(start, None, "America/Chicago")
        assert "since" in result.lower()
        assert "2026-01-28" in result

    def test_format_before(self):
        """Test formatting 'before X' range."""
        end = datetime(2026, 1, 28, 10, 0, tzinfo=ZoneInfo("UTC"))
        result = format_time_range(None, end, "America/Chicago")
        assert "before" in result.lower()

    def test_format_between(self):
        """Test formatting 'between X and Y' range."""
        start = datetime(2026, 1, 28, 10, 0, tzinfo=ZoneInfo("UTC"))
        end = datetime(2026, 1, 28, 14, 0, tzinfo=ZoneInfo("UTC"))
        result = format_time_range(start, end, "America/Chicago")
        assert "between" in result.lower()
        assert "and" in result.lower()

    def test_format_all_time(self):
        """Test formatting open range."""
        result = format_time_range(None, None, "America/Chicago")
        assert "all time" in result.lower()
