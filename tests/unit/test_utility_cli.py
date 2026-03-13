"""
Tests for Utility CLI Integration.

Tests cover:
1. Duration parsing (10s, 5m, 1h, 1h30m)
2. Time parsing (7am, 7:00pm, 19:00)
3. Remind argument parsing
4. Command routing
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from episodic.utility.cli_integration import (
    _parse_duration,
    _parse_time,
    _parse_remind_args,
    handle_utility_command,
    is_utility_command,
)
from episodic.utility.types import ResultStatus


class TestDurationParsing:
    """Tests for duration string parsing."""

    def test_seconds(self):
        """Parse seconds."""
        assert _parse_duration("10s") == 10
        assert _parse_duration("30sec") == 30
        assert _parse_duration("5secs") == 5

    def test_minutes(self):
        """Parse minutes."""
        assert _parse_duration("5m") == 300
        assert _parse_duration("10min") == 600
        assert _parse_duration("2mins") == 120
        assert _parse_duration("1minute") == 60

    def test_hours(self):
        """Parse hours."""
        assert _parse_duration("1h") == 3600
        assert _parse_duration("2hr") == 7200
        assert _parse_duration("1hour") == 3600

    def test_combined(self):
        """Parse combined durations."""
        assert _parse_duration("1h30m") == 5400
        assert _parse_duration("2h15m") == 8100
        assert _parse_duration("1h30m45s") == 5445

    def test_invalid(self):
        """Invalid durations return None."""
        assert _parse_duration("") is None
        assert _parse_duration("abc") is None

    def test_plain_numbers(self):
        """Numbers without units treated as seconds."""
        assert _parse_duration("60") == 60
        assert _parse_duration("120") == 120


class TestTimeParsing:
    """Tests for time string parsing."""

    def test_simple_am(self):
        """Parse simple AM times."""
        result = _parse_time("7am")
        assert result is not None
        assert result.hour == 7

    def test_simple_pm(self):
        """Parse simple PM times."""
        result = _parse_time("7pm")
        assert result is not None
        assert result.hour == 19

    def test_with_minutes_am(self):
        """Parse times with minutes."""
        result = _parse_time("7:30am")
        assert result is not None
        assert result.hour == 7
        assert result.minute == 30

    def test_with_minutes_pm(self):
        """Parse PM times with minutes."""
        result = _parse_time("3:45pm")
        assert result is not None
        assert result.hour == 15
        assert result.minute == 45

    def test_24_hour(self):
        """Parse 24-hour format."""
        result = _parse_time("19:00")
        assert result is not None
        assert result.hour == 19
        assert result.minute == 0

    def test_midnight(self):
        """Parse midnight."""
        result = _parse_time("12am")
        assert result is not None
        assert result.hour == 0

    def test_noon(self):
        """Parse noon."""
        result = _parse_time("12pm")
        assert result is not None
        assert result.hour == 12

    def test_invalid(self):
        """Invalid times return None."""
        assert _parse_time("") is None
        assert _parse_time("abc") is None
        assert _parse_time("25:00") is None

    def test_tomorrow_if_past(self):
        """Times in the past schedule for tomorrow."""
        # Use a time that's definitely in the past
        user_tz = "America/Chicago"
        tz = ZoneInfo(user_tz)
        now = datetime.now(tz)

        # Get a time that's definitely past
        past_hour = (now.hour - 2) % 24
        time_str = f"{past_hour}:00"

        result = _parse_time(time_str, user_tz)
        assert result is not None
        # Should be tomorrow
        assert result > now


class TestRemindParsing:
    """Tests for remind argument parsing."""

    def test_in_duration(self):
        """Parse 'X in Y' format."""
        text, duration_s, alarm_time = _parse_remind_args("call mom in 2h")
        assert text == "call mom"
        assert duration_s == 7200
        assert alarm_time is None

    def test_in_duration_minutes(self):
        """Parse reminder with minutes."""
        text, duration_s, alarm_time = _parse_remind_args("meeting in 30m")
        assert text == "meeting"
        assert duration_s == 1800
        assert alarm_time is None

    def test_at_time(self):
        """Parse 'X at Y' format."""
        text, duration_s, alarm_time = _parse_remind_args("dentist at 3pm")
        assert text == "dentist"
        assert duration_s is None
        assert alarm_time is not None
        assert alarm_time.hour == 15

    def test_complex_text_in(self):
        """Parse complex text with 'in'."""
        text, duration_s, alarm_time = _parse_remind_args("pick up dry cleaning in 1h30m")
        assert text == "pick up dry cleaning"
        assert duration_s == 5400

    def test_invalid(self):
        """Invalid formats return None."""
        text, duration_s, alarm_time = _parse_remind_args("just text")
        assert text is None
        assert duration_s is None
        assert alarm_time is None


class TestIsUtilityCommand:
    """Tests for utility command detection."""

    def test_utility_commands(self):
        """Utility commands are detected."""
        assert is_utility_command("stop") is True
        assert is_utility_command("timer") is True
        assert is_utility_command("alarm") is True
        assert is_utility_command("time") is True
        assert is_utility_command("calc") is True
        assert is_utility_command("note") is True
        assert is_utility_command("remind") is True
        assert is_utility_command("play") is True
        assert is_utility_command("pause") is True
        assert is_utility_command("cancel") is True
        assert is_utility_command("undo") is True
        assert is_utility_command("dnd") is True
        assert is_utility_command("status") is True

    def test_non_utility_commands(self):
        """Non-utility commands are not detected."""
        assert is_utility_command("help") is False
        assert is_utility_command("config") is False
        assert is_utility_command("muse") is False
        assert is_utility_command("voice") is False


class TestHandleUtilityCommand:
    """Tests for command handling (unit tests, no real execution).

    Note: These tests only test input validation that happens BEFORE
    the scheduler/database is accessed. Full integration tests
    would require a test database setup.
    """

    def test_calc_missing_expression(self):
        """Calc without expression returns error (before scheduler needed)."""
        # This returns error immediately without needing scheduler
        result = handle_utility_command("calc", "")
        assert result is not None
        assert result.status == ResultStatus.ERROR
        assert "expression" in result.error_message.lower()

    def test_play_missing_query_shows_status(self):
        """Play without query shows playback status."""
        result = handle_utility_command("play", "")
        assert result is not None
        assert result.status == ResultStatus.OK

    def test_unknown_command(self):
        """Unknown command returns None."""
        result = handle_utility_command("unknown", "args")
        assert result is None

    def test_timer_invalid_duration(self):
        """Timer with invalid duration returns error (before scheduler needed)."""
        result = handle_utility_command("timer", "xyz")
        assert result is not None
        assert result.status == ResultStatus.ERROR
        assert "duration" in result.error_message.lower()

    def test_alarm_invalid_time(self):
        """Alarm with invalid time returns error (before scheduler needed)."""
        result = handle_utility_command("alarm", "xyz")
        assert result is not None
        assert result.status == ResultStatus.ERROR
        assert "time" in result.error_message.lower()

    def test_remind_invalid_format(self):
        """Remind with invalid format returns error (before scheduler needed)."""
        result = handle_utility_command("remind", "no time specified")
        assert result is not None
        assert result.status == ResultStatus.ERROR
        assert "usage" in result.error_message.lower()

    def test_dnd_invalid_duration(self):
        """DND with invalid duration returns error (before scheduler needed)."""
        result = handle_utility_command("dnd", "xyz")
        assert result is not None
        assert result.status == ResultStatus.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
