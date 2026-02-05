"""Tests for speech formatters."""

from datetime import datetime

import pytest

from episodic.utility.speech.formatters import (
    format_time_speech,
    format_duration_speech,
    format_temp_speech,
    format_ordinal,
    format_date_speech,
)


class TestFormatTimeSpeech:
    """Test time formatting for speech."""

    def test_midnight(self):
        dt = datetime(2026, 2, 4, 0, 0)
        assert format_time_speech(dt) == "midnight"

    def test_noon(self):
        dt = datetime(2026, 2, 4, 12, 0)
        assert format_time_speech(dt) == "noon"

    def test_morning_on_the_hour(self):
        dt = datetime(2026, 2, 4, 7, 0)
        assert format_time_speech(dt) == "seven AM"

    def test_afternoon_on_the_hour(self):
        dt = datetime(2026, 2, 4, 15, 0)
        assert format_time_speech(dt) == "three PM"

    def test_half_hour(self):
        dt = datetime(2026, 2, 4, 7, 30)
        assert format_time_speech(dt) == "seven thirty AM"

    def test_quarter_past(self):
        dt = datetime(2026, 2, 4, 15, 15)
        assert format_time_speech(dt) == "three fifteen PM"

    def test_single_digit_minutes(self):
        dt = datetime(2026, 2, 4, 9, 5)
        assert format_time_speech(dt) == "nine oh five AM"

    def test_evening(self):
        dt = datetime(2026, 2, 4, 19, 45)
        assert format_time_speech(dt) == "seven forty-five PM"

    def test_11pm(self):
        dt = datetime(2026, 2, 4, 23, 0)
        assert format_time_speech(dt) == "eleven PM"


class TestFormatDurationSpeech:
    """Test duration formatting for speech."""

    def test_one_second(self):
        assert format_duration_speech(1) == "one second"

    def test_multiple_seconds(self):
        assert format_duration_speech(30) == "thirty seconds"

    def test_one_minute(self):
        assert format_duration_speech(60) == "one minute"

    def test_five_minutes(self):
        assert format_duration_speech(300) == "five minutes"

    def test_minutes_and_seconds(self):
        assert format_duration_speech(90) == "one minute and thirty seconds"

    def test_one_hour(self):
        assert format_duration_speech(3600) == "one hour"

    def test_one_hour_thirty_minutes(self):
        assert format_duration_speech(5400) == "one hour and thirty minutes"

    def test_two_hours(self):
        assert format_duration_speech(7200) == "two hours"

    def test_hours_and_minutes(self):
        # 2 hours 15 minutes
        assert format_duration_speech(8100) == "two hours and fifteen minutes"

    def test_hours_only_ignores_seconds(self):
        # 1 hour 30 seconds - seconds dropped when hours present
        assert format_duration_speech(3630) == "one hour"


class TestFormatTempSpeech:
    """Test temperature formatting for speech."""

    def test_positive_temp(self):
        assert format_temp_speech(72) == "72 degrees"

    def test_zero(self):
        assert format_temp_speech(0) == "zero degrees"

    def test_negative_temp(self):
        assert format_temp_speech(-5) == "negative 5 degrees"

    def test_freezing(self):
        assert format_temp_speech(32) == "32 degrees"


class TestFormatOrdinal:
    """Test ordinal number formatting."""

    def test_first(self):
        assert format_ordinal(1) == "1st"

    def test_second(self):
        assert format_ordinal(2) == "2nd"

    def test_third(self):
        assert format_ordinal(3) == "3rd"

    def test_fourth(self):
        assert format_ordinal(4) == "4th"

    def test_eleventh(self):
        assert format_ordinal(11) == "11th"

    def test_twelfth(self):
        assert format_ordinal(12) == "12th"

    def test_thirteenth(self):
        assert format_ordinal(13) == "13th"

    def test_twenty_first(self):
        assert format_ordinal(21) == "21st"

    def test_twenty_second(self):
        assert format_ordinal(22) == "22nd"

    def test_thirty_first(self):
        assert format_ordinal(31) == "31st"


class TestFormatDateSpeech:
    """Test date formatting for speech."""

    def test_february_4th(self):
        dt = datetime(2026, 2, 4)
        assert format_date_speech(dt) == "February 4th, 2026"

    def test_january_1st(self):
        dt = datetime(2026, 1, 1)
        assert format_date_speech(dt) == "January 1st, 2026"

    def test_december_25th(self):
        dt = datetime(2026, 12, 25)
        assert format_date_speech(dt) == "December 25th, 2026"
