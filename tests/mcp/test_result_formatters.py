"""Tests for MCP result formatters."""

import json
import pytest

from episodic.mcp.result_formatters import (
    parse_content,
    format_result,
)


class TestParseContent:
    """Tests for parse_content()."""

    def test_json_string_in_content(self):
        """JSON string in content list is parsed."""
        raw = {"content": [json.dumps([{"id": "1", "subject": "Hi"}])]}
        items = parse_content(raw)
        assert len(items) == 1
        assert items[0]["id"] == "1"

    def test_multiple_json_items(self):
        """Multiple items in JSON array are extracted."""
        raw = {"content": [json.dumps([
            {"id": "1", "subject": "A"},
            {"id": "2", "subject": "B"},
        ])]}
        items = parse_content(raw)
        assert len(items) == 2

    def test_dict_in_content(self):
        """Dict items in content list are kept as-is."""
        raw = {"content": [{"id": "1"}]}
        items = parse_content(raw)
        assert len(items) == 1
        assert items[0]["id"] == "1"

    def test_plain_string_in_content(self):
        """Non-JSON strings are kept as-is."""
        raw = {"content": ["No emails found"]}
        items = parse_content(raw)
        assert len(items) == 1
        assert items[0] == "No emails found"

    def test_empty_content(self):
        """Empty content list returns empty."""
        assert parse_content({"content": []}) == []

    def test_non_dict_input(self):
        """Non-dict input returns empty."""
        assert parse_content("string") == []
        assert parse_content(None) == []

    def test_no_content_key(self):
        """Dict without content key returns empty."""
        assert parse_content({"error": "bad"}) == []

    def test_empty_string_in_content(self):
        """Empty strings are skipped."""
        raw = {"content": ["", "  ", json.dumps({"id": "1"})]}
        items = parse_content(raw)
        assert len(items) == 1


class TestFormatEmails:
    """Tests for email formatting."""

    def test_single_email(self):
        """Single email formatted with subject and sender."""
        raw = {"content": [json.dumps([{
            "id": "msg1",
            "subject": "Test Subject",
            "from": "Alice <alice@example.com>",
            "date": "Thu, 12 Feb 2026 04:20:32 GMT",
            "snippet": "Hello there",
        }])]}
        display, speech, items = format_result("email.search", raw)
        assert "Test Subject" in display
        assert "Alice" in display
        assert "Feb 12" in display
        assert "Hello there" in display
        assert len(items) == 1
        assert "1 email" in speech

    def test_multiple_emails(self):
        """Multiple emails formatted as numbered list."""
        emails = [
            {"id": f"msg{i}", "subject": f"Subject {i}", "from": f"User{i}"}
            for i in range(3)
        ]
        raw = {"content": [json.dumps(emails)]}
        display, speech, items = format_result("email.search", raw)
        assert "3 emails" in display
        assert "1." in display
        assert "2." in display
        assert "3." in display
        assert len(items) == 3

    def test_no_emails(self):
        """Empty email result shows 'no emails'."""
        raw = {"content": ["[]"]}
        display, speech, items = format_result("email.search", raw)
        assert "No emails" in display
        assert items == []

    def test_long_snippet_truncated(self):
        """Long snippets are truncated."""
        raw = {"content": [json.dumps([{
            "id": "1",
            "subject": "Test",
            "from": "X",
            "snippet": "A" * 200,
        }])]}
        display, _, _ = format_result("email.search", raw)
        assert "..." in display

    def test_email_get_uses_email_formatter(self):
        """email.get also uses email formatter."""
        raw = {"content": [json.dumps([{
            "id": "1", "subject": "Single Email", "from": "Bob",
        }])]}
        display, _, items = format_result("email.get", raw)
        assert "Single Email" in display


class TestFormatEvents:
    """Tests for calendar event formatting."""

    def test_single_event(self):
        """Single event with summary and time."""
        raw = {"content": [json.dumps([{
            "id": "evt1",
            "summary": "Team Standup",
            "start": "2026-02-13T09:00:00-06:00",
            "end": "2026-02-13T09:30:00-06:00",
        }])]}
        display, speech, items = format_result("calendar.query", raw)
        assert "Team Standup" in display
        assert "9:00 AM" in display
        assert "9:30 AM" in display
        assert "1 event" in speech

    def test_event_with_location(self):
        """Event location is shown."""
        raw = {"content": [json.dumps([{
            "id": "evt1",
            "summary": "Lunch",
            "start": "2026-02-13T12:00:00",
            "location": "Cafe XYZ",
        }])]}
        display, _, _ = format_result("calendar.query", raw)
        assert "Cafe XYZ" in display

    def test_no_events(self):
        """Empty event result shows 'no events'."""
        raw = {"content": ["[]"]}
        display, _, items = format_result("calendar.query", raw)
        assert "No events" in display

    def test_event_dict_start(self):
        """Events with dict start/end objects are handled."""
        raw = {"content": [json.dumps([{
            "id": "evt1",
            "summary": "All Day",
            "start": {"date": "2026-02-14"},
            "end": {"date": "2026-02-15"},
        }])]}
        display, _, _ = format_result("calendar.query", raw)
        assert "All Day" in display
        assert "2026-02-14" in display

    def test_freebusy_uses_event_formatter(self):
        """calendar.freebusy also uses event formatter."""
        raw = {"content": [json.dumps([{
            "id": "1", "summary": "Busy Block",
        }])]}
        display, _, _ = format_result("calendar.freebusy", raw)
        assert "Busy Block" in display


class TestFormatCalendars:
    """Tests for calendar list formatting."""

    def test_calendar_list(self):
        """Calendar list shows names and primary marker."""
        raw = {"content": [json.dumps([
            {"id": "primary", "summary": "Main Calendar", "primary": True},
            {"id": "work@group.calendar.google.com", "summary": "Work"},
        ])]}
        display, speech, items = format_result("calendar.list", raw)
        assert "Main Calendar" in display
        assert "(primary)" in display
        assert "Work" in display
        assert "2 calendars" in speech

    def test_empty_calendar_list(self):
        """No calendars shows appropriate message."""
        raw = {"content": ["[]"]}
        display, _, _ = format_result("calendar.list", raw)
        assert "No calendars" in display


class TestFormatWriteOps:
    """Tests for write operation formatting."""

    def test_draft_created(self):
        """Draft creation shows confirmation."""
        raw = {"content": [json.dumps({
            "to": "bob@test.com",
            "subject": "Lunch",
        })]}
        display, _, _ = format_result("email.create_draft", raw)
        assert "Draft created" in display
        assert "bob@test.com" in display

    def test_reply_sent(self):
        """Reply shows sent confirmation."""
        raw = {"content": ["Reply sent"]}
        display, _, _ = format_result("email.reply", raw)
        assert "Reply sent" in display

    def test_event_created(self):
        """Event creation shows summary."""
        raw = {"content": [json.dumps({
            "summary": "Team Lunch",
            "id": "evt_new",
        })]}
        display, _, _ = format_result("calendar.create", raw)
        assert "Event created" in display
        assert "Team Lunch" in display


class TestFormatGeneric:
    """Tests for generic/fallback formatting."""

    def test_unknown_command_shows_content(self):
        """Unknown commands format content as-is."""
        raw = {"content": ["Some server response"]}
        display, _, _ = format_result("unknown.command", raw)
        assert "Some server response" in display

    def test_empty_result(self):
        """Empty result shows 'Done'."""
        raw = {"content": []}
        display, _, _ = format_result("unknown.command", raw)
        assert display == "Done."
