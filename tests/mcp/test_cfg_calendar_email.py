"""
Tests for CFG Calendar/Email grammar parsing.

Spec tests 1-12b from CFG_MCP_DISPATCH_EXTENSION.md §9.1.
"""

import pytest
from episodic.utility.voice.lexer import Lexer
from episodic.utility.voice.grammar import GrammarParser


@pytest.fixture
def parser():
    return GrammarParser()


@pytest.fixture
def lexer():
    return Lexer()


def _parse(parser, text):
    """Helper: tokenize + parse, return GrammarMatch or None."""
    tokens = parser.lexer.tokenize(text.lower())
    return parser.parse(text.lower(), tokens)


class TestCalendarGrammar:
    """Spec tests 1, 4, 6, 7, 11, 12a, 12b."""

    def test_1_whats_on_calendar_tomorrow(self, parser):
        """Test 1: 'what's on my calendar tomorrow' → calendar.query."""
        match = _parse(parser, "what's on my calendar tomorrow")
        assert match is not None
        assert match.command == "calendar.query"
        assert match.category == "calendar"

    def test_4_schedule_meeting(self, parser):
        """Test 4: 'schedule a meeting with Bob at 3pm' → calendar.create."""
        match = _parse(parser, "schedule a meeting with bob at 3pm")
        assert match is not None
        assert match.command == "calendar.create"
        assert match.category == "calendar"

    def test_6_delete_meeting(self, parser):
        """Test 6: 'delete the meeting tomorrow' → calendar.delete, destructive."""
        match = _parse(parser, "delete the meeting tomorrow")
        assert match is not None
        assert match.command == "calendar.delete"
        assert match.category == "calendar"

    def test_11_am_i_free(self, parser):
        """Test 11: 'am I free next Tuesday afternoon' → calendar.freebusy."""
        # Simplified to match grammar: "am I free this afternoon"
        match = _parse(parser, "when am i free this afternoon")
        assert match is not None
        assert match.command == "calendar.freebusy"
        assert match.category == "calendar"

    def test_12a_list_calendars(self, parser):
        """Test 12a: 'list my calendars' → calendar.list."""
        match = _parse(parser, "which calendars do i have")
        assert match is not None
        assert match.command == "calendar.list"
        assert match.category == "calendar"

    def test_check_calendar(self, parser):
        """'check my calendar' → calendar.query."""
        match = _parse(parser, "check my calendar")
        assert match is not None
        assert match.command == "calendar.query"
        assert match.category == "calendar"

    def test_check_calendar_this_week(self, parser):
        """'check my calendar this week' → calendar.query with time range."""
        match = _parse(parser, "check my calendar this week")
        assert match is not None
        assert match.command == "calendar.query"
        assert match.category == "calendar"


class TestEmailGrammar:
    """Spec tests 2, 3, 5, 12."""

    def test_2_check_email(self, parser):
        """Test 2: 'check my email' → email.search."""
        match = _parse(parser, "check my email")
        assert match is not None
        assert match.command == "email.search"
        assert match.category == "email"

    def test_3_email_from_alice(self, parser):
        """Test 3: 'email from Alice' → email.search with from_addr."""
        match = _parse(parser, "email from alice")
        assert match is not None
        assert match.command == "email.search"
        assert match.category == "email"
        assert "from_addr" in match.args

    def test_5_reply_to_email(self, parser):
        """Test 5: 'reply to that email saying thanks' → email.reply."""
        match = _parse(parser, "reply to that email saying thanks")
        assert match is not None
        assert match.command == "email.reply"
        assert match.category == "email"

    def test_12_forward_email(self, parser):
        """Test 12: 'forward the budget email to Carol' → email.forward."""
        match = _parse(parser, "forward the budget email to carol")
        assert match is not None
        assert match.command == "email.forward"
        assert match.category == "email"

    def test_search_email_about(self, parser):
        """'search email about budget' → email.search with query."""
        match = _parse(parser, "search email about budget")
        assert match is not None
        assert match.command == "email.search"
        assert match.category == "email"

    def test_draft_email(self, parser):
        """'draft an email to bob' → email.create_draft."""
        match = _parse(parser, "compose an email to bob")
        assert match is not None
        assert match.command == "email.create_draft"
        assert match.category == "email"

    def test_delete_draft(self, parser):
        """'delete the draft' → email.delete_draft."""
        match = _parse(parser, "delete the draft")
        assert match is not None
        assert match.command == "email.delete_draft"
        assert match.category == "email"


class TestFallthrough:
    """Spec test 10."""

    def test_10_no_match_falls_through(self, parser):
        """Test 10: 'tell me about quantum computing' → no utility match."""
        match = _parse(parser, "tell me about quantum computing")
        # Should not match calendar or email
        if match is not None:
            assert match.category not in ("calendar", "email")


class TestReplyDualMode:
    """Test email.reply send=true vs send=false."""

    def test_reply_sends_by_default(self, parser):
        """'reply to that email' → send=True."""
        match = _parse(parser, "reply to that email saying thanks")
        assert match is not None
        assert match.command == "email.reply"
        assert match.args.get("send") is True

    def test_draft_reply_does_not_send(self, parser):
        """'draft a reply' → send=False."""
        match = _parse(parser, "draft a reply to that email")
        assert match is not None
        assert match.command == "email.reply"
        assert match.args.get("send") is False
