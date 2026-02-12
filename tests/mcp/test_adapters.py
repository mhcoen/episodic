"""
Tests for MCP Argument Adapters and CFG Directive Adapter.

Spec tests 17-20d from CFG_MCP_DISPATCH_EXTENSION.md §9.3.
"""

import pytest
from episodic.mcp.dispatch_types import MCPResolution
from episodic.mcp.adapters import ARGUMENT_ADAPTERS
from episodic.mcp.adapters.argument_adapters.calendar import (
    CalendarQueryAdapter,
    CalendarListAdapter,
)
from episodic.mcp.adapters.argument_adapters.email import (
    EmailSearchAdapter,
    EmailReplyAdapter,
    EmailForwardAdapter,
)
from episodic.mcp.adapters.cfg_directive import CFGDirectiveAdapter
from episodic.utility.types import UtilityQuery


@pytest.fixture
def cal_resolution():
    return MCPResolution(
        server_id="gsuite",
        tool_name="get_calendar_events",
        sensitivity="read",
        requires_auth_event=False,
    )


@pytest.fixture
def email_resolution():
    return MCPResolution(
        server_id="gsuite",
        tool_name="query_gmail_emails",
        sensitivity="read",
        requires_auth_event=False,
    )


@pytest.fixture
def reply_resolution():
    return MCPResolution(
        server_id="gsuite",
        tool_name="reply_gmail_email",
        sensitivity="write",
        requires_auth_event=True,
    )


class TestCalendarAdapters:
    """Spec tests 17, 19, 20a."""

    def test_17_calendar_query_iso_timestamps(self, cal_resolution):
        """Test 17: CalendarQueryAdapter produces ISO timestamps."""
        adapter = CalendarQueryAdapter()
        result = adapter.adapt(
            {"time_min": "2026-02-13T00:00:00-06:00",
             "time_max": "2026-02-14T00:00:00-06:00"},
            cal_resolution,
        )
        assert "time_min" in result
        assert "time_max" in result
        assert "2026-02-13" in result["time_min"]
        assert "2026-02-14" in result["time_max"]

    def test_19_user_id_injection(self, cal_resolution):
        """Test 19: __user_id__ is injected from config."""
        adapter = CalendarQueryAdapter()
        result = adapter.adapt({}, cal_resolution)
        assert "__user_id__" in result
        assert "@" in result["__user_id__"]

    def test_19_user_id_from_config(self, cal_resolution):
        """Test 19: __user_id__ uses config default_account."""
        adapter = CalendarQueryAdapter()
        result = adapter.adapt(
            {},
            cal_resolution,
            config={"default_account": "test@example.com"},
        )
        assert result["__user_id__"] == "test@example.com"

    def test_20_missing_optional_args(self, cal_resolution):
        """Test 20: Missing optional args don't crash."""
        adapter = CalendarQueryAdapter()
        result = adapter.adapt({}, cal_resolution)
        # Should have __user_id__ but no time fields
        assert "__user_id__" in result
        assert "time_min" not in result
        assert "time_max" not in result

    def test_20a_calendar_id_default_omission(self, cal_resolution):
        """Test 20a: __calendar_id__ only present when user specifies."""
        adapter = CalendarQueryAdapter()
        result = adapter.adapt({}, cal_resolution)
        assert "__calendar_id__" not in result

        result_with_cal = adapter.adapt(
            {"calendar_id": "work"}, cal_resolution
        )
        assert result_with_cal["__calendar_id__"] == "work"


class TestEmailAdapters:
    """Spec tests 18, 20b, 20c, 20d."""

    def test_18_email_search_query_string(self, email_resolution):
        """Test 18: EmailSearchAdapter builds Gmail query string."""
        adapter = EmailSearchAdapter()
        result = adapter.adapt(
            {"from_addr": "alice@example.com", "unread_only": True},
            email_resolution,
        )
        assert "from:alice@example.com" in result["query"]
        assert "is:unread" in result["query"]

    def test_18_email_search_default_query(self, email_resolution):
        """Default query when no args specified."""
        adapter = EmailSearchAdapter()
        result = adapter.adapt({}, email_resolution)
        assert result["query"] == "is:unread"

    def test_20b_reply_send_true(self, reply_resolution):
        """Test 20b: EmailReplyAdapter send=true for 'reply to'."""
        adapter = EmailReplyAdapter()
        result = adapter.adapt(
            {"email_ref": "abc123", "body": "thanks", "send": True},
            reply_resolution,
        )
        assert result["send"] is True
        assert result["email_id"] == "abc123"

    def test_20c_reply_send_false(self, reply_resolution):
        """Test 20c: EmailReplyAdapter send=false for 'draft a reply'."""
        adapter = EmailReplyAdapter()
        result = adapter.adapt(
            {"email_ref": "abc123", "body": "thanks", "send": False},
            reply_resolution,
        )
        assert result["send"] is False

    def test_20d_forward_creates_draft_args(self):
        """Test 20d: EmailForwardAdapter produces create_gmail_draft args."""
        resolution = MCPResolution(
            server_id="gsuite",
            tool_name="create_gmail_draft",
            sensitivity="write",
            requires_auth_event=True,
        )
        adapter = EmailForwardAdapter()
        result = adapter.adapt(
            {
                "to": "carol@example.com",
                "original_subject": "Budget Report",
                "original_body": "Here are the numbers...",
            },
            resolution,
        )
        assert result["to"] == "carol@example.com"
        assert result["subject"].startswith("[Fwd]")
        assert "Budget Report" in result["subject"]
        assert "Forwarded message" in result["body"]
        assert "Here are the numbers" in result["body"]


class TestCFGDirectiveAdapter:
    """Tests for AuthorizationEvent production."""

    def test_read_intent_no_auth_event(self):
        """Read intents produce no auth event."""
        adapter = CFGDirectiveAdapter()
        query = UtilityQuery(
            category="email", command="email.search",
            args={}, confidence=0.9, source="cli",
            raw_input="check my email",
        )
        event = adapter.produce(query, "check my email")
        assert event is None

    def test_write_intent_produces_auth_event(self):
        """Write intents produce an auth event."""
        adapter = CFGDirectiveAdapter()
        query = UtilityQuery(
            category="email", command="email.reply",
            args={"to": "alice@example.com", "body": "thanks"},
            confidence=0.9, source="cli",
            raw_input="reply to alice saying thanks",
        )
        event = adapter.produce(query, "reply to alice saying thanks")
        assert event is not None
        assert event.action == "email.reply"
        assert event.scope["action"] == "email.reply"
        assert event.scope["recipient"] == "alice@example.com"
        assert event.source == "cfg_parser"

    def test_destructive_intent_produces_auth_event(self):
        """Destructive intents produce auth events."""
        adapter = CFGDirectiveAdapter()
        query = UtilityQuery(
            category="calendar", command="calendar.delete",
            args={"event_ref": "standup"}, confidence=0.9,
            source="cli", raw_input="delete the standup",
        )
        event = adapter.produce(query, "delete the standup")
        assert event is not None
        assert event.action == "calendar.delete"


class TestAdapterRegistry:
    """Tests for the adapter registry."""

    def test_all_intents_have_adapters(self):
        """Every intent in default mapping has an adapter."""
        from episodic.mcp.dispatch_types import DEFAULT_INTENT_MAPPING
        for intent in DEFAULT_INTENT_MAPPING:
            assert intent in ARGUMENT_ADAPTERS, f"Missing adapter for {intent}"

    def test_adapter_classes_instantiate(self):
        """All adapter classes can be instantiated."""
        for intent, cls in ARGUMENT_ADAPTERS.items():
            adapter = cls()
            assert hasattr(adapter, "adapt"), f"{intent} adapter missing adapt()"
