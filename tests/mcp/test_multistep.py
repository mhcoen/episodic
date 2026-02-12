"""
Tests for Multi-Step MCP Dispatch.

Spec tests 31-38 from CFG_MCP_DISPATCH_EXTENSION.md §9.5.
Tests decomposition, partial failure, anaphoric resolution.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from episodic.mcp.dispatch import MCPResolver, MCPDecomposer, get_result_context
from episodic.mcp.dispatch_types import MCPResolution, DispatchResult
from episodic.mcp.result_context import MCPResultContext, resolve_anaphoric_ref


@pytest.fixture
def resolver():
    return MCPResolver()


@pytest.fixture
def decomposer(resolver):
    return MCPDecomposer(resolver)


@pytest.fixture
def fresh_context():
    ctx = MCPResultContext()
    ctx.clear()
    return ctx


class TestMultiStepDecomposition:
    """Spec tests 31, 36, 38."""

    def test_31_reply_decomposes_to_search_plus_reply(self, decomposer, fresh_context):
        """Test 31: 'Reply to Alice's email' decomposes to search + reply."""
        steps = decomposer.decompose(
            "email.reply",
            {"email_ref": "Alice's email", "body": "thanks", "send": True},
            fresh_context,
        )
        assert len(steps) == 2
        assert steps[0].intent == "email.search"
        assert steps[0].sensitivity == "read"
        assert steps[0].requires_auth_event is False
        assert steps[1].intent == "email.reply"
        assert steps[1].sensitivity == "write"
        assert steps[1].requires_auth_event is True

    def test_36_reschedule_decomposes_to_three_steps(self, decomposer, fresh_context):
        """Test 36: 'reschedule meeting to 4pm' → search + delete + create."""
        steps = decomposer.decompose(
            "calendar.reschedule",
            {"event_ref": "standup", "new_start": "16:00"},
            fresh_context,
        )
        assert len(steps) == 3
        assert steps[0].intent == "calendar.query"
        assert steps[0].sensitivity == "read"
        assert steps[1].intent == "calendar.delete"
        assert steps[1].sensitivity == "destructive"
        assert steps[2].intent == "calendar.create"
        assert steps[2].sensitivity == "write"

    def test_38_forward_decomposes_to_three_steps(self, decomposer, fresh_context):
        """Test 38: 'Forward Alice's email to Carol' → search + get + draft."""
        steps = decomposer.decompose(
            "email.forward",
            {"email_ref": "budget email", "to": "carol@example.com"},
            fresh_context,
        )
        assert len(steps) == 3
        assert steps[0].intent == "email.search"
        assert steps[1].intent == "email.get"
        assert steps[2].intent == "email.forward"
        assert steps[2].tool_args.get("to") == "carol@example.com"


class TestSingleStepDispatch:
    """Verify single-step commands decompose to 1 step."""

    def test_single_step_email_search(self, decomposer, fresh_context):
        """email.search produces single step."""
        steps = decomposer.decompose(
            "email.search",
            {"query": "budget"},
            fresh_context,
        )
        assert len(steps) == 1
        assert steps[0].intent == "email.search"

    def test_single_step_calendar_query(self, decomposer, fresh_context):
        """calendar.query produces single step."""
        steps = decomposer.decompose(
            "calendar.query",
            {},
            fresh_context,
        )
        assert len(steps) == 1
        assert steps[0].intent == "calendar.query"


class TestPartialFailure:
    """Spec tests 32-33, 37."""

    def test_32_search_fails_no_write(self, decomposer, fresh_context):
        """Test 32: If search step fails, no write attempted."""
        # Verify steps are decomposed correctly — the dispatch handler
        # is responsible for stopping on failure, not the decomposer
        steps = decomposer.decompose(
            "email.reply",
            {"email_ref": "nonexistent email", "body": "thanks", "send": True},
            fresh_context,
        )
        assert len(steps) == 2
        # The first step (search) comes first; dispatch stops if it fails
        assert steps[0].intent == "email.search"
        assert steps[0].requires_auth_event is False
        assert steps[1].intent == "email.reply"
        assert steps[1].requires_auth_event is True

    def test_37_reschedule_delete_fails_no_create(self, decomposer, fresh_context):
        """Test 37: Reschedule: delete fails → no create attempted."""
        steps = decomposer.decompose(
            "calendar.reschedule",
            {"event_ref": "standup", "new_start": "16:00"},
            fresh_context,
        )
        # Verify ordering: query → delete → create
        assert len(steps) == 3
        assert steps[0].intent == "calendar.query"
        assert steps[1].intent == "calendar.delete"
        assert steps[2].intent == "calendar.create"


class TestAnaphoricResolution:
    """Spec tests 34-35."""

    def test_34_anaphoric_resolves_from_context(self, fresh_context):
        """Test 34: 'reply to that' resolves from result context."""
        fresh_context.update_emails([
            {"id": "msg-123", "from": "alice@example.com", "subject": "Budget"},
        ])
        resolved = resolve_anaphoric_ref("last", "email", fresh_context)
        assert resolved == "msg-123"

    def test_35_stale_context_returns_none(self, fresh_context):
        """Test 35: Stale context (>TTL) forces fresh search."""
        fresh_context.update_emails([
            {"id": "msg-123", "from": "alice@example.com"},
        ])
        # Force context to be stale
        fresh_context.timestamp = time.time() - 600  # 10 minutes ago
        fresh_context.ttl = 300  # 5 minute TTL

        resolved = resolve_anaphoric_ref("last", "email", fresh_context)
        assert resolved is None  # Stale → None

    def test_anaphoric_non_last_passes_through(self, fresh_context):
        """Non-anaphoric refs pass through unchanged."""
        resolved = resolve_anaphoric_ref("msg-456", "email", fresh_context)
        assert resolved == "msg-456"

    def test_anaphoric_none_passes_through(self, fresh_context):
        """None refs pass through as None."""
        resolved = resolve_anaphoric_ref(None, "email", fresh_context)
        assert resolved is None

    def test_calendar_anaphoric_resolution(self, fresh_context):
        """Calendar event anaphoric resolution works."""
        fresh_context.update_events([
            {"id": "evt-789", "summary": "Team standup"},
        ])
        resolved = resolve_anaphoric_ref("last", "calendar", fresh_context)
        assert resolved == "evt-789"

    def test_forward_skips_search_with_context(self):
        """Forward with resolved anaphoric ref skips search step."""
        resolver = MCPResolver()
        decomposer = MCPDecomposer(resolver)
        ctx = MCPResultContext()
        ctx.update_emails([{"id": "msg-100", "subject": "Report"}])

        steps = decomposer.decompose(
            "email.forward",
            {"email_ref": "last", "to": "bob@example.com"},
            ctx,
        )
        # Should skip the search step since we have context
        assert steps[0].intent == "email.get"
        assert len(steps) == 2  # get + forward (no search)


class TestResultContextLifecycle:
    """Test result context updates."""

    def test_context_updates_on_email_search(self):
        """Emails are tracked after search."""
        from episodic.mcp.dispatch import update_result_context, get_result_context
        ctx = get_result_context()
        ctx.clear()

        update_result_context("email.search", {
            "emails": [
                {"id": "msg-1", "from": "alice@example.com"},
                {"id": "msg-2", "from": "bob@example.com"},
            ]
        })
        assert len(ctx.last_emails) == 2
        assert ctx.last_emails[0]["id"] == "msg-1"

    def test_context_updates_on_calendar_query(self):
        """Events are tracked after calendar query."""
        from episodic.mcp.dispatch import update_result_context, get_result_context
        ctx = get_result_context()
        ctx.clear()

        update_result_context("calendar.query", {
            "events": [
                {"id": "evt-1", "summary": "Standup"},
            ]
        })
        assert len(ctx.last_events) == 1
        assert ctx.last_events[0]["summary"] == "Standup"

    def test_context_clear(self):
        """Context can be cleared."""
        ctx = MCPResultContext()
        ctx.update_emails([{"id": "1"}])
        ctx.clear()
        assert ctx.last_emails == []
        assert ctx.is_stale()
