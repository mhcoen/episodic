"""Tests for dispatchability checker (episodic.mcp.extraction.extractor)."""

import pytest

from episodic.mcp.extraction.extractor import check_dispatchability
from episodic.mcp.extraction.prompt import GSUITE_INTENTS
from episodic.mcp.extraction.types import ExtractionResult


def _make_result(
    intent=None, args=None, confidence=0.9, followup=None, raw=""
) -> ExtractionResult:
    return ExtractionResult(
        intent=intent,
        args=args or {},
        confidence=confidence,
        followup_suggestion=followup,
        raw_json=raw,
    )


class TestDispatchability:
    def test_valid_extraction_dispatchable(self):
        result = _make_result(
            intent="calendar.query",
            args={"query": "doctor", "time_range": "tomorrow"},
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is True
        assert verdict.intent == "calendar.query"
        assert verdict.action_class == "read"
        assert verdict.missing_required_args == []
        assert verdict.error is None

    def test_null_intent_not_dispatchable(self):
        result = _make_result(intent=None)
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is False
        assert verdict.intent is None
        assert verdict.error is None  # null intent is not an error

    def test_unregistered_intent_error(self):
        result = _make_result(intent="slack.post")
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is False
        assert "Unregistered intent" in verdict.error

    def test_missing_required_arg(self):
        # email.draft requires "to"
        result = _make_result(
            intent="email.draft",
            args={"subject": "Q3 numbers"},
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is False
        assert "to" in verdict.missing_required_args

    def test_empty_string_required_arg(self):
        result = _make_result(
            intent="email.draft",
            args={"to": "", "subject": "Q3"},
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is False
        assert "to" in verdict.missing_required_args

    def test_wrong_arg_type_stripped(self):
        # unread_only should be boolean, pass string instead
        result = _make_result(
            intent="email.search",
            args={"unread_only": "yes"},  # should be bool
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is True  # no required args for email.search
        assert "unread_only" not in verdict.args  # stripped due to wrong type

    def test_router_unknown_command(self):
        result = _make_result(
            intent="router.unknown_command",
            args={"hint": "user wants to delete a calendar event"},
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is False
        assert verdict.is_unknown_command is True
        assert verdict.unknown_command_hint == "user wants to delete a calendar event"
        assert verdict.error is None

    def test_followup_suggestion_passthrough(self):
        result = _make_result(
            intent="email.search",
            args={},
            followup="also check calendar for tomorrow",
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.followup_suggestion == "also check calendar for tomorrow"

    def test_all_required_args_present_dispatchable(self):
        # email.draft with required "to" present
        result = _make_result(
            intent="email.draft",
            args={"to": "bob", "subject": "Q3"},
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is True
        assert verdict.action_class == "draft"
        assert verdict.missing_required_args == []

    def test_calendar_create_missing_summary(self):
        # calendar.create requires "summary"
        result = _make_result(
            intent="calendar.create",
            args={"start": "3pm tomorrow"},
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is False
        assert "summary" in verdict.missing_required_args
