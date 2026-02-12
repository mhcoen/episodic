"""Golden fixture tests for the extraction pipeline.

Gate and dispatchability tests run without LLM calls. Extraction tests
that need LLM calls are marked with @pytest.mark.llm (auto-applied to
tests with "llm" in the name by conftest).

Run LLM tests with: pytest -m llm tests/test_extraction/test_golden_fixtures.py
"""

import asyncio

import pytest

from episodic.mcp.extraction.extractor import check_dispatchability
from episodic.mcp.extraction.gate import matched_domains
from episodic.mcp.extraction.prompt import GSUITE_INTENTS
from episodic.mcp.extraction.types import ExtractionResult


# --- Fixture definitions ---

FIXTURES = [
    # --- True positives ---
    {
        "id": "tp_doctor_appointment",
        "utterance": "Do I have a doctor's appointment tomorrow?",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.query",
        "expected_args": {"query": "doctor appointment", "time_range": "tomorrow"},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    {
        "id": "tp_next_appointment",
        "utterance": "When's my next doctor appointment?",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.query",
        "expected_args": {"query": "doctor appointment", "time_range": "upcoming"},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    {
        "id": "tp_check_email",
        "utterance": "Check my email",
        "context": None,
        "expected_gate_domains": {"email"},
        "expected_intent": "email.search",
        "expected_args": {},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    {
        "id": "tp_unread_from_jane",
        "utterance": "Any unread emails from Jane?",
        "context": None,
        "expected_gate_domains": {"email"},
        "expected_intent": "email.search",
        "expected_args": {"from_addr": "jane", "unread_only": True},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    {
        "id": "tp_schedule_meeting",
        "utterance": "Schedule a meeting with Bob at 3pm tomorrow",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.create",
        "expected_args": {
            "summary": "meeting with Bob",
            "start": "3pm tomorrow",
            "attendees": ["bob"],
        },
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    {
        "id": "tp_am_i_free",
        "utterance": "Am I free Thursday afternoon?",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.query",
        "expected_args": {"time_range": "Thursday afternoon"},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    {
        "id": "tp_calendar_next_week",
        "utterance": "What's on my calendar next week?",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.query",
        "expected_args": {"time_range": "next week"},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    {
        "id": "tp_draft_email",
        "utterance": "Draft an email to Bob about the Q3 numbers",
        "context": None,
        "expected_gate_domains": {"email"},
        "expected_intent": "email.draft",
        "expected_args": {"to": "bob", "subject": "Q3 numbers"},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "true_positive",
    },
    # --- True negatives ---
    {
        "id": "tn_hate_appointments",
        "utterance": "I hate doctor appointments",
        "context": None,
        "expected_gate_domains": {"calendar"},  # gate fires, extraction returns null
        "expected_intent": None,
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": [],
        "category": "true_negative",
    },
    {
        "id": "tn_hate_meetings",
        "utterance": "I hate meetings",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": None,
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": [],
        "category": "true_negative",
    },
    {
        "id": "tn_email_annoying",
        "utterance": "Email is so annoying",
        "context": None,
        "expected_gate_domains": {"email"},
        "expected_intent": None,
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": [],
        "category": "true_negative",
    },
    {
        "id": "tn_should_schedule",
        "utterance": "I should schedule something eventually",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": None,
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": [],
        "category": "true_negative",
    },
    {
        "id": "tn_meetings_worst",
        "utterance": "Meetings are the worst part of my job",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": None,
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": [],
        "category": "true_negative",
    },
    # --- Missing required args ---
    {
        "id": "ma_draft_no_to",
        "utterance": "Draft an email",
        "context": None,
        "expected_gate_domains": {"email"},
        "expected_intent": "email.draft",
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": ["to"],
        "category": "missing_args",
    },
    {
        "id": "ma_schedule_no_summary",
        "utterance": "Schedule something",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.create",
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": ["summary"],
        "category": "missing_args",
    },
    # --- Anaphoric (require context) ---
    {
        "id": "ana_reply_first",
        "utterance": "Reply to the first one saying I'll have numbers Friday",
        "context": (
            "Last email search returned:\n"
            '  1. From: jane@company.com, Subject: "Budget Q3", Date: Feb 11\n'
            '  2. From: bob@company.com, Subject: "Meeting notes", Date: Feb 10'
        ),
        "expected_gate_domains": {"email"},
        "expected_intent": "email.reply",
        "expected_args": {"ref": "1", "body": "I'll have numbers Friday"},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "anaphoric",
    },
    {
        "id": "ana_read_from_bob",
        "utterance": "Read that email from Bob",
        "context": (
            "Last email search returned:\n"
            '  1. From: jane@company.com, Subject: "Budget Q3", Date: Feb 11\n'
            '  2. From: bob@company.com, Subject: "Meeting notes", Date: Feb 10'
        ),
        "expected_gate_domains": {"email"},
        "expected_intent": "email.read",
        "expected_args": {},  # ref varies by model
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "anaphoric",
    },
    # --- Adversarial formatting ---
    {
        "id": "adv_whitespace",
        "utterance": "  check   my   calendar  ",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.query",
        "expected_args": {},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "adversarial",
    },
    {
        "id": "adv_speech_artifacts",
        "utterance": "umm check my uhh email",
        "context": None,
        "expected_gate_domains": {"email"},
        "expected_intent": "email.search",
        "expected_args": {},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "adversarial",
    },
    {
        "id": "adv_all_caps",
        "utterance": "CHECK MY CALENDAR TOMORROW",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "calendar.query",
        "expected_args": {"time_range": "tomorrow"},
        "expected_dispatchable": True,
        "expected_missing_args": [],
        "category": "adversarial",
    },
    # --- Cross-domain ---
    {
        "id": "xd_email_about_meeting",
        "utterance": "Email Bob about tomorrow's meeting",
        "context": None,
        "expected_gate_domains": {"email", "calendar"},
        "expected_intent": "email.draft",
        "expected_args": {"to": "bob"},
        "expected_dispatchable": True,  # has required "to"
        "expected_missing_args": [],
        "category": "cross_domain",
    },
    # --- Unknown command ---
    {
        "id": "unk_delete_event",
        "utterance": "Delete that calendar event",
        "context": None,
        "expected_gate_domains": {"calendar"},
        "expected_intent": "router.unknown_command",
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": [],
        "category": "unknown_command",
    },
    {
        "id": "unk_forward_email",
        "utterance": "Forward the email to marketing",
        "context": None,
        "expected_gate_domains": {"email"},
        "expected_intent": "router.unknown_command",
        "expected_args": {},
        "expected_dispatchable": False,
        "expected_missing_args": [],
        "category": "unknown_command",
    },
]


# --- Gate tests (no LLM) ---


class TestGoldenGate:
    """Test keyword gate against golden fixtures."""

    @pytest.mark.parametrize(
        "fixture",
        FIXTURES,
        ids=[f["id"] for f in FIXTURES],
    )
    def test_gate_domains(self, fixture):
        domains = matched_domains(fixture["utterance"])
        expected = fixture["expected_gate_domains"]
        assert domains == expected, (
            f"Gate mismatch for '{fixture['utterance']}': "
            f"got {domains}, expected {expected}"
        )


# --- Dispatchability tests (no LLM) ---


class TestGoldenDispatchability:
    """Test dispatchability checker against golden fixtures.

    Uses the expected_intent and expected_args from fixtures to construct
    an ExtractionResult, then checks dispatchability.
    """

    @pytest.mark.parametrize(
        "fixture",
        [f for f in FIXTURES if f["expected_intent"] is not None],
        ids=[f["id"] for f in FIXTURES if f["expected_intent"] is not None],
    )
    def test_dispatchability(self, fixture):
        result = ExtractionResult(
            intent=fixture["expected_intent"],
            args=fixture["expected_args"],
            confidence=0.9,
            followup_suggestion=None,
            raw_json="",
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable == fixture["expected_dispatchable"], (
            f"Dispatchability mismatch for '{fixture['utterance']}': "
            f"got {verdict.dispatchable}, expected {fixture['expected_dispatchable']}"
        )
        for arg in fixture["expected_missing_args"]:
            assert arg in verdict.missing_required_args, (
                f"Expected missing arg '{arg}' for '{fixture['utterance']}'"
            )

    @pytest.mark.parametrize(
        "fixture",
        [f for f in FIXTURES if f["expected_intent"] is None],
        ids=[f["id"] for f in FIXTURES if f["expected_intent"] is None],
    )
    def test_null_intent_not_dispatchable(self, fixture):
        result = ExtractionResult(
            intent=None,
            args={},
            confidence=0.9,
            followup_suggestion=None,
            raw_json="",
        )
        verdict = check_dispatchability(result, GSUITE_INTENTS)
        assert verdict.dispatchable is False


# --- LLM extraction tests ---
# These are marked @pytest.mark.llm by conftest (test name contains "llm").
# Run with: pytest -m llm


class TestGoldenExtractionLlm:
    """Test full extraction pipeline including LLM call.

    These tests make real LLM calls and are marked with @pytest.mark.llm.
    Run explicitly with: pytest -m llm tests/test_extraction/test_golden_fixtures.py
    """

    @pytest.fixture(autouse=True)
    def _skip_unless_llm_selected(self, request):
        """Skip LLM tests unless explicitly selected with -m llm."""
        marker_expr = request.config.getoption("-m", default="")
        if "llm" not in marker_expr:
            pytest.skip("LLM tests - run with: pytest -m llm")

    @pytest.mark.parametrize(
        "fixture",
        [f for f in FIXTURES if f["category"] == "true_positive"],
        ids=[f["id"] for f in FIXTURES if f["category"] == "true_positive"],
    )
    def test_llm_true_positive(self, fixture):
        from episodic.mcp.extraction.extractor import extract_intent

        result = asyncio.run(extract_intent(
            utterance=fixture["utterance"],
            matched_domains=fixture["expected_gate_domains"],
            contacts={"bob": "bob@company.com", "jane": "jane@company.com"},
            recent_context=fixture["context"],
        ))
        assert result.intent == fixture["expected_intent"], (
            f"Intent mismatch for '{fixture['utterance']}': "
            f"got {result.intent}, expected {fixture['expected_intent']}"
        )

    @pytest.mark.parametrize(
        "fixture",
        [f for f in FIXTURES if f["category"] == "true_negative"],
        ids=[f["id"] for f in FIXTURES if f["category"] == "true_negative"],
    )
    def test_llm_true_negative(self, fixture):
        from episodic.mcp.extraction.extractor import extract_intent

        result = asyncio.run(extract_intent(
            utterance=fixture["utterance"],
            matched_domains=fixture["expected_gate_domains"],
            contacts={},
            recent_context=fixture["context"],
        ))
        assert result.intent is None, (
            f"Expected null intent for '{fixture['utterance']}', "
            f"got {result.intent}"
        )

    @pytest.mark.parametrize(
        "fixture",
        [f for f in FIXTURES if f["category"] == "unknown_command"],
        ids=[f["id"] for f in FIXTURES if f["category"] == "unknown_command"],
    )
    def test_llm_unknown_command(self, fixture):
        from episodic.mcp.extraction.extractor import extract_intent

        result = asyncio.run(extract_intent(
            utterance=fixture["utterance"],
            matched_domains=fixture["expected_gate_domains"],
            contacts={},
            recent_context=fixture["context"],
        ))
        assert result.intent == "router.unknown_command", (
            f"Expected router.unknown_command for '{fixture['utterance']}', "
            f"got {result.intent}"
        )
