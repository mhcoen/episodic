"""
Tests for correction detection and resolution.

Tests the best-guess-then-correction flow for topic disambiguation.
"""

import pytest
from dataclasses import field

from episodic.recall.correction import (
    CorrectionState,
    CORRECTION_PATTERNS,
    detect_correction,
    resolve_correction,
)
from episodic.recall.reactivation import DisambiguationOption


# ============================================================================
# detect_correction tests
# ============================================================================

class TestDetectCorrection:
    """Tests for detect_correction() pattern matching."""

    def test_direct_negation_no(self):
        """'no' at start triggers correction."""
        is_correction, hint = detect_correction("no")
        assert is_correction is True
        assert hint is None

    def test_direct_negation_no_with_text(self):
        """'no' at start with more text triggers correction."""
        is_correction, hint = detect_correction("no I don't want that")
        assert is_correction is True
        assert hint is None

    def test_direct_negation_nope(self):
        """'nope' at start triggers correction."""
        is_correction, hint = detect_correction("nope")
        assert is_correction is True
        assert hint is None

    def test_not_that(self):
        """'not that' triggers correction."""
        is_correction, hint = detect_correction("that's not that topic")
        assert is_correction is True
        assert hint is None

    def test_wrong_one(self):
        """'wrong one' triggers correction."""
        is_correction, hint = detect_correction("that's the wrong one")
        assert is_correction is True
        assert hint is None

    def test_wrong_topic(self):
        """'wrong topic' triggers correction."""
        is_correction, hint = detect_correction("wrong topic")
        assert is_correction is True
        assert hint is None

    def test_the_other_one(self):
        """'the other one' triggers correction."""
        is_correction, hint = detect_correction("give me the other one")
        assert is_correction is True
        assert hint is None

    def test_the_second_one(self):
        """'the second one' triggers correction."""
        is_correction, hint = detect_correction("the second one please")
        assert is_correction is True
        assert hint is None

    def test_the_different_one(self):
        """'the different one' triggers correction."""
        is_correction, hint = detect_correction("the different one")
        assert is_correction is True
        assert hint is None

    def test_no_the_coffee_one(self):
        """'no, the coffee one' captures hint."""
        is_correction, hint = detect_correction("no, the coffee one")
        assert is_correction is True
        assert hint == "coffee"

    def test_no_coffee_one(self):
        """'no coffee one' captures hint."""
        is_correction, hint = detect_correction("no the python one")
        assert is_correction is True
        assert hint == "python"

    def test_i_meant(self):
        """'I meant X' captures hint."""
        is_correction, hint = detect_correction("I meant the coffee discussion")
        assert is_correction is True
        assert hint == "the coffee discussion"

    def test_no_about(self):
        """'no, about X' captures hint."""
        is_correction, hint = detect_correction("no, about database optimization")
        assert is_correction is True
        assert hint == "database optimization"

    def test_no_correction_normal_input(self):
        """Normal input doesn't trigger correction."""
        is_correction, hint = detect_correction("Tell me more about Python")
        assert is_correction is False
        assert hint is None

    def test_no_correction_question(self):
        """Question doesn't trigger correction."""
        is_correction, hint = detect_correction("What is the best way to do this?")
        assert is_correction is False
        assert hint is None

    def test_no_in_middle_not_correction(self):
        """'no' in middle of sentence isn't correction."""
        is_correction, hint = detect_correction("I have no idea what to do")
        assert is_correction is False
        assert hint is None

    def test_case_insensitive(self):
        """Pattern matching is case insensitive (hint is lowercased)."""
        is_correction, hint = detect_correction("NO, THE COFFEE ONE")
        assert is_correction is True
        # Hint is lowercase because we lowercase input before matching
        assert hint == "coffee"


# ============================================================================
# resolve_correction tests
# ============================================================================

def make_option(name: str, snippets: list[str] = None) -> DisambiguationOption:
    """Helper to create DisambiguationOption for tests."""
    return DisambiguationOption(
        topic_name=name,
        topic_start_node_id=f"node_{name.lower().replace(' ', '_')}",
        similarity=0.8,
        support_count=3,
        preview=f"Preview of {name}",
        turns_ago=5,
        snippets=snippets or [],
    )


class TestResolveCorrection:
    """Tests for resolve_correction() hint matching."""

    def test_no_runner_ups_returns_none(self):
        """No runner-ups returns None."""
        state = CorrectionState(
            query="test query",
            chosen_option=make_option("Python Programming"),
            runner_ups=[],
            turn_created=10,
        )
        result = resolve_correction(state, None)
        assert result is None

    def test_no_hint_returns_first_runner_up(self):
        """No hint returns first runner-up."""
        state = CorrectionState(
            query="test query",
            chosen_option=make_option("Python Programming"),
            runner_ups=[
                make_option("Coffee Brewing"),
                make_option("Database Design"),
            ],
            turn_created=10,
        )
        result = resolve_correction(state, None)
        assert result is not None
        assert result.topic_name == "Coffee Brewing"

    def test_hint_matches_topic_name_substring(self):
        """Hint that matches topic name substring."""
        state = CorrectionState(
            query="test query",
            chosen_option=make_option("Python Programming"),
            runner_ups=[
                make_option("Coffee Brewing"),
                make_option("Database Design"),
            ],
            turn_created=10,
        )
        result = resolve_correction(state, "database")
        assert result is not None
        assert result.topic_name == "Database Design"

    def test_hint_matches_word_in_topic_name(self):
        """Hint with word matching in topic name."""
        state = CorrectionState(
            query="test query",
            chosen_option=make_option("Python Programming"),
            runner_ups=[
                make_option("Morning Coffee"),
                make_option("Database Schema"),
            ],
            turn_created=10,
        )
        result = resolve_correction(state, "coffee")
        assert result is not None
        assert result.topic_name == "Morning Coffee"

    def test_hint_matches_snippet(self):
        """Hint that matches a snippet."""
        state = CorrectionState(
            query="test query",
            chosen_option=make_option("Python Programming"),
            runner_ups=[
                make_option("Topic A", snippets=["discussing espresso machines"]),
                make_option("Topic B", snippets=["database indexes"]),
            ],
            turn_created=10,
        )
        result = resolve_correction(state, "espresso")
        assert result is not None
        assert result.topic_name == "Topic A"

    def test_hint_no_match_returns_first(self):
        """Hint with no match returns first runner-up."""
        state = CorrectionState(
            query="test query",
            chosen_option=make_option("Python Programming"),
            runner_ups=[
                make_option("Coffee Brewing"),
                make_option("Database Design"),
            ],
            turn_created=10,
        )
        result = resolve_correction(state, "xyz123")
        assert result is not None
        assert result.topic_name == "Coffee Brewing"

    def test_case_insensitive_matching(self):
        """Hint matching is case insensitive."""
        state = CorrectionState(
            query="test query",
            chosen_option=make_option("Python Programming"),
            runner_ups=[
                make_option("COFFEE Brewing"),
                make_option("Database Design"),
            ],
            turn_created=10,
        )
        result = resolve_correction(state, "coffee")
        assert result is not None
        assert result.topic_name == "COFFEE Brewing"


# ============================================================================
# CorrectionState tests
# ============================================================================

class TestCorrectionState:
    """Tests for CorrectionState dataclass."""

    def test_state_creation(self):
        """State can be created with all fields."""
        chosen = make_option("Main Topic")
        runner_ups = [make_option("Alt 1"), make_option("Alt 2")]

        state = CorrectionState(
            query="original query",
            chosen_option=chosen,
            runner_ups=runner_ups,
            turn_created=42,
        )

        assert state.query == "original query"
        assert state.chosen_option.topic_name == "Main Topic"
        assert len(state.runner_ups) == 2
        assert state.turn_created == 42
