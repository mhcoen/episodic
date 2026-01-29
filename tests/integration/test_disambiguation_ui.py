"""
Integration tests for disambiguation UI.

Tests the three main paths:
1. User selects a topic -> reactivate
2. User selects 0 -> continue
3. Invalid input handling
"""

import pytest

from episodic.recall.reactivation import DisambiguationOption
from episodic.ui.disambiguation import (
    DisambiguationResult,
    format_disambiguation_options,
    handle_disambiguation_input,
)


def create_option(
    name: str,
    topic_id: str,
    turns_ago: int = 10,
    support_count: int = 3,
    snippets: list = None,
) -> DisambiguationOption:
    """Helper to create test disambiguation options."""
    return DisambiguationOption(
        topic_name=name,
        topic_start_node_id=topic_id,
        similarity=0.8,
        support_count=support_count,
        preview=f"Preview for {name}",
        turns_ago=turns_ago,
        snippets=snippets or [f"Sample snippet from {name}"],
    )


class TestDisambiguationInput:
    """Tests for handle_disambiguation_input."""

    def test_select_first_option(self):
        """User selects option 1 -> reactivates that topic."""
        options = [
            create_option("python-debugging", "node_001"),
            create_option("coffee-brewing", "node_002"),
        ]

        result = handle_disambiguation_input("1", options)

        assert result.action == "reactivate"
        assert result.topic_start_node_id == "node_001"
        assert result.topic_name == "python-debugging"

    def test_select_second_option(self):
        """User selects option 2 -> reactivates that topic."""
        options = [
            create_option("python-debugging", "node_001"),
            create_option("coffee-brewing", "node_002"),
        ]

        result = handle_disambiguation_input("2", options)

        assert result.action == "reactivate"
        assert result.topic_start_node_id == "node_002"
        assert result.topic_name == "coffee-brewing"

    def test_select_zero_continues(self):
        """User selects 0 -> continues current topic."""
        options = [
            create_option("python-debugging", "node_001"),
            create_option("coffee-brewing", "node_002"),
        ]

        result = handle_disambiguation_input("0", options)

        assert result.action == "continue"
        assert result.topic_start_node_id is None
        assert result.topic_name is None

    def test_invalid_then_valid(self):
        """Invalid input on attempt 1 returns reprompt."""
        options = [create_option("python-debugging", "node_001")]

        # First attempt - invalid
        result = handle_disambiguation_input("xyz", options, attempt=1)
        assert result.action == "reprompt"

        # Second attempt - valid
        result = handle_disambiguation_input("1", options, attempt=2)
        assert result.action == "reactivate"
        assert result.topic_start_node_id == "node_001"

    def test_invalid_twice_skips(self):
        """Two invalid inputs -> skip to continue."""
        options = [create_option("python-debugging", "node_001")]

        # First attempt - invalid
        result = handle_disambiguation_input("abc", options, attempt=1)
        assert result.action == "reprompt"

        # Second attempt - still invalid -> auto-continue
        result = handle_disambiguation_input("xyz", options, attempt=2)
        assert result.action == "continue"
        assert result.topic_start_node_id is None

    def test_out_of_range_number(self):
        """Number out of range treated as invalid."""
        options = [create_option("python-debugging", "node_001")]

        # 5 is out of range (only option 1 is valid)
        result = handle_disambiguation_input("5", options, attempt=1)
        assert result.action == "reprompt"

    def test_negative_number_invalid(self):
        """Negative numbers are invalid."""
        options = [create_option("python-debugging", "node_001")]

        result = handle_disambiguation_input("-1", options, attempt=1)
        assert result.action == "reprompt"

    def test_empty_input_invalid(self):
        """Empty input is invalid."""
        options = [create_option("python-debugging", "node_001")]

        result = handle_disambiguation_input("", options, attempt=1)
        assert result.action == "reprompt"

    def test_whitespace_stripped(self):
        """Whitespace is stripped from input."""
        options = [create_option("python-debugging", "node_001")]

        result = handle_disambiguation_input("  1  ", options)
        assert result.action == "reactivate"
        assert result.topic_start_node_id == "node_001"


class TestDisambiguationFormat:
    """Tests for format_disambiguation_options."""

    def test_shows_topic_name(self):
        """Formatted output includes topic name."""
        options = [
            create_option("python-debugging", "node_001", turns_ago=12),
        ]

        output = format_disambiguation_options(options)

        assert "python-debugging" in output

    def test_shows_turns_ago(self):
        """Formatted output includes turns ago."""
        options = [
            create_option("python-debugging", "node_001", turns_ago=12),
        ]

        output = format_disambiguation_options(options)

        assert "12 turns ago" in output

    def test_shows_snippets(self):
        """Formatted output includes evidence snippets."""
        options = [
            create_option(
                "python-debugging",
                "node_001",
                snippets=["How do I fix IndexError?", "What about try-except?"],
            ),
        ]

        output = format_disambiguation_options(options)

        assert "IndexError" in output

    def test_shows_support_count(self):
        """Formatted output includes hit count."""
        options = [
            create_option("python-debugging", "node_001", support_count=3),
        ]

        output = format_disambiguation_options(options)

        assert "3 matching exchanges" in output

    def test_shows_continue_option(self):
        """Formatted output includes option 0 to continue."""
        options = [
            create_option("python-debugging", "node_001"),
        ]

        output = format_disambiguation_options(options)

        assert "[0]" in output
        assert "Neither" in output or "Continue" in output

    def test_limits_to_three_options(self):
        """Only shows up to 3 options."""
        options = [
            create_option("topic-1", "node_001"),
            create_option("topic-2", "node_002"),
            create_option("topic-3", "node_003"),
            create_option("topic-4", "node_004"),
        ]

        output = format_disambiguation_options(options)

        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output
        assert "[4]" not in output

    def test_uses_preview_when_no_snippets(self):
        """Falls back to preview when no snippets available."""
        options = [
            DisambiguationOption(
                topic_name="python-debugging",
                topic_start_node_id="node_001",
                similarity=0.8,
                support_count=3,
                preview="This is the preview text",
                turns_ago=10,
                snippets=[],  # Empty snippets
            )
        ]

        output = format_disambiguation_options(options)

        assert "This is the preview text" in output

    def test_numbered_options(self):
        """Options are numbered 1, 2, 3."""
        options = [
            create_option("topic-1", "node_001"),
            create_option("topic-2", "node_002"),
        ]

        output = format_disambiguation_options(options)

        assert "[1] topic-1" in output
        assert "[2] topic-2" in output


class TestDisambiguationResult:
    """Tests for DisambiguationResult dataclass."""

    def test_reactivate_result(self):
        """Test reactivate result creation."""
        result = DisambiguationResult(
            action="reactivate",
            topic_start_node_id="node_001",
            topic_name="python",
        )

        assert result.action == "reactivate"
        assert result.topic_start_node_id == "node_001"
        assert result.topic_name == "python"

    def test_continue_result(self):
        """Test continue result creation."""
        result = DisambiguationResult(action="continue")

        assert result.action == "continue"
        assert result.topic_start_node_id is None
        assert result.topic_name is None

    def test_reprompt_result(self):
        """Test reprompt result creation."""
        result = DisambiguationResult(action="reprompt")

        assert result.action == "reprompt"
        assert result.topic_start_node_id is None
        assert result.topic_name is None
