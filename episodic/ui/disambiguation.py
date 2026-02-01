"""
Disambiguation UI for topic selection.

When multiple topics match a user's input, this module provides
formatted display and input handling for the disambiguation prompt.

Also provides conversational hints for best-guess-then-correction flow.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from episodic.recall.reactivation import DisambiguationOption


def shorten_topic_name(option: DisambiguationOption, max_words: int = 3) -> str:
    """
    Create short label from topic_name, truncating if needed.

    Args:
        option: The disambiguation option
        max_words: Maximum words to include

    Returns:
        Shortened topic name for display in hints
    """
    words = option.topic_name.split()
    if len(words) <= max_words:
        return option.topic_name
    return " ".join(words[:max_words]) + "..."


def format_disambiguation_hint(
    runner_ups: List[DisambiguationOption],
    mode: str = "text"
) -> str:
    """
    Format hint for appending after response.

    Args:
        runner_ups: The runner-up options not selected
        mode: Either "text" or "voice"

    Returns:
        Hint string to display after LLM response:
        - Voice: "...or did you mean something else?"
        - Text: "(Also found: topic2, topic3)"
    """
    if not runner_ups:
        return ""

    if mode == "voice":
        return "\n...or did you mean something else?"

    # Text mode - list the runner-up topics
    topic_names = [shorten_topic_name(opt) for opt in runner_ups[:2]]
    return f"\n(Also found: {', '.join(topic_names)})"


@dataclass
class DisambiguationResult:
    """Result of disambiguation input handling."""

    action: Literal["reactivate", "continue", "reprompt"]
    topic_start_node_id: Optional[str] = None
    topic_name: Optional[str] = None


def format_disambiguation_options(options: List[DisambiguationOption]) -> str:
    """
    Format disambiguation options for display.

    Shows evidence (snippets, hit count, turns ago), not just labels.

    Args:
        options: List of disambiguation options from probe_reactivation()

    Returns:
        Formatted string for display to user
    """
    lines = ["\nI found multiple topics that might match:\n"]

    # Limit to 3 options max
    for i, opt in enumerate(options[:3], 1):
        # Topic name and recency
        lines.append(f"[{i}] {opt.topic_name} ({opt.turns_ago} turns ago)")

        # Show snippets as evidence (max 2 per topic)
        for snippet in opt.snippets[:2]:
            # Already truncated in _get_topic_snippets, but ensure clean display
            lines.append(f'    \u2022 "{snippet}"')

        # Fallback to preview if no snippets
        if not opt.snippets and opt.preview:
            preview = opt.preview[:60] + "..." if len(opt.preview) > 60 else opt.preview
            lines.append(f'    \u2022 "{preview}"')

        # Hit count
        lines.append(f"    {opt.support_count} matching exchanges\n")

    lines.append("[0] Neither / Continue current topic\n")
    lines.append("Which topic? ")

    return "\n".join(lines)


def handle_disambiguation_input(
    user_input: str,
    options: List[DisambiguationOption],
    attempt: int = 1,
) -> DisambiguationResult:
    """
    Handle user's disambiguation choice.

    Args:
        user_input: User's input string
        options: Available disambiguation options
        attempt: Which attempt this is (1 or 2)

    Returns:
        DisambiguationResult with action and optional topic info:
        - ("reactivate", topic_id, topic_name) - user selected a topic
        - ("continue", None, None) - user chose 0 or skipped
        - ("reprompt", None, None) - invalid input, ask again (only on attempt 1)
    """
    user_input = user_input.strip()

    # Option 0: Continue current topic
    if user_input == "0":
        return DisambiguationResult(action="continue")

    # Try to parse as number
    try:
        choice = int(user_input)
        if 1 <= choice <= len(options):
            selected = options[choice - 1]
            return DisambiguationResult(
                action="reactivate",
                topic_start_node_id=selected.topic_start_node_id,
                topic_name=selected.topic_name,
            )
    except ValueError:
        pass

    # Invalid input
    if attempt == 1:
        return DisambiguationResult(action="reprompt")
    else:
        # Second invalid input -> skip to continue
        return DisambiguationResult(action="continue")
