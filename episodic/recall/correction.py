"""
Correction detection for conversational disambiguation.

When multiple topics match a user's query, we proceed with the best guess
and allow natural correction ("no, the other one") rather than forcing
numbered selection.

This module detects correction intent and resolves which runner-up
the user is referring to.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from episodic.recall.reactivation import DisambiguationOption


@dataclass
class CorrectionState:
    """State for pending correction after ambiguous disambiguation."""

    query: str  # Original user query that triggered disambiguation
    chosen_option: DisambiguationOption  # The option we proceeded with
    runner_ups: List[DisambiguationOption]  # Other viable options
    turn_created: int  # Turn index when state was created (expires after 1-2 turns)


# Correction patterns: (regex, capture_group_for_hint)
# capture_group_for_hint is the group number (1-indexed) to extract as hint, or None
# More specific patterns (with hints) come first to take priority
CORRECTION_PATTERNS: List[Tuple[str, Optional[int]]] = [
    # With hint - capture the hint text (most specific, check first)
    (r"\bno[,.]?\s*(?:the\s+)?(\w+)\s+one\b", 1),  # "no, the coffee one"
    (r"\bi meant\s+(.+)$", 1),  # "I meant X"
    (r"\bno[,.]?\s*about\s+(.+)$", 1),  # "no, about X"
    # Ordinal reference
    (r"\bthe other one\b", None),
    (r"\bthe (second|different) one\b", None),
    # Direct negation (least specific, check last)
    (r"^no\b", None),
    (r"^nope\b", None),
    (r"\bnot that\b", None),
    (r"\bwrong (one|topic)\b", None),
]


def detect_correction(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if user input is a correction to previous disambiguation.

    Returns:
        Tuple of (is_correction, target_hint)
        - is_correction: True if user is correcting the disambiguation
        - target_hint: Optional keyword hint for matching to runner-up
    """
    text = user_input.lower().strip()

    for pattern, capture_group in CORRECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hint = None
            if capture_group is not None:
                try:
                    hint = match.group(capture_group).strip()
                except (IndexError, AttributeError):
                    pass
            return True, hint

    return False, None


def resolve_correction(
    state: CorrectionState, hint: Optional[str]
) -> Optional[DisambiguationOption]:
    """
    Match hint to runner-up label terms, or return first runner-up if no hint.

    Args:
        state: The correction state with runner-ups
        hint: Optional keyword hint from user (e.g., "coffee" from "no, the coffee one")

    Returns:
        The matched runner-up option, or None if no runner-ups available
    """
    if not state.runner_ups:
        return None

    # If no hint, return first runner-up
    if not hint:
        return state.runner_ups[0]

    hint_lower = hint.lower()

    # Try to match hint to topic names
    for option in state.runner_ups:
        topic_lower = option.topic_name.lower()
        # Check if hint is in topic name
        if hint_lower in topic_lower:
            return option
        # Check if any word in hint matches any word in topic name
        hint_words = set(hint_lower.split())
        topic_words = set(topic_lower.split())
        if hint_words & topic_words:
            return option

    # Try to match hint to snippets
    for option in state.runner_ups:
        for snippet in option.snippets:
            if hint_lower in snippet.lower():
                return option

    # No match found - return first runner-up as fallback
    return state.runner_ups[0]
