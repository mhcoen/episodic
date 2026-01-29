"""
Resume cue detection for topic reactivation.

Detects patterns that indicate the user wants to resume a previous topic,
e.g., "Back to that Python thing", "where were we with the API".
"""

import re
from typing import List

# Resume cue patterns
RESUME_CUE_PATTERNS: List[str] = [
    r"\bback to\b",
    r"\bcontinuing\b",
    r"\bthat\s+.{1,30}\s+thing\b",  # "that Python thing"
    r"\bas we were\b",
    r"\banyway\b",
    r"\bwhere were we\b",
    r"\blet'?s get back\b",
    r"\bresume\b",
    r"\bpicking up\b",
    r"\breturning to\b",
]


def has_resume_cues(text: str) -> bool:
    """
    Check if text contains resume cues suggesting topic continuation.

    Resume cues indicate the user wants to continue a previous topic,
    not just recall information about it.

    Args:
        text: User input text to check

    Returns:
        True if resume cues detected, False otherwise
    """
    text_lower = text.lower()

    # Check explicit patterns
    for pattern in RESUME_CUE_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    # Anaphoric reference + forward question (e.g., "that thing - should I...?")
    if "that" in text_lower and "?" in text:
        return True

    return False
