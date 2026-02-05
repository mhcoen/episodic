"""
Label Extractor for Voice Grammar.

Extracts free-text LABEL from token stream with rule-specific stop conditions.
Stop conditions are TOKEN SEQUENCES, not substrings.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict


@dataclass
class LabelCapture:
    """Result of label extraction."""
    text: str
    stop_reason: Optional[str] = None


class LabelExtractor:
    """
    Extract LABEL from token stream with rule-specific stop conditions.
    Stop conditions are TOKEN SEQUENCES, not substrings.
    """

    STOP_CONDITIONS: Dict[str, Dict] = {
        "timer_set": {
            "tokens": {"for"},
            "sequences": [["in"], ["at"]],
            "preserve": set(),
        },
        "alarm_set": {
            "tokens": {"at", "for"},
            "sequences": [["tomorrow"], ["this", "morning"], ["this", "evening"]],
            "preserve": set(),
        },
        "remind_set": {
            "tokens": {"in", "at"},
            "sequences": [["tomorrow"], ["this", "morning"]],
            "preserve": set(),
        },
        "note_add": {
            # Notes capture almost everything
            "tokens": set(),
            "sequences": [["and", "also"], ["and", "then"]],
            "preserve": {"for", "in", "at", "on"},
        },
        "media_play": {
            "tokens": set(),
            "sequences": [["louder"], ["quieter"], ["at", "volume"]],
            "preserve": {"for", "in", "at", "on"},
        },
    }

    DURATION_WORDS: Set[str] = {"couple", "few", "half", "quarter"}
    TIME_UNITS: Set[str] = {
        "minute", "minutes", "min", "mins",
        "hour", "hours", "hr", "hrs",
        "second", "seconds", "sec", "secs"
    }

    def extract(
        self,
        tokens: List[str],
        start_idx: int,
        rule: str
    ) -> Tuple[LabelCapture, int]:
        """
        Extract LABEL starting at start_idx for given rule.
        Returns (LabelCapture, next_idx).
        """
        if rule not in self.STOP_CONDITIONS:
            label_tokens = tokens[start_idx:]
            return (LabelCapture(" ".join(label_tokens)), len(tokens))

        config = self.STOP_CONDITIONS[rule]
        label_tokens: List[str] = []
        i = start_idx

        while i < len(tokens):
            token_lower = tokens[i].lower()

            # Check single-token stop (unless preserved)
            if token_lower in config["tokens"] and token_lower not in config["preserve"]:
                if self._is_stop_context(tokens, i, rule):
                    return (LabelCapture(" ".join(label_tokens), f"token:{token_lower}"), i)

            # Check sequence stops
            for seq in config["sequences"]:
                if self._matches_sequence(tokens, i, seq):
                    return (LabelCapture(" ".join(label_tokens), f"sequence:{seq}"), i)

            label_tokens.append(tokens[i])
            i += 1

        return (LabelCapture(" ".join(label_tokens)), i)

    def _is_stop_context(self, tokens: List[str], idx: int, rule: str) -> bool:
        """
        Verify stop token is in valid stop context.
        E.g., "for" followed by duration is stop; "for" in "looking for" is not.
        """
        if idx + 1 >= len(tokens):
            return False

        next_token = tokens[idx + 1].lower()

        # Next is a digit
        if next_token.isdigit():
            return True

        # Next is "a/an" followed by time unit
        if next_token in {"a", "an"} and idx + 2 < len(tokens):
            unit = tokens[idx + 2].lower()
            if unit in self.TIME_UNITS:
                return True

        # Next is a duration word
        if next_token in self.DURATION_WORDS:
            return True

        return False

    def _matches_sequence(self, tokens: List[str], start: int, sequence: List[str]) -> bool:
        """Check if a sequence matches at position start."""
        if start + len(sequence) > len(tokens):
            return False
        for i, expected in enumerate(sequence):
            if tokens[start + i].lower() != expected:
                return False
        return True
