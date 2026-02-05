"""
Memory Query Detector for Voice Grammar.

Detects past-tense and memory queries that should bypass utility commands
and go directly to the LLM for conversational response.
"""

from typing import Tuple, List, Optional, Set


class MemoryQueryDetector:
    """
    Detect past-tense and memory/history queries.
    Uses token boundaries, not substring matching.
    """

    # Tokens that can be skipped when looking for memory patterns
    SKIPPABLE_TOKENS: Set[str] = {
        "um", "uh", "er", "ah", "like", "you know", "basically",
        "hey", "hi", "ok", "okay", "please", "can", "you", "could",
        "would", "will", "just"
    }

    PAST_TOKENS: Set[str] = {
        "did", "was", "were", "had", "started", "ended", "finished",
        "earlier", "yesterday", "previously", "already"
    }

    PAST_SEQUENCES: List[List[str]] = [
        ["last", "time"], ["last", "week"], ["last", "month"],
        ["last", "year"], ["used", "to"], ["have", "been"], ["has", "been"],
    ]

    MEMORY_PATTERNS: List[List[str]] = [
        ["when", "did"], ["what", "time", "did"], ["where", "did"],
        ["who", "did"], ["how", "did"], ["what", "did", "we"],
        ["what", "did", "i"], ["do", "you", "remember"],
        ["did", "we", "discuss"], ["did", "i", "mention"],
    ]

    def should_bypass_utilities(self, tokens: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if the utterance should bypass utilities and go to LLM.

        Returns:
            (should_bypass, reason) - reason is None if not bypassing
        """
        tokens_lower = [t.lower() for t in tokens]

        # Skip leading skippable tokens
        content_start = 0
        while (content_start < len(tokens_lower) and
               tokens_lower[content_start] in self.SKIPPABLE_TOKENS):
            content_start += 1

        content_tokens = tokens_lower[content_start:]

        # Check single-token past markers
        for token in content_tokens:
            if token in self.PAST_TOKENS:
                return (True, f"past_token:{token}")

        # Check past sequences
        for seq in self.PAST_SEQUENCES:
            if self._contains_sequence(content_tokens, seq):
                return (True, f"past_sequence:{' '.join(seq)}")

        # Check memory patterns (at start of content)
        for pattern in self.MEMORY_PATTERNS:
            if content_tokens[:len(pattern)] == pattern:
                return (True, f"memory_pattern:{' '.join(pattern)}")

        return (False, None)

    def _contains_sequence(self, tokens: List[str], sequence: List[str]) -> bool:
        """Check if a sequence of tokens appears in the token list."""
        for i in range(len(tokens) - len(sequence) + 1):
            if tokens[i:i + len(sequence)] == sequence:
                return True
        return False
