"""
Base interface for dialogue segmenters.

All segmenters produce boundaries in canonical format:
- boundary at index t means "boundary between message t-1 and message t"
- valid range: [1, T-1] where T is the number of messages
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class SegmenterResult:
    """Result from a segmenter."""

    # Core output: boundary indices in canonical format
    boundaries: List[int]

    # Optional: confidence scores per boundary
    scores: Optional[Dict[int, float]] = None

    # Optional: metadata about the segmentation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_set(self) -> Set[int]:
        """Return boundaries as a set."""
        return set(self.boundaries)


class Segmenter(ABC):
    """
    Base class for dialogue segmentation methods.

    All segmenters implement predict_boundaries() which takes a dialogue
    and returns boundary positions in canonical format.

    Canonical boundary format:
    - Index t means "topic changes at message t"
    - Equivalently: boundary between message t-1 and message t
    - Valid indices: [1, T-1] for T messages
    - No boundary at 0 (nothing before first message)
    - No boundary at T (nothing after last message)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the segmenter."""
        pass

    @property
    def short_name(self) -> str:
        """Short name for tables/filenames."""
        return self.name.lower().replace(" ", "_")

    @property
    def description(self) -> str:
        """Description of the method."""
        return ""

    @abstractmethod
    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> SegmenterResult:
        """
        Predict topic boundaries in a dialogue.

        Args:
            messages: List of dicts with 'role' and 'content' keys
            **kwargs: Method-specific parameters

        Returns:
            SegmenterResult with boundary indices in canonical format
        """
        pass

    def predict_boundaries_batch(
        self,
        dialogues: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[SegmenterResult]:
        """
        Predict boundaries for multiple dialogues.

        Default implementation calls predict_boundaries() for each dialogue.
        Subclasses may override for more efficient batched processing.
        """
        return [
            self.predict_boundaries(messages, **kwargs)
            for messages in dialogues
        ]

    def _validate_boundaries(self, boundaries: List[int], num_messages: int) -> List[int]:
        """Validate and filter boundaries to canonical range."""
        valid = []
        for b in sorted(set(boundaries)):
            if 1 <= b < num_messages:
                valid.append(b)
        return valid


def messages_to_utterances(messages: List[Dict[str, str]]) -> List[str]:
    """
    Convert messages to list of utterance strings.

    Many segmenters expect a flat list of strings rather than dicts.
    """
    return [msg.get("content", "") for msg in messages]


def utterances_to_messages(
    utterances: List[str],
    alternating: bool = True
) -> List[Dict[str, str]]:
    """
    Convert utterances back to message format.

    Args:
        utterances: List of utterance strings
        alternating: If True, alternate user/assistant roles

    Returns:
        List of message dicts
    """
    messages = []
    for i, utt in enumerate(utterances):
        if alternating:
            role = "user" if i % 2 == 0 else "assistant"
        else:
            role = "user"
        messages.append({"role": role, "content": utt})
    return messages
