"""
Sentence buffer for streaming TTS in Episodic voice mode.

Accumulates streaming text and emits complete sentences for TTS.
"""

import re
from typing import Callable, List, Optional


class SentenceBuffer:
    """
    Buffers streaming text and emits complete sentences.

    Used to feed TTS while LLM is still generating, enabling
    concurrent text streaming and speech output.
    """

    # Sentence-ending patterns
    SENTENCE_ENDS = re.compile(
        r'(?<=[.!?])'  # After sentence-ending punctuation
        r'(?:\s+|$)'   # Followed by whitespace or end
    )

    # Patterns that look like sentence ends but aren't
    FALSE_ENDS = re.compile(
        r'(?:'
        r'Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|'  # Titles
        r'Inc\.|Corp\.|Ltd\.|Co\.|'                 # Business
        r'vs\.|etc\.|e\.g\.|i\.e\.|'                # Abbreviations
        r'\d+\.\d+'                                  # Numbers
        r')$'
    )

    def __init__(
        self,
        on_sentence: Optional[Callable[[str], None]] = None,
        min_sentence_length: int = 10,
    ):
        """
        Initialize sentence buffer.

        Args:
            on_sentence: Callback for each complete sentence
            min_sentence_length: Minimum characters before emitting
        """
        self._buffer = ""
        self._on_sentence = on_sentence
        self._min_length = min_sentence_length
        self._emitted: List[str] = []

    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer, return complete sentence if available.

        Args:
            text: Text chunk to add (e.g., from streaming LLM)

        Returns:
            Complete sentence if one was detected, None otherwise
        """
        self._buffer += text
        return self._try_emit()

    def _try_emit(self) -> Optional[str]:
        """Try to emit a complete sentence from the buffer."""
        if len(self._buffer) < self._min_length:
            return None

        # Find potential sentence breaks
        matches = list(self.SENTENCE_ENDS.finditer(self._buffer))

        if not matches:
            return None

        # Check each potential break
        for match in matches:
            pos = match.start()
            candidate = self._buffer[:pos].strip()

            # Skip if too short
            if len(candidate) < self._min_length:
                continue

            # Skip false positives (abbreviations, etc.)
            if self.FALSE_ENDS.search(candidate):
                continue

            # We have a complete sentence
            sentence = candidate
            self._buffer = self._buffer[match.end():].lstrip()

            self._emitted.append(sentence)

            if self._on_sentence:
                self._on_sentence(sentence)

            return sentence

        return None

    def flush(self) -> Optional[str]:
        """
        Flush any remaining text as a final sentence.

        Call this when streaming is complete.

        Returns:
            Remaining text if any, None otherwise
        """
        remaining = self._buffer.strip()
        self._buffer = ""

        if remaining:
            self._emitted.append(remaining)
            if self._on_sentence:
                self._on_sentence(remaining)
            return remaining

        return None

    def clear(self):
        """Clear the buffer without emitting."""
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Get current buffered text (not yet emitted)."""
        return self._buffer

    @property
    def emitted(self) -> List[str]:
        """Get list of all emitted sentences."""
        return self._emitted.copy()

    def reset(self):
        """Reset buffer and emitted history."""
        self._buffer = ""
        self._emitted = []


class ParagraphBuffer:
    """
    Buffers text and emits complete paragraphs.

    Alternative to SentenceBuffer for longer TTS chunks.
    """

    def __init__(
        self,
        on_paragraph: Optional[Callable[[str], None]] = None,
        min_paragraph_length: int = 50,
    ):
        """
        Initialize paragraph buffer.

        Args:
            on_paragraph: Callback for each complete paragraph
            min_paragraph_length: Minimum characters before emitting
        """
        self._buffer = ""
        self._on_paragraph = on_paragraph
        self._min_length = min_paragraph_length
        self._emitted: List[str] = []

    def add(self, text: str) -> Optional[str]:
        """Add text, return paragraph if complete."""
        self._buffer += text

        # Look for double newlines (paragraph breaks)
        if "\n\n" in self._buffer:
            parts = self._buffer.split("\n\n", 1)
            paragraph = parts[0].strip()
            self._buffer = parts[1] if len(parts) > 1 else ""

            if len(paragraph) >= self._min_length:
                self._emitted.append(paragraph)
                if self._on_paragraph:
                    self._on_paragraph(paragraph)
                return paragraph

        return None

    def flush(self) -> Optional[str]:
        """Flush remaining text."""
        remaining = self._buffer.strip()
        self._buffer = ""

        if remaining and len(remaining) >= self._min_length:
            self._emitted.append(remaining)
            if self._on_paragraph:
                self._on_paragraph(remaining)
            return remaining

        return None

    def clear(self):
        """Clear buffer."""
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Get buffered text."""
        return self._buffer

    def reset(self):
        """Reset buffer and history."""
        self._buffer = ""
        self._emitted = []
