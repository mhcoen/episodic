"""
InstructGPT (text-davinci-003) dialogue segmenter.

Configuration-matched reproduction of the SuperDialseg InstructGPT baseline.
Uses text-davinci-003 with temperature=0, max_tokens=512.

This is a single-shot instruction model, not a chat model.
The SuperDialseg paper does not release the exact prompt string, so this
reproduces the task intent and output format, not a verbatim prompt.

Usage:
    segmenter = InstructGPTSegmenter()
    result = segmenter.predict_boundaries(messages)
"""

import os
import re
import logging
from typing import List, Dict, Optional

from .base import Segmenter, SegmenterResult

logger = logging.getLogger(__name__)

# Default model parameters matching SuperDialseg
DEFAULT_MODEL = "text-davinci-003"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 512


class InstructGPTSegmenter(Segmenter):
    """
    InstructGPT-based dialogue topic segmenter.

    Uses text-davinci-003 (instruction-following model, NOT chat model)
    with temperature=0 for deterministic outputs.

    This is a configuration-matched InstructGPT reproduction, not a
    verbatim reproduction (exact prompt not released in SuperDialseg).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
    ):
        """
        Initialize InstructGPT segmenter.

        Args:
            model: OpenAI model to use (default: text-davinci-003)
            temperature: Sampling temperature (default: 0)
            max_tokens: Maximum tokens in response (default: 512)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily initialize OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key to constructor."
                )
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "InstructGPT"

    @property
    def short_name(self) -> str:
        return "instructgpt"

    @property
    def description(self) -> str:
        return f"InstructGPT ({self.model}) configuration-matched reproduction"

    def _format_dialogue(self, messages: List[Dict[str, str]]) -> str:
        """Format dialogue as numbered utterances."""
        lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            # Use 1-indexed for human readability
            lines.append(f"[{i+1}] {role.upper()}: {content}")
        return "\n".join(lines)

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Build instruction prompt for topic boundary detection.

        The prompt:
        - Presents the full dialogue as plain text
        - Instructs the model to identify topic boundaries
        - Requests boundaries in structured, machine-readable format
        """
        dialogue_text = self._format_dialogue(messages)
        n_messages = len(messages)

        prompt = f"""You are a dialogue topic segmentation system. Your task is to identify where topic changes occur in the following dialogue.

A topic boundary occurs between two consecutive utterances when the conversation shifts from one topic to another.

Dialogue:
{dialogue_text}

Instructions:
1. Analyze the dialogue above and identify all topic boundaries
2. A boundary at position N means the topic changes AFTER utterance N and BEFORE utterance N+1
3. Valid boundary positions are 1 through {n_messages - 1}
4. Return ONLY the boundary positions as a comma-separated list of numbers
5. If there are no topic boundaries, return "none"

Boundary positions:"""

        return prompt

    def _parse_response(self, response_text: str, num_messages: int) -> List[int]:
        """
        Parse model response to extract boundary indices.

        Returns boundaries in canonical format (between-message indices).
        """
        text = response_text.strip().lower()

        # Handle "none" or empty response
        if text in ("none", "n/a", "no boundaries", ""):
            return []

        # Extract numbers from response
        numbers = re.findall(r'\d+', text)

        boundaries = []
        for num_str in numbers:
            try:
                # The prompt uses 1-indexed positions
                # "boundary at position N" means after utterance N
                # In canonical format, this is boundary index N (between N-1 and N in 0-indexed)
                pos = int(num_str)
                # Validate range: must be in [1, num_messages-1]
                if 1 <= pos < num_messages:
                    boundaries.append(pos)
            except ValueError:
                continue

        return sorted(set(boundaries))

    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> SegmenterResult:
        """
        Predict topic boundaries using InstructGPT.

        Args:
            messages: List of dicts with 'role' and 'content' keys
            **kwargs: Additional arguments (ignored)

        Returns:
            SegmenterResult with boundary indices in canonical format
        """
        num_messages = len(messages)
        if num_messages <= 1:
            return SegmenterResult(boundaries=[], metadata={"method": "instructgpt"})

        # Build prompt
        prompt = self._build_prompt(messages)

        try:
            # Call OpenAI Completions API
            client = self._get_client()
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            response_text = response.choices[0].text

            # Parse boundaries
            boundaries = self._parse_response(response_text, num_messages)

            return SegmenterResult(
                boundaries=boundaries,
                metadata={
                    "method": "instructgpt",
                    "model": self.model,
                    "temperature": self.temperature,
                    "raw_response": response_text.strip(),
                }
            )

        except Exception as e:
            logger.error(f"InstructGPT API error: {e}")
            raise


class InstructGPTSegmenterMock(Segmenter):
    """
    Mock InstructGPT segmenter for testing without API access.

    Returns random boundaries to simulate model behavior.
    """

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "InstructGPT-Mock"

    @property
    def description(self) -> str:
        return "Mock InstructGPT for testing"

    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> SegmenterResult:
        num_messages = len(messages)
        if num_messages <= 1:
            return SegmenterResult(boundaries=[])

        # Random boundaries with ~20% probability
        boundaries = [
            i for i in range(1, num_messages)
            if self.rng.random() < 0.2
        ]

        return SegmenterResult(
            boundaries=boundaries,
            metadata={"method": "instructgpt_mock"}
        )
