"""
Token Counting: Protocol, registry, and estimation functions.

Extracted from token_guard.py to break the token_guard <-> truncation circular import.
truncation.py imports HeuristicTokenCounter from here instead of token_guard.

- TokenCounter protocol with is_exact() capability
- Registry for provider/model-specific counters
- Estimation functions for text, messages, and assemblies
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# TokenCounter Protocol and Registry
# =============================================================================

@runtime_checkable
class TokenCounter(Protocol):
    """
    Protocol for token counting backends.

    Implementations must provide:
    - count_text: Count tokens in a text string
    - count_messages: Count tokens in a list of messages
    - is_exact: Whether this counter uses exact tokenization
    - backend_name: Identifier for logging/debugging
    """

    def count_text(self, text: str) -> int:
        """Count tokens in a text string."""
        ...

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in a list of messages (exact structure used by validate_assembly)."""
        ...

    def is_exact(self) -> bool:
        """Return True if this counter uses exact tokenization, False if heuristic."""
        ...

    def backend_name(self) -> str:
        """Return identifier for this counting backend."""
        ...


class HeuristicTokenCounter:
    """
    Heuristic token counter using chars/4 approximation.

    This is the default counter when no exact tokenizer is available.
    Safety factor should be applied externally when is_exact()==False.
    """

    # Overhead for message structure (role markers, delimiters)
    MESSAGE_OVERHEAD = 4
    # Rough token estimate for image content blocks
    IMAGE_BLOCK_TOKENS = 85

    def count_text(self, text: str) -> int:
        """Count tokens using chars/4 heuristic."""
        if not text:
            return 0
        return len(text) // 4

    def count_message(self, message: Dict[str, Any]) -> int:
        """Count tokens for a single message."""
        content = message.get("content", "")

        # Handle multimodal content (list of content blocks)
        if isinstance(content, list):
            text_tokens = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_tokens += self.count_text(block.get("text", ""))
                    elif block.get("type") in ("image_url", "image"):
                        text_tokens += self.IMAGE_BLOCK_TOKENS
                elif isinstance(block, str):
                    text_tokens += self.count_text(block)
            content_tokens = text_tokens
        else:
            content_tokens = self.count_text(content)

        return content_tokens + self.MESSAGE_OVERHEAD

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        return sum(self.count_message(msg) for msg in messages)

    def is_exact(self) -> bool:
        """Heuristic counter is not exact."""
        return False

    def backend_name(self) -> str:
        """Return backend identifier."""
        return "heuristic_chars_div_4"


class _TokenCounterRegistry:
    """
    Registry for token counting backends.

    Keyed by (provider_id, model_id). Returns exact counter if available,
    otherwise returns the default heuristic counter.
    """

    def __init__(self):
        self._counters: Dict[Tuple[str, str], TokenCounter] = {}
        self._default_counter = HeuristicTokenCounter()

    def register(self, provider_id: str, model_id: str, counter: TokenCounter) -> None:
        """Register a counter for a specific provider/model combination."""
        key = (provider_id.lower(), model_id.lower())
        self._counters[key] = counter

    def get(self, provider_id: Optional[str], model_id: Optional[str]) -> TokenCounter:
        """
        Get a counter for the given provider/model.

        Returns exact counter if registered, otherwise default heuristic.
        """
        if provider_id and model_id:
            key = (provider_id.lower(), model_id.lower())
            if key in self._counters:
                return self._counters[key]

        # No exact counter available - return heuristic
        return self._default_counter

    def get_default(self) -> TokenCounter:
        """Get the default heuristic counter."""
        return self._default_counter


# Global registry instance
token_counter_registry = _TokenCounterRegistry()


def get_token_counter(provider_id: Optional[str] = None, model_id: Optional[str] = None) -> TokenCounter:
    """
    Get a token counter for the given provider/model.

    Convenience function that delegates to the global registry.
    """
    return token_counter_registry.get(provider_id, model_id)


# =============================================================================
# Token Estimate Data Class
# =============================================================================

@dataclass
class TokenEstimate:
    """Token estimate for a message assembly."""
    # Estimated tokens per component
    system_prompt: int = 0
    summary: int = 0
    anchors: int = 0
    recency: int = 0
    rag_context: int = 0
    web_context: int = 0
    user_message: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        """Total estimated tokens."""
        return (
            self.system_prompt
            + self.summary
            + self.anchors
            + self.recency
            + self.rag_context
            + self.web_context
            + self.user_message
            + self.other
        )


# =============================================================================
# Token Estimation Functions
# =============================================================================

def estimate_tokens_text(text: str, safety_factor: float = 1.0) -> int:
    """
    Estimate token count from text.

    Uses the default heuristic counter.

    Args:
        text: The text to estimate tokens for
        safety_factor: Multiplicative safety factor (applied externally)

    Returns:
        Estimated token count
    """
    counter = token_counter_registry.get_default()
    base_count = counter.count_text(text)
    return int(base_count * safety_factor)


def estimate_tokens_message(message: Dict[str, Any], safety_factor: float = 1.0) -> int:
    """
    Estimate tokens for a single message.

    Args:
        message: Message dict with 'role' and 'content'
        safety_factor: Multiplicative safety factor

    Returns:
        Estimated token count
    """
    counter = token_counter_registry.get_default()
    if isinstance(counter, HeuristicTokenCounter):
        base_count = counter.count_message(message)
    else:
        # For protocol-only counters, use messages list
        base_count = counter.count_messages([message])
    return int(base_count * safety_factor)


def estimate_tokens_messages(
    messages: List[Dict[str, Any]],
    safety_factor: float = 1.0
) -> Tuple[int, TokenEstimate]:
    """
    Estimate total tokens for a list of messages.

    Categorizes tokens by message type for detailed breakdown.

    Args:
        messages: List of message dicts
        safety_factor: Multiplicative safety factor

    Returns:
        Tuple of (total_tokens, breakdown)
    """
    estimate = TokenEstimate()
    total = 0

    for msg in messages:
        tokens = estimate_tokens_message(msg, safety_factor)
        total += tokens

        role = msg.get("role", "")
        content = msg.get("content", "")

        # Categorize by content type
        if role == "system":
            content_str = content if isinstance(content, str) else str(content)
            if "# Topic:" in content_str or "## Summary" in content_str:
                if "## Summary" in content_str:
                    estimate.summary += tokens
                else:
                    estimate.system_prompt += tokens
            elif "## Relevant Past Context" in content_str:
                estimate.anchors += tokens
            elif "Relevant context from knowledge base" in content_str or "[Doc:" in content_str:
                estimate.rag_context += tokens
            elif "[Memory]" in content_str:
                estimate.rag_context += tokens
            elif "search results" in content_str.lower() or "web" in content_str.lower():
                estimate.web_context += tokens
            else:
                estimate.system_prompt += tokens
        elif role == "user":
            estimate.user_message += tokens
        elif role == "assistant":
            estimate.recency += tokens
        else:
            estimate.other += tokens

    return total, estimate
