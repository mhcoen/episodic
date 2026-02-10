"""
Token Guard: Centralized token validation and assembly cap enforcement.

Gap B Implementation:
- Full-assembly token assertion after context is built
- Fail-closed policy when cap is exceeded
- Bug event logging for overflow conditions

Gap A Implementation:
- TokenCounter protocol with is_exact() capability
- Registry for provider/model-specific counters
- Safety factor applied only when is_exact()==False
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Gap A: TokenCounter Protocol and Registry
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
# Gap B: Drop Actions and Budget Configuration
# =============================================================================

class DropAction(Enum):
    """Actions taken to reduce token count."""
    TRUNCATE_SUMMARY = "truncate_summary"
    DROP_RECENCY = "drop_recency"
    DROP_ANCHORS = "drop_anchors"
    RELEVANCE_TRUNCATION = "relevance_truncation"
    ABORT = "abort"


@dataclass
class TokenBudget:
    """Token budget configuration for assembly validation."""
    # Total cap for the full assembly (all messages combined)
    full_cap: int = 8000
    # Minimum summary tokens after truncation (s_min)
    summary_min: int = 100
    # Reserved for system overhead (prompt templates, delimiters, etc.)
    overhead_reserve: int = 500


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


@dataclass
class ValidationResult:
    """Result of assembly token validation."""
    valid: bool
    original_tokens: int
    final_tokens: int
    actions_taken: List[DropAction] = field(default_factory=list)
    messages: Optional[List[Dict[str, Any]]] = None
    fallback_response: Optional[str] = None
    bug_event_logged: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    # Truncation result for logging (when relevance truncation enabled)
    truncation_result: Optional[Any] = None  # TruncationResult from truncation module


# =============================================================================
# Token Estimation Functions (for backward compatibility)
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


# =============================================================================
# Bug Event Logging
# =============================================================================

def log_bug_event(
    event_type: str,
    details: Dict[str, Any],
    severity: str = "warning"
) -> None:
    """
    Log a bug event for token overflow or validation failure.

    This creates a structured log entry that can be analyzed for debugging
    and identifying systematic issues with token estimation.

    Args:
        event_type: Type of bug event (e.g., "token_overflow", "estimation_mismatch")
        details: Additional details about the event
        severity: Log severity level
    """
    log_record = {
        "event": "token_guard_bug",
        "event_type": event_type,
        **details
    }

    if severity == "error":
        logger.error(f"TOKEN_GUARD_BUG: {log_record}")
    elif severity == "warning":
        logger.warning(f"TOKEN_GUARD_BUG: {log_record}")
    else:
        logger.info(f"TOKEN_GUARD_BUG: {log_record}")


# =============================================================================
# Drop Policy Helper Functions
# =============================================================================

def _find_summary_message_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    """Find the index of the summary-containing message."""
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and "## Summary" in content:
                return i
    return None


def _truncate_summary_in_messages(
    messages: List[Dict[str, Any]],
    target_reduction: int,
    summary_min: int,
    counter: Optional[TokenCounter] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Truncate summary section to reduce tokens.

    Returns:
        Tuple of (modified_messages, tokens_reduced)
    """
    if counter is None:
        counter = token_counter_registry.get_default()

    idx = _find_summary_message_index(messages)
    if idx is None:
        return messages, 0

    msg = messages[idx]
    content = msg.get("content", "")

    if "## Summary" not in content:
        return messages, 0

    # Find and truncate the summary section
    parts = content.split("## Summary")
    if len(parts) < 2:
        return messages, 0

    before_summary = parts[0]
    after_summary = parts[1]

    # Find where summary ends (next ## or end)
    summary_end = len(after_summary)
    for marker in ["##", "\n\n## "]:
        pos = after_summary.find(marker, 1)
        if pos != -1 and pos < summary_end:
            summary_end = pos

    summary_text = after_summary[:summary_end]
    rest_of_content = after_summary[summary_end:]

    original_summary_tokens = counter.count_text(summary_text)
    target_summary_tokens = max(summary_min, original_summary_tokens - target_reduction)

    # Truncate summary to target
    if target_summary_tokens < original_summary_tokens:
        max_chars = target_summary_tokens * 4
        if len(summary_text) > max_chars:
            truncated_summary = summary_text[:max_chars - 3].rstrip() + "..."
        else:
            truncated_summary = summary_text
    else:
        truncated_summary = summary_text

    new_content = before_summary + "## Summary" + truncated_summary + rest_of_content

    new_messages = messages.copy()
    new_messages[idx] = {**msg, "content": new_content}

    tokens_reduced = original_summary_tokens - counter.count_text(truncated_summary)
    return new_messages, tokens_reduced


def _find_recency_messages(messages: List[Dict[str, Any]]) -> List[int]:
    """
    Find indices of recency (conversation history) messages.

    Recency messages are user/assistant pairs that aren't the current user message.
    """
    indices = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        # Skip system messages
        if role == "system":
            continue
        # The last user message is the current query, not recency
        if i == len(messages) - 1 and role == "user":
            continue
        indices.append(i)
    return indices


def _drop_recency_messages(
    messages: List[Dict[str, Any]],
    target_reduction: int,
    counter: Optional[TokenCounter] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Drop recency messages (oldest first) to reduce tokens.

    Returns:
        Tuple of (modified_messages, tokens_reduced)
    """
    if counter is None:
        counter = token_counter_registry.get_default()

    recency_indices = _find_recency_messages(messages)

    if not recency_indices:
        return messages, 0

    # Calculate tokens for each recency message
    recency_with_tokens = []
    for idx in recency_indices:
        if isinstance(counter, HeuristicTokenCounter):
            tokens = counter.count_message(messages[idx])
        else:
            tokens = counter.count_messages([messages[idx]])
        recency_with_tokens.append((idx, tokens))

    # Drop from oldest (lowest index) first
    indices_to_drop = []
    tokens_to_reduce = 0

    for idx, tokens in recency_with_tokens:
        if tokens_to_reduce >= target_reduction:
            break
        indices_to_drop.append(idx)
        tokens_to_reduce += tokens

    # Create new message list without dropped indices
    new_messages = [
        msg for i, msg in enumerate(messages)
        if i not in indices_to_drop
    ]

    return new_messages, tokens_to_reduce


def _find_anchor_section_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    """Find the index of the message containing anchors."""
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and "## Relevant Past Context" in content:
                return i
    return None


def _drop_anchors_from_messages(
    messages: List[Dict[str, Any]],
    counter: Optional[TokenCounter] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Drop anchor section entirely.

    Returns:
        Tuple of (modified_messages, tokens_reduced)
    """
    if counter is None:
        counter = token_counter_registry.get_default()

    idx = _find_anchor_section_index(messages)
    if idx is None:
        return messages, 0

    msg = messages[idx]
    content = msg.get("content", "")

    if "## Relevant Past Context" not in content:
        return messages, 0

    # Remove the anchor section
    parts = content.split("## Relevant Past Context")
    before_anchors = parts[0]

    if len(parts) > 1:
        after_anchors = parts[1]
        # Find where anchors end (next ## or end)
        anchor_end = len(after_anchors)
        for marker in ["\n\n## "]:
            pos = after_anchors.find(marker)
            if pos != -1 and pos < anchor_end:
                anchor_end = pos
        rest_of_content = after_anchors[anchor_end:]
    else:
        rest_of_content = ""

    # Calculate tokens saved
    original_tokens = counter.count_messages(messages)

    new_content = before_anchors.rstrip() + rest_of_content

    new_messages = messages.copy()
    if new_content.strip():
        new_messages[idx] = {**msg, "content": new_content}
    else:
        # Remove the message entirely if empty
        new_messages.pop(idx)

    new_tokens = counter.count_messages(new_messages)
    tokens_reduced = original_tokens - new_tokens

    return new_messages, tokens_reduced


# =============================================================================
# Category D: Event Emission Helper
# =============================================================================

def _emit_token_guard_event(
    result: ValidationResult,
    assembly_id: str,
    turn_id: str,
    counter: TokenCounter,
    raw_tokens: int,
    effective_cap: int,
    effective_safety_factor: float,
) -> None:
    """
    Emit a structured token guard event.

    Helper function to ensure consistent event emission regardless of code path.
    """
    try:
        from episodic.token_guard_events import get_event_logger, EventType

        event_logger = get_event_logger()

        # Determine event type
        if not result.valid:
            event_type = EventType.TOKEN_OVERFLOW_ABORT
        elif result.actions_taken:
            event_type = EventType.TOKEN_OVERFLOW_RECOVERED
        else:
            event_type = EventType.TOKEN_OK

        # Build extra fields
        extra = {
            "actions_taken": [a.value for a in result.actions_taken],
            "original_tokens": result.original_tokens,
        }

        # Include truncation details if relevance truncation was used
        if result.truncation_result is not None:
            tr = result.truncation_result
            extra["truncation"] = {
                "tokens_before": tr.tokens_before,
                "tokens_after": tr.tokens_after,
                "tokens_freed": tr.tokens_before - tr.tokens_after,
                "decisions_count": len(tr.decisions),
                "decisions": [d.to_dict() for d in tr.decisions[:10]],  # Limit for event size
            }

        event_logger.emit(
            event_type=event_type,
            assembly_id=assembly_id,
            turn_id=turn_id,
            counter_backend=counter.backend_name(),
            counter_exact=counter.is_exact(),
            applied_safety_factor=effective_safety_factor if not counter.is_exact() else None,
            raw_tokens=raw_tokens,
            effective_tokens=result.final_tokens,
            cap=effective_cap,
            budget_breakdown=result.details["breakdown"],
            extra=extra
        )
    except Exception as e:
        # Don't fail validation due to event emission errors
        logger.warning(f"Failed to emit token guard event: {e}")


# =============================================================================
# Main Validation Functions
# =============================================================================

def validate_assembly(
    messages: List[Dict[str, Any]],
    budget: Optional[TokenBudget] = None,
    safety_factor: float = 1.0,
    apply_drops: bool = True,
    counter: Optional[TokenCounter] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    assembly_id: Optional[str] = None,
    emit_event: bool = True,
    enable_relevance_truncation: Optional[bool] = None,
    current_query: Optional[str] = None,
    anchor_indices: Optional[set] = None,
) -> ValidationResult:
    """
    Validate assembled messages against token cap.

    Implements fail-closed policy:
    0. If relevance truncation enabled: use importance-based drop (Phase 2)
    1. Check if total tokens exceed cap
    2. If over: truncate summary (down to s_min)
    3. If still over: drop recency messages (oldest first)
    4. If still over: drop anchors
    5. If still over: abort with safe fallback and log bug event

    Gap A: Safety factor is only applied when counter.is_exact()==False.
    Category D: Emits exactly one structured event per call.
    Phase 2: Relevance-aware truncation drops by importance score when enabled.

    Args:
        messages: List of assembled messages
        budget: Token budget configuration (uses defaults if None)
        safety_factor: Multiplicative safety factor for heuristic counters
        apply_drops: Whether to apply drop actions or just validate
        counter: Explicit token counter (overrides registry lookup)
        provider_id: Provider ID for registry lookup
        model_id: Model ID for registry lookup
        turn_id: Unique ID for conversation turn (auto-generated if None)
        assembly_id: Unique ID for this assembly call (auto-generated if None)
        emit_event: Whether to emit structured event (default True)
        enable_relevance_truncation: Use importance-based truncation (default: from config)
        current_query: Current user query for relevance scoring (required if truncation enabled)
        anchor_indices: Set of message indices that are anchors (for truncation scoring)

    Returns:
        ValidationResult with valid flag, actions taken, and optionally modified messages
    """
    import uuid
    from episodic.config import config

    # Generate IDs if not provided
    if turn_id is None:
        turn_id = str(uuid.uuid4())
    if assembly_id is None:
        assembly_id = str(uuid.uuid4())

    # Resolve token counter
    if counter is None:
        counter = get_token_counter(provider_id, model_id)

    # Determine effective safety factor
    # Only apply safety factor if counter is NOT exact
    if counter.is_exact():
        effective_safety_factor = 1.0
    else:
        effective_safety_factor = safety_factor if safety_factor != 1.0 else config.get("token_safety_factor_heuristic", 1.2)

    if budget is None:
        budget = TokenBudget(
            full_cap=config.get("token_full_cap", 8000),
            summary_min=config.get("token_summary_min", 100),
            overhead_reserve=config.get("token_overhead_reserve", 500),
        )

    # Count tokens with counter
    raw_tokens = counter.count_messages(messages)
    original_tokens = int(raw_tokens * effective_safety_factor)

    # Build breakdown (uses internal estimation for categorization)
    _, breakdown = estimate_tokens_messages(messages, effective_safety_factor)

    result = ValidationResult(
        valid=True,
        original_tokens=original_tokens,
        final_tokens=original_tokens,
        messages=messages,
        details={
            "breakdown": {
                "system_prompt": breakdown.system_prompt,
                "summary": breakdown.summary,
                "anchors": breakdown.anchors,
                "recency": breakdown.recency,
                "rag_context": breakdown.rag_context,
                "web_context": breakdown.web_context,
                "user_message": breakdown.user_message,
                "other": breakdown.other,
            },
            "budget": {
                "full_cap": budget.full_cap,
                "summary_min": budget.summary_min,
                "overhead_reserve": budget.overhead_reserve,
            },
            # Gap A: Log counter info
            "counter_backend": counter.backend_name(),
            "counter_exact": counter.is_exact(),
            "applied_safety_factor": effective_safety_factor,
            "raw_tokens": raw_tokens,
        }
    )

    # Resolve relevance truncation setting from config if not explicitly provided
    if enable_relevance_truncation is None:
        enable_relevance_truncation = config.get("enable_relevance_truncation", False)

    # Fail-fast invariant: If relevance truncation is enabled, anchor_indices MUST be provided
    # This prevents silent loss of anchor priority when truncation is active
    if enable_relevance_truncation and (anchor_indices is None or len(anchor_indices) == 0):
        raise ValueError(
            "enable_relevance_truncation=True requires anchor_indices to be provided and non-empty. "
            "Without anchor_indices, anchor priority cannot be enforced during truncation."
        )

    # Check if within cap
    effective_cap = budget.full_cap - budget.overhead_reserve
    if original_tokens <= effective_cap:
        # Category D: Emit event for successful validation too
        if emit_event:
            _emit_token_guard_event(
                result=result,
                assembly_id=assembly_id,
                turn_id=turn_id,
                counter=counter,
                raw_tokens=raw_tokens,
                effective_cap=effective_cap,
                effective_safety_factor=effective_safety_factor,
            )
        return result

    # Over cap - need to reduce
    current_messages = messages.copy() if apply_drops else messages
    current_tokens = original_tokens
    target = effective_cap

    # Phase 2: Relevance-aware truncation (when enabled)
    # This runs BEFORE the legacy drop policy as the preferred approach
    if enable_relevance_truncation and apply_drops and current_tokens > target:
        try:
            from episodic.truncation import truncate_by_relevance

            # Extract current query from last user message if not provided
            query = current_query
            if query is None:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            query = content
                        elif isinstance(content, list):
                            query = " ".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in content
                            )
                        break
                if query is None:
                    query = ""

            # Run relevance truncation with same counter for consistency
            truncation_result = truncate_by_relevance(
                messages=current_messages,
                target_tokens=target,
                current_query=query,
                counter=counter,
                anchor_indices=anchor_indices,
            )

            # Check if truncation made progress
            if truncation_result.tokens_after < truncation_result.tokens_before:
                result.actions_taken.append(DropAction.RELEVANCE_TRUNCATION)
                current_messages = truncation_result.messages
                # Recount with safety factor for consistency
                raw_after = counter.count_messages(current_messages)
                current_tokens = int(raw_after * effective_safety_factor)
                result.truncation_result = truncation_result
                result.details["truncation_applied"] = True
                result.details["truncation_tokens_freed"] = (
                    truncation_result.tokens_before - truncation_result.tokens_after
                )

        except ImportError:
            logger.warning("Relevance truncation enabled but truncation module not available")
        except Exception as e:
            logger.warning(f"Relevance truncation failed, falling back to legacy: {e}")

    # Step 1: Truncate summary (legacy fallback if still over)
    if apply_drops and current_tokens > target:
        reduction_needed = current_tokens - target
        new_messages, reduced = _truncate_summary_in_messages(
            current_messages, reduction_needed, budget.summary_min, counter
        )
        if reduced > 0:
            reduced_with_factor = int(reduced * effective_safety_factor)
            result.actions_taken.append(DropAction.TRUNCATE_SUMMARY)
            current_messages = new_messages
            current_tokens -= reduced_with_factor

    # Step 2: Drop recency
    if apply_drops and current_tokens > target:
        reduction_needed = current_tokens - target
        new_messages, reduced = _drop_recency_messages(current_messages, reduction_needed, counter)
        if reduced > 0:
            reduced_with_factor = int(reduced * effective_safety_factor)
            result.actions_taken.append(DropAction.DROP_RECENCY)
            current_messages = new_messages
            current_tokens -= reduced_with_factor

    # Step 3: Drop anchors
    if apply_drops and current_tokens > target:
        new_messages, reduced = _drop_anchors_from_messages(current_messages, counter)
        if reduced > 0:
            reduced_with_factor = int(reduced * effective_safety_factor)
            result.actions_taken.append(DropAction.DROP_ANCHORS)
            current_messages = new_messages
            current_tokens -= reduced_with_factor

    # Recount final tokens to be accurate
    if apply_drops:
        raw_final = counter.count_messages(current_messages)
        current_tokens = int(raw_final * effective_safety_factor)

    # Step 4: Still over - abort with fallback
    if current_tokens > target:
        result.valid = False
        result.actions_taken.append(DropAction.ABORT)
        result.fallback_response = (
            "I apologize, but I'm unable to process this request due to context "
            "size limitations. Please try rephrasing your question or starting "
            "a new conversation."
        )
        result.bug_event_logged = True

        log_bug_event(
            event_type="token_overflow_abort",
            details={
                "original_tokens": original_tokens,
                "final_tokens": current_tokens,
                "target": target,
                "actions_taken": [a.value for a in result.actions_taken],
                "breakdown": result.details["breakdown"],
                "counter_backend": counter.backend_name(),
                "counter_exact": counter.is_exact(),
                "applied_safety_factor": effective_safety_factor,
            },
            severity="error"
        )
    else:
        # Successfully reduced
        result.messages = current_messages
        result.final_tokens = current_tokens

        if result.actions_taken:
            log_bug_event(
                event_type="token_overflow_recovered",
                details={
                    "original_tokens": original_tokens,
                    "final_tokens": current_tokens,
                    "target": target,
                    "actions_taken": [a.value for a in result.actions_taken],
                    "counter_backend": counter.backend_name(),
                    "counter_exact": counter.is_exact(),
                    "applied_safety_factor": effective_safety_factor,
                },
                severity="warning"
            )

    result.details["final_tokens"] = current_tokens
    result.details["within_cap"] = current_tokens <= target
    result.details["assembly_id"] = assembly_id
    result.details["turn_id"] = turn_id

    # Category D: Emit exactly one structured event
    if emit_event:
        _emit_token_guard_event(
            result=result,
            assembly_id=assembly_id,
            turn_id=turn_id,
            counter=counter,
            raw_tokens=raw_tokens,
            effective_cap=effective_cap,
            effective_safety_factor=effective_safety_factor,
        )

    return result


def guard_assembly(
    messages: List[Dict[str, Any]],
    budget: Optional[TokenBudget] = None,
    safety_factor: float = 1.0,
    counter: Optional[TokenCounter] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    assembly_id: Optional[str] = None,
    emit_event: bool = True,
    enable_relevance_truncation: Optional[bool] = None,
    current_query: Optional[str] = None,
    anchor_indices: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Guard assembly - validate and return safe messages.

    Convenience function that either returns validated messages
    or a fallback response if validation fails.

    Args:
        messages: List of assembled messages
        budget: Token budget configuration
        safety_factor: Multiplicative safety factor (only used if counter is heuristic)
        counter: Explicit token counter (overrides registry lookup)
        provider_id: Provider ID for registry lookup
        model_id: Model ID for registry lookup
        turn_id: Unique ID for conversation turn
        assembly_id: Unique ID for this assembly call
        emit_event: Whether to emit structured event
        enable_relevance_truncation: Use importance-based truncation (default: from config)
        current_query: Current user query for relevance scoring
        anchor_indices: Set of message indices that are anchors

    Returns:
        Tuple of (messages, fallback_response)
        - If valid: (validated_messages, None)
        - If abort: ([], fallback_response_string)
    """
    result = validate_assembly(
        messages, budget, safety_factor, apply_drops=True,
        counter=counter, provider_id=provider_id, model_id=model_id,
        turn_id=turn_id, assembly_id=assembly_id, emit_event=emit_event,
        enable_relevance_truncation=enable_relevance_truncation,
        current_query=current_query, anchor_indices=anchor_indices
    )

    if result.valid:
        return result.messages or messages, None
    else:
        return [], result.fallback_response
