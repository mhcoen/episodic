"""
Token Guard: Centralized token validation and assembly cap enforcement.

Full-assembly token assertion with fail-closed drop policy and bug event logging.
Token counting infrastructure lives in token_counting.py and is re-exported here
for backward compatibility.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

# Re-export token counting infrastructure for backward compatibility.
# Callers that do `from episodic.token_guard import TokenCounter` etc. still work.
from episodic.token_counting import (  # noqa: F401
    TokenCounter,
    HeuristicTokenCounter,
    _TokenCounterRegistry,
    token_counter_registry,
    get_token_counter,
    TokenEstimate,
    estimate_tokens_text,
    estimate_tokens_message,
    estimate_tokens_messages,
)

logger = logging.getLogger(__name__)


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


def log_bug_event(
    event_type: str,
    details: Dict[str, Any],
    severity: str = "warning"
) -> None:
    """Log a structured bug event for token overflow or validation failure."""
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
    """Truncate summary section to reduce tokens. Returns (messages, tokens_reduced)."""
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
    """Find indices of recency messages (user/assistant pairs excluding current query)."""
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
    """Drop recency messages oldest-first to reduce tokens. Returns (messages, tokens_reduced)."""
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
    """Drop anchor section entirely. Returns (messages, tokens_reduced)."""
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


def _emit_token_guard_event(
    result: ValidationResult,
    assembly_id: str,
    turn_id: str,
    counter: TokenCounter,
    raw_tokens: int,
    effective_cap: int,
    effective_safety_factor: float,
) -> None:
    """Emit a structured token guard event for consistent event logging."""
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
    Validate assembled messages against token cap with fail-closed drop policy.

    Drop order: relevance truncation (if enabled) -> truncate summary -> drop recency
    (oldest first) -> drop anchors -> abort with fallback.

    Safety factor only applied when counter.is_exact()==False.
    Emits exactly one structured event per call (Category D).
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

    raw_tokens = counter.count_messages(messages)
    original_tokens = int(raw_tokens * effective_safety_factor)
    _, breakdown = estimate_tokens_messages(messages, effective_safety_factor)

    breakdown_dict = {
        "system_prompt": breakdown.system_prompt, "summary": breakdown.summary,
        "anchors": breakdown.anchors, "recency": breakdown.recency,
        "rag_context": breakdown.rag_context, "web_context": breakdown.web_context,
        "user_message": breakdown.user_message, "other": breakdown.other,
    }
    budget_dict = {
        "full_cap": budget.full_cap, "summary_min": budget.summary_min,
        "overhead_reserve": budget.overhead_reserve,
    }
    result = ValidationResult(
        valid=True, original_tokens=original_tokens, final_tokens=original_tokens,
        messages=messages,
        details={
            "breakdown": breakdown_dict, "budget": budget_dict,
            "counter_backend": counter.backend_name(),
            "counter_exact": counter.is_exact(),
            "applied_safety_factor": effective_safety_factor,
            "raw_tokens": raw_tokens,
        }
    )

    if enable_relevance_truncation is None:
        enable_relevance_truncation = config.get("enable_relevance_truncation", False)

    # Fail-fast: relevance truncation requires anchor_indices for priority enforcement
    if enable_relevance_truncation and (anchor_indices is None or len(anchor_indices) == 0):
        raise ValueError(
            "enable_relevance_truncation=True requires anchor_indices to be provided and non-empty. "
            "Without anchor_indices, anchor priority cannot be enforced during truncation."
        )

    effective_cap = budget.full_cap - budget.overhead_reserve
    if original_tokens <= effective_cap:
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

    current_messages = messages.copy() if apply_drops else messages
    current_tokens = original_tokens
    target = effective_cap

    # Phase 2: Relevance-aware truncation (runs before legacy drop policy)
    if enable_relevance_truncation and apply_drops and current_tokens > target:
        try:
            from episodic.truncation import truncate_by_relevance

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

            truncation_result = truncate_by_relevance(
                messages=current_messages,
                target_tokens=target,
                current_query=query,
                counter=counter,
                anchor_indices=anchor_indices,
            )

            if truncation_result.tokens_after < truncation_result.tokens_before:
                result.actions_taken.append(DropAction.RELEVANCE_TRUNCATION)
                current_messages = truncation_result.messages
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

    # Legacy drop policy: summary -> recency -> anchors -> abort
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

    if apply_drops and current_tokens > target:
        reduction_needed = current_tokens - target
        new_messages, reduced = _drop_recency_messages(current_messages, reduction_needed, counter)
        if reduced > 0:
            reduced_with_factor = int(reduced * effective_safety_factor)
            result.actions_taken.append(DropAction.DROP_RECENCY)
            current_messages = new_messages
            current_tokens -= reduced_with_factor

    if apply_drops and current_tokens > target:
        new_messages, reduced = _drop_anchors_from_messages(current_messages, counter)
        if reduced > 0:
            reduced_with_factor = int(reduced * effective_safety_factor)
            result.actions_taken.append(DropAction.DROP_ANCHORS)
            current_messages = new_messages
            current_tokens -= reduced_with_factor

    if apply_drops:
        raw_final = counter.count_messages(current_messages)
        current_tokens = int(raw_final * effective_safety_factor)

    if current_tokens > target:
        result.valid = False
        result.actions_taken.append(DropAction.ABORT)
        result.fallback_response = (
            "I apologize, but I'm unable to process this request due to context "
            "size limitations. Please try rephrasing your question or starting "
            "a new conversation."
        )
        result.bug_event_logged = True

        _bug_details = {
            "original_tokens": original_tokens, "final_tokens": current_tokens,
            "target": target, "actions_taken": [a.value for a in result.actions_taken],
            "breakdown": result.details["breakdown"],
            "counter_backend": counter.backend_name(), "counter_exact": counter.is_exact(),
            "applied_safety_factor": effective_safety_factor,
        }
        log_bug_event("token_overflow_abort", _bug_details, severity="error")
    else:
        result.messages = current_messages
        result.final_tokens = current_tokens

        if result.actions_taken:
            _bug_details = {
                "original_tokens": original_tokens, "final_tokens": current_tokens,
                "target": target, "actions_taken": [a.value for a in result.actions_taken],
                "counter_backend": counter.backend_name(), "counter_exact": counter.is_exact(),
                "applied_safety_factor": effective_safety_factor,
            }
            log_bug_event("token_overflow_recovered", _bug_details, severity="warning")

    result.details["final_tokens"] = current_tokens
    result.details["within_cap"] = current_tokens <= target
    result.details["assembly_id"] = assembly_id
    result.details["turn_id"] = turn_id

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
    Validate and return safe messages, or a fallback response if validation fails.

    Returns (validated_messages, None) if valid, or ([], fallback_string) if abort.
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
