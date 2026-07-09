"""
Snapshot-Based Replay Infrastructure for Episodic.

Category E Implementation:
- Deterministic replay of context assembly
- Snapshot schema for inputs, retrieval state, outputs, and events
- Diff detection for divergence localization
- Hash chain verification for event stream integrity
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

from episodic.token_guard_events import (
    EventLogger,
    EventVerifier,
    canonical_json,
)
from episodic.token_guard import (
    TokenBudget,
    validate_assembly,
    token_counter_registry,
)

from episodic.replay_types import (  # noqa: F401  (re-exported)
    SNAPSHOT_SCHEMA_VERSION, RetrievalState, TokenGuardConfig, ContextInputs,
    ReplaySnapshot, ReplayDiff, ReplayResult,
)

def _snippet(value: Any, max_len: int = 100) -> str:
    """Create a snippet of a value for diff reporting."""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


def _compare_messages(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]]
) -> List[ReplayDiff]:
    """
    Compare two message lists for byte-identical equality.

    Returns list of diffs found.
    """
    diffs = []

    if len(expected) != len(actual):
        diffs.append(ReplayDiff(
            field_path="assembled_messages.length",
            expected_snippet=str(len(expected)),
            actual_snippet=str(len(actual)),
            message=f"Message count mismatch: expected {len(expected)}, got {len(actual)}"
        ))
        return diffs

    for i, (exp_msg, act_msg) in enumerate(zip(expected, actual)):
        # Compare role
        exp_role = exp_msg.get("role", "")
        act_role = act_msg.get("role", "")
        if exp_role != act_role:
            diffs.append(ReplayDiff(
                field_path=f"assembled_messages[{i}].role",
                expected_snippet=exp_role,
                actual_snippet=act_role,
                message=f"Role mismatch at message {i}"
            ))

        # Compare content (byte-identical)
        exp_content = exp_msg.get("content", "")
        act_content = act_msg.get("content", "")

        # Handle both string and list content (multimodal)
        exp_content_str = canonical_json(exp_content) if isinstance(exp_content, (list, dict)) else str(exp_content)
        act_content_str = canonical_json(act_content) if isinstance(act_content, (list, dict)) else str(act_content)

        if exp_content_str != act_content_str:
            diffs.append(ReplayDiff(
                field_path=f"assembled_messages[{i}].content",
                expected_snippet=_snippet(exp_content_str),
                actual_snippet=_snippet(act_content_str),
                message=f"Content mismatch at message {i}"
            ))

    return diffs


def _compare_events(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    ignore_timestamps: bool = True
) -> List[ReplayDiff]:
    """
    Compare two event lists.

    Timestamps are ignored by default since they use wall clock.

    Returns list of diffs found.
    """
    diffs = []

    if len(expected) != len(actual):
        diffs.append(ReplayDiff(
            field_path="token_guard_events.length",
            expected_snippet=str(len(expected)),
            actual_snippet=str(len(actual)),
            message=f"Event count mismatch: expected {len(expected)}, got {len(actual)}"
        ))
        return diffs

    # Fields to compare (exclude timestamps)
    compare_fields = [
        "schema_version", "event_type", "event_seq",
        "counter_backend", "counter_exact", "applied_safety_factor",
        "raw_tokens", "effective_tokens", "cap", "budget_breakdown",
    ]

    for i, (exp_evt, act_evt) in enumerate(zip(expected, actual)):
        for field_name in compare_fields:
            exp_val = exp_evt.get(field_name)
            act_val = act_evt.get(field_name)

            if exp_val != act_val:
                diffs.append(ReplayDiff(
                    field_path=f"token_guard_events[{i}].{field_name}",
                    expected_snippet=_snippet(exp_val),
                    actual_snippet=_snippet(act_val),
                    message=f"Event field mismatch at event {i}"
                ))

    return diffs


def assemble_from_snapshot(snapshot: ReplaySnapshot) -> List[Dict[str, Any]]:
    """
    Assemble messages from snapshot inputs.

    This is a simplified assembly that uses the frozen inputs
    without any live retrieval or database access.
    """
    messages = []

    # Add system prompt if present
    if snapshot.inputs.system_prompt:
        messages.append({
            "role": "system",
            "content": snapshot.inputs.system_prompt
        })

    # Add summary if present
    if snapshot.inputs.summary_text:
        # Check if there's already a system message to append to
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n## Summary\n{snapshot.inputs.summary_text}"
        else:
            messages.insert(0, {
                "role": "system",
                "content": f"## Summary\n{snapshot.inputs.summary_text}"
            })

    # Add anchor exchanges if present
    if snapshot.inputs.anchor_exchanges:
        anchor_text = "\n\n".join([
            f"{ex['role']}: {ex['content']}"
            for ex in snapshot.inputs.anchor_exchanges
        ])
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n## Relevant Past Context\n{anchor_text}"
        else:
            messages.insert(0, {
                "role": "system",
                "content": f"## Relevant Past Context\n{anchor_text}"
            })

    # Add RAG context if present
    if snapshot.inputs.rag_context:
        messages.append({
            "role": "system",
            "content": f"Relevant context from knowledge base:\n\n{snapshot.inputs.rag_context}"
        })

    # Add recency exchanges
    for ex in snapshot.inputs.recency_exchanges:
        messages.append({
            "role": ex.get("role", "user"),
            "content": ex.get("content", "")
        })

    # Add current user message
    messages.append({
        "role": "user",
        "content": snapshot.inputs.user_turn_text
    })

    return messages


def replay(
    snapshot_path: Union[str, Path],
    use_stored_messages: bool = True
) -> ReplayResult:
    """
    Replay a snapshot and verify outputs match.

    Args:
        snapshot_path: Path to the snapshot JSON file
        use_stored_messages: If True, use stored assembled_messages directly.
                            If False, re-assemble from inputs (more comprehensive test).

    Returns:
        ReplayResult with success flag and any diffs found
    """
    # Load snapshot
    snapshot = ReplaySnapshot.load(snapshot_path)

    # Initialize result
    result = ReplayResult(
        success=True,
        snapshot=snapshot,
    )

    # Determine which messages to use for replay
    if use_stored_messages:
        # Use the stored assembled messages directly
        replay_messages = snapshot.assembled_messages
    else:
        # Re-assemble from inputs
        replay_messages = assemble_from_snapshot(snapshot)

    result.replayed_messages = replay_messages

    # Get or create token counter
    counter = token_counter_registry.get(snapshot.provider_id, snapshot.model_id)
    result.counter_backend_used = counter.backend_name()

    # Check if counter matches expected
    if counter.backend_name() != snapshot.tokenizer_backend_name:
        result.counter_verified = False

    # Create in-memory event logger for replay
    from io import StringIO
    event_output = StringIO()
    replay_logger = EventLogger(
        output=event_output,
        run_id=snapshot.run_id
    )

    # Configure global logger to use our replay logger
    from episodic.token_guard_events import configure_event_logger, reset_event_logger

    # Use a unique assembly_id for replay to avoid exactly-once conflict
    replay_assembly_id = f"replay-{snapshot.turn_id}"

    # Run token guard validation
    budget = snapshot.token_guard_config.to_budget()
    safety_factor = snapshot.safety_factor_config if not snapshot.exact_flag else 1.0

    # Temporarily configure global logger
    old_logger = configure_event_logger(
        output=event_output,
        run_id=snapshot.run_id
    )

    try:
        validation_result = validate_assembly(
            messages=replay_messages,
            budget=budget,
            safety_factor=safety_factor,
            counter=counter,
            turn_id=snapshot.turn_id,
            assembly_id=replay_assembly_id,
            emit_event=True,
        )

        result.replayed_token_count = validation_result.final_tokens

        # Get emitted events
        from episodic.token_guard_events import get_event_logger
        replayed_events = [e.to_dict() for e in get_event_logger().events]
        result.replayed_events = replayed_events

    finally:
        reset_event_logger()

    # Compare messages (only if not using stored - if using stored, they're identical by definition)
    if not use_stored_messages:
        msg_diffs = _compare_messages(snapshot.assembled_messages, replay_messages)
        if msg_diffs:
            result.messages_match = False
            result.all_diffs.extend(msg_diffs)
            if result.first_diff is None:
                result.first_diff = msg_diffs[0]

    # Compare token counts
    expected_tokens = 0
    if snapshot.token_guard_events:
        # Get token count from stored events
        for evt in snapshot.token_guard_events:
            if "effective_tokens" in evt:
                expected_tokens = evt["effective_tokens"]
                break

    if expected_tokens > 0 and result.replayed_token_count != expected_tokens:
        result.tokens_match = False
        diff = ReplayDiff(
            field_path="token_count",
            expected_snippet=str(expected_tokens),
            actual_snippet=str(result.replayed_token_count),
            message="Token count mismatch"
        )
        result.all_diffs.append(diff)
        if result.first_diff is None:
            result.first_diff = diff

    # Compare events (excluding timestamps and IDs that will differ)
    if snapshot.token_guard_events:
        evt_diffs = _compare_events(
            snapshot.token_guard_events,
            result.replayed_events,
            ignore_timestamps=True
        )
        if evt_diffs:
            result.events_match = False
            result.all_diffs.extend(evt_diffs)
            if result.first_diff is None:
                result.first_diff = evt_diffs[0]

    # Verify hash chain on stored events
    if snapshot.token_guard_events:
        verification = EventVerifier.verify_stream(snapshot.token_guard_events)
        if not verification["valid"]:
            result.hash_chain_valid = False
            diff = ReplayDiff(
                field_path="token_guard_events.hash_chain",
                expected_snippet="valid",
                actual_snippet="invalid",
                message=f"Hash chain verification failed: {verification['errors']}"
            )
            result.all_diffs.append(diff)
            if result.first_diff is None:
                result.first_diff = diff

    # Final success check
    result.success = (
        result.messages_match and
        result.tokens_match and
        result.events_match and
        result.hash_chain_valid
    )

    return result


def create_snapshot(
    user_turn_text: str,
    assembled_messages: List[Dict[str, Any]],
    token_guard_events: List[Dict[str, Any]],
    run_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    inputs: Optional[ContextInputs] = None,
    retrieval: Optional[RetrievalState] = None,
    token_guard_config: Optional[TokenGuardConfig] = None,
    tokenizer_backend_name: str = "heuristic_chars_div_4",
    exact_flag: bool = False,
    safety_factor: float = 1.2,
) -> ReplaySnapshot:
    """
    Create a snapshot from current assembly state.

    Helper function to create snapshots during normal operation.
    """
    # Generate IDs if not provided
    if run_id is None:
        run_id = str(uuid.uuid4())
    if turn_id is None:
        turn_id = str(uuid.uuid4())

    # Default inputs from user_turn_text
    if inputs is None:
        inputs = ContextInputs(user_turn_text=user_turn_text)

    # Default retrieval state
    if retrieval is None:
        retrieval = RetrievalState()

    # Default token guard config
    if token_guard_config is None:
        token_guard_config = TokenGuardConfig()

    # Get final event hash
    final_hash = None
    if token_guard_events:
        final_hash = token_guard_events[-1].get("hash")

    return ReplaySnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        run_id=run_id,
        turn_id=turn_id,
        provider_id=provider_id,
        model_id=model_id,
        tokenizer_backend_name=tokenizer_backend_name,
        exact_flag=exact_flag,
        safety_factor_config=safety_factor,
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        retrieval=retrieval,
        assembled_messages=assembled_messages,
        token_guard_config=token_guard_config,
        token_guard_events=token_guard_events,
        final_event_hash=final_hash,
    )
