"""
Conversation pipeline: phased execution of handle_chat_message().

Each phase is a standalone function taking (manager, ctx) where:
  - manager is the ConversationManager instance (provides self.* access)
  - ctx is a TurnContext dataclass carrying inter-phase data

The orchestrator in conversation.py calls these phases sequentially.
Early returns are signaled via ctx.early_return = True.

Phases 1-6 (setup through context assembly) live here.
Phases 7-9 (message augmentation, LLM query, post-processing) live in
conversation_pipeline_llm.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from episodic.conversation import ConversationManager


@dataclass
class TurnContext:
    """Carries data between pipeline phases for a single conversational turn."""

    # --- Inputs (set by caller) ---
    user_input: str = ""
    model: str = ""
    system_message: str = ""
    context_depth: int = 10

    # --- Phase 1: setup ---
    recent_nodes: List[Dict[str, Any]] = field(default_factory=list)
    user_node_id: Optional[str] = None
    user_short_id: Optional[str] = None
    semantic_drift: Optional[float] = None

    # --- Phase 2: correction / reactivation ---
    reactivation_packet: Optional[str] = None
    reactivation_applied: bool = False
    correction_applied: bool = False

    # --- Phase 3: topic detection ---
    topic_changed: bool = False
    new_topic_name: Optional[str] = None
    topic_cost_info: Any = None
    topic_change_info: Any = None

    # --- Phase 5: memory enhancement ---
    memory_context: Optional[str] = None
    memory_indicator: Optional[str] = None

    # --- Phase 6: context assembly ---
    reactivation_decision: Any = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    raw_messages: Any = None
    rag_context: Any = None
    web_context: Any = None
    context_debug: Any = None

    # --- Phase 8: LLM query ---
    display_response: Optional[str] = None

    # --- Phase 9: post-processing ---
    assistant_node_id: Optional[str] = None
    assistant_short_id: Optional[str] = None

    # --- Flow control ---
    early_return: bool = False
    early_return_value: Optional[Tuple[Optional[str], Optional[str]]] = None


# ---------------------------------------------------------------------------
# Phase 1: Setup
# ---------------------------------------------------------------------------

def phase_setup(manager: "ConversationManager", ctx: TurnContext) -> None:
    """Insert user node, fire KG extraction, compute drift."""
    from episodic.benchmark import benchmark_resource
    from episodic.config import config
    from episodic.db import get_recent_nodes, insert_node
    from episodic.debug_utils import debug_print

    # Get recent messages for context BEFORE adding the new message
    with benchmark_resource("Database", "get recent nodes"):
        # For sliding window detection, we need more history
        detection_history_limit = 50 if config.get("use_sliding_window_detection") else 10
        ctx.recent_nodes = get_recent_nodes(limit=detection_history_limit)

    # Add the user message to the database
    with benchmark_resource("Database", "insert user node"):
        ctx.user_node_id, ctx.user_short_id = insert_node(
            ctx.user_input, manager.current_node_id, role="user"
        )

    # Real-time KG extraction (fire-and-forget, non-blocking)
    if config.get("kg_realtime", False):
        from episodic.kg.realtime import extract_node_async
        extract_node_async(ctx.user_node_id, ctx.user_input)

    # Compute semantic drift BEFORE topic detection (for hybrid trigger)
    # This allows high embedding drift to fast-path into SUSPECT state
    ctx.semantic_drift = None
    if config.get("show_drift") or config.get("use_drift_trigger", True):
        ctx.semantic_drift = manager.compute_semantic_drift(ctx.user_node_id)


# ---------------------------------------------------------------------------
# Phase 2: Correction and reactivation probing
# ---------------------------------------------------------------------------

def phase_correction_reactivation(manager: "ConversationManager", ctx: TurnContext) -> None:
    """Check correction state and probe for implicit topic reactivation."""
    from episodic.color_utils import secho_color
    from episodic.config import config
    from episodic.configuration import get_system_color
    from episodic.debug_utils import debug_print

    ctx.reactivation_packet = None
    ctx.reactivation_applied = False
    ctx.correction_applied = False

    # Check for correction to previous disambiguation (before normal reactivation probe)
    if manager.pending_correction:
        state = manager.pending_correction
        turn_idx = manager._get_turn_idx()

        # Check if state is still valid (expires after 1-2 turns)
        if turn_idx - state.turn_created > 1:
            manager.pending_correction = None  # Expired
        else:
            from episodic.recall.correction import detect_correction, resolve_correction

            is_correction, hint = detect_correction(ctx.user_input)
            if is_correction:
                new_option = resolve_correction(state, hint)
                manager.pending_correction = None  # Clear state after correction

                if new_option:
                    # Apply reactivation with corrected topic
                    ctx.reactivation_packet = manager.apply_topic_reactivation(
                        new_option.topic_name,
                        new_option.topic_start_node_id,
                        ctx.user_input,
                    )
                    if ctx.reactivation_packet:
                        ctx.reactivation_applied = True
                        ctx.correction_applied = True
                        if config.get("show_reactivation_decisions", False) or config.get("debug"):
                            secho_color(
                                f"\U0001f504 Switching to: {new_option.topic_name}",
                                fg=get_system_color(),
                            )

    # Probe for implicit topic reactivation (skip if correction was applied)
    # NOTE: should_reactivate etc. are only defined when enable_topic_reactivation
    # is True. The debug log and reactivate check below mirror the original
    # indentation structure where they sit at the outer scope.
    should_reactivate = False
    react_topic_name = None
    react_start_node_id = None

    if config.get("enable_topic_reactivation", False) and not ctx.correction_applied:
        should_reactivate, react_topic_name, react_start_node_id = (
            manager.probe_topic_reactivation(
                user_input=ctx.user_input,
                recent_nodes=ctx.recent_nodes,
                is_meta_query=False,
                is_recall_intent=False,
            )
        )

    # Debug: Log probe result if show_reactivation_decisions is enabled
    # (runs regardless of enable_topic_reactivation, matching original indentation)
    if config.get("show_reactivation_decisions") or config.get("debug"):
        decision = getattr(manager, "_last_reactivation_decision", None)
        if decision:
            debug_print(
                f"Probe result: {decision.action}, topic={decision.topic_name}, "
                f"gates_failed={decision.debug.get('gates_failed', [])}, "
                f"exit_reason={decision.debug.get('exit_reason')}",
                category="memory",
            )

    if should_reactivate and react_topic_name and react_start_node_id:
        # Check if tracker would want a new topic (dry run)
        raw_topic_changed, _, _, _ = manager.topic_handler.detect_and_handle_topic_change(
            ctx.recent_nodes,
            ctx.user_input,
            ctx.user_node_id,
            semantic_drift=ctx.semantic_drift,
            dry_run=True,
        )

        # Reactivation wins unless tracker wants new topic with weak evidence
        # (For now, always let reactivation win if probe returned positive)
        ctx.reactivation_packet = manager.apply_topic_reactivation(
            react_topic_name, react_start_node_id, ctx.user_input
        )
        if ctx.reactivation_packet:
            ctx.reactivation_applied = True
            # Show reactivation one-liner if enabled (separate from full debug)
            if config.get("show_reactivation_decisions", False):
                secho_color(
                    f"\U0001f504 Resuming topic: {react_topic_name}",
                    fg=get_system_color(),
                )
            elif config.get("debug"):
                secho_color(
                    f"\n\U0001f4cc Reactivated topic: {react_topic_name}",
                    fg=get_system_color(),
                )

    # Persist reactivation decision for calibration (if feature logging enabled)
    if config.get("reactivation_log_features", True) and hasattr(
        manager, "_last_reactivation_decision"
    ):
        try:
            from episodic.db_reactivation_decisions import persist_reactivation_decision

            persist_reactivation_decision(
                ctx.user_node_id, manager._last_reactivation_decision
            )
        except Exception as e:
            debug_print(
                f"Failed to persist reactivation decision: {e}", category="memory"
            )


# ---------------------------------------------------------------------------
# Phase 3: Topic detection
# ---------------------------------------------------------------------------

def phase_topic_detection(manager: "ConversationManager", ctx: TurnContext) -> None:
    """Detect topic change and store detection scores."""
    from episodic.config import config

    if ctx.reactivation_applied:
        # Force continue - we've already switched topics via reactivation
        ctx.topic_changed, ctx.new_topic_name, ctx.topic_cost_info, ctx.topic_change_info = (
            manager.topic_handler.detect_and_handle_topic_change(
                ctx.recent_nodes,
                ctx.user_input,
                ctx.user_node_id,
                semantic_drift=ctx.semantic_drift,
                decision_override="FORCE_CONTINUE",
            )
        )
    else:
        ctx.topic_changed, ctx.new_topic_name, ctx.topic_cost_info, ctx.topic_change_info = (
            manager.topic_handler.detect_and_handle_topic_change(
                ctx.recent_nodes,
                ctx.user_input,
                ctx.user_node_id,
                semantic_drift=ctx.semantic_drift,
            )
        )

    # Store topic detection scores for debugging
    manager.topic_handler.store_topic_detection_scores(
        ctx.recent_nodes, ctx.user_node_id, ctx.topic_cost_info, ctx.topic_changed
    )

    # Update node ID
    manager.set_current_node_id(ctx.user_node_id)


# ---------------------------------------------------------------------------
# Phase 4: Skip-LLM early return
# ---------------------------------------------------------------------------

def phase_skip_llm(manager: "ConversationManager", ctx: TurnContext) -> None:
    """Handle skip_llm_response mode (testing/debug). Sets early_return if active."""
    import typer

    from episodic.benchmark import benchmark_resource
    from episodic.color_utils import secho_color
    from episodic.config import config
    from episodic.configuration import get_system_color
    from episodic.db import insert_node
    from episodic.debug_utils import debug_print

    if not config.get("skip_llm_response", False):
        return

    # Build reactivation decision for strategy-based context assembly
    reactivation_decision = None
    if ctx.reactivation_applied:
        from episodic.recall.reactivation import ReactivationDecision

        reactivation_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name=manager.current_topic[0] if manager.current_topic else None,
            topic_start_node_id=manager.current_topic[1] if manager.current_topic else None,
            debug={"source": "probe_topic_reactivation"},
        )

    # Get active topic ID (POST-reactivation)
    active_topic_start_node_id = (
        manager.current_topic[1] if manager.current_topic else None
    )

    # Build context using strategy (even though we skip LLM, ensures "B disappears")
    _, _, _, _, context_debug = manager.context_builder.build_context_full(
        user_node_id=ctx.user_node_id,
        user_input=ctx.user_input,
        active_topic_start_node_id=active_topic_start_node_id,
        model=ctx.model,
        reactivation_decision=reactivation_decision,
        skip_rag=True,  # Skip RAG for performance when LLM is skipped
    )

    # Persist context assembly debug info
    from episodic.db_context_debug import persist_context_assembly_debug

    persist_context_assembly_debug(ctx.user_node_id, context_debug, reactivation_decision)

    # Show context restoration feedback to user
    if ctx.reactivation_applied and context_debug:
        node_count = len(context_debug.get("included_node_ids", []))
        if node_count > 0:
            topic_name = manager.current_topic[0] if manager.current_topic else "unknown"
            secho_color(
                f"\n📎 Pulled {node_count} earlier messages about: {topic_name}",
                fg=get_system_color(),
            )

    # Create a placeholder response
    display_response = "[LLM response skipped]"
    # Extract provider from model if present
    if "/" in ctx.model:
        provider, model_name = ctx.model.split("/", 1)
    else:
        provider = None
        model_name = ctx.model

    assistant_node_id, assistant_short_id = insert_node(
        display_response,
        ctx.user_node_id,
        role="assistant",
        provider=provider,
        model=ctx.model,
    )

    # Display drift if enabled
    if config.get("show_drift"):
        manager.display_semantic_drift(ctx.user_node_id, cached_drift=ctx.semantic_drift)

    # Display the skipped response message
    typer.echo("")
    secho_color(f"\U0001f916 {display_response}", fg=get_system_color())

    # Update current node
    manager.set_current_node_id(assistant_node_id)

    # Display disambiguation hint if we have pending correction state
    if manager.pending_correction and not ctx.correction_applied:
        from episodic.ui.disambiguation import format_disambiguation_hint

        mode = "voice" if config.get("voice_mode", False) else "text"
        hint = format_disambiguation_hint(manager.pending_correction.runner_ups, mode)
        if hint:
            secho_color(hint, fg=get_system_color(), dim=True)

    # Handle topic boundaries
    manager.topic_handler.handle_topic_boundaries(
        ctx.topic_changed,
        ctx.user_node_id,
        assistant_node_id,
        ctx.topic_change_info,
        ctx.new_topic_name,
    )

    # Check for first topic creation
    if (
        not ctx.topic_changed
        and not manager.current_topic
        and config.get("automatic_topic_detection")
    ):
        manager.topic_handler.check_and_create_first_topic(
            ctx.user_node_id, assistant_node_id
        )

    # Add nodes to topic_nodes table for topic-local context assembly
    manager.add_nodes_to_current_topic(ctx.user_node_id, assistant_node_id)

    # Auto-index in memory system - fire-and-forget (non-blocking)
    # Index if memory RAG OR topic reactivation is enabled (reactivation needs embeddings)
    if config.get("enable_memory_rag", False) or config.get(
        "enable_topic_reactivation", False
    ):
        from episodic.conversation import _fire_and_forget_index

        user_node = {"id": ctx.user_node_id, "content": ctx.user_input, "role": "user"}
        assistant_node = {
            "id": assistant_node_id,
            "content": display_response,
            "role": "assistant",
        }
        # Include topic_start_node_id for anchor retrieval filtering
        topic_id = manager.current_topic[1] if manager.current_topic else None
        _fire_and_forget_index(user_node, assistant_node, topic_id)

    ctx.early_return = True
    ctx.early_return_value = (assistant_node_id, display_response)


# ---------------------------------------------------------------------------
# Phase 5: Memory enhancement
# ---------------------------------------------------------------------------

def phase_memory_enhancement(manager: "ConversationManager", ctx: TurnContext) -> None:
    """Check for memory context enhancement via RAG."""
    from episodic.config import config
    from episodic.db import get_ancestry
    from episodic.debug_utils import debug_print

    ctx.memory_context = None
    ctx.memory_indicator = None

    if not config.get("enable_memory_rag", False):
        return

    try:
        # Use smart detection if enabled (Milestone 2)
        if config.get("enable_smart_memory", False):
            from episodic.rag_memory_smart import enhance_with_smart_context
            import asyncio

            # Build conversation state for smart detection
            conv_state = {
                "current_topic_name": (
                    manager.current_topic[0] if manager.current_topic else None
                ),
                "messages_since_topic_change": len(ctx.recent_nodes),
                "total_messages": (
                    len(get_ancestry(ctx.user_node_id)) if ctx.user_node_id else 0
                ),
            }

            loop = asyncio.new_event_loop()
            ctx.memory_context, ctx.memory_indicator = loop.run_until_complete(
                enhance_with_smart_context(ctx.user_input, conv_state)
            )
            loop.close()
        else:
            # Original referential detection (Milestone 1)
            from episodic.rag_memory_sqlite import enhance_with_memory_context
            import asyncio

            loop = asyncio.new_event_loop()
            ctx.memory_context = loop.run_until_complete(
                enhance_with_memory_context(ctx.user_input)
            )
            loop.close()

        if ctx.memory_context:
            debug_print(
                "Added relevant context from past conversations", category="memory"
            )
    except Exception as e:
        debug_print(f"Context enhancement error: {e}", category="memory")


# ---------------------------------------------------------------------------
# Phase 6: Context assembly
# ---------------------------------------------------------------------------

def phase_context_assembly(manager: "ConversationManager", ctx: TurnContext) -> None:
    """Build conversation context using strategy-based assembly."""
    from episodic.config import config

    # Build reactivation decision for strategy-based context assembly
    ctx.reactivation_decision = None
    if ctx.reactivation_applied:
        from episodic.recall.reactivation import ReactivationDecision

        ctx.reactivation_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name=manager.current_topic[0] if manager.current_topic else None,
            topic_start_node_id=(
                manager.current_topic[1] if manager.current_topic else None
            ),
            debug={"source": "probe_topic_reactivation"},
        )

    # Get active topic ID (POST-reactivation)
    active_topic_start_node_id = (
        manager.current_topic[1] if manager.current_topic else None
    )

    # Build conversation context using strategy-based assembly
    ctx.messages, ctx.raw_messages, ctx.rag_context, ctx.web_context, ctx.context_debug = (
        manager.context_builder.build_context_full(
            user_node_id=ctx.user_node_id,
            user_input=ctx.user_input,
            active_topic_start_node_id=active_topic_start_node_id,
            model=ctx.model,
            reactivation_decision=ctx.reactivation_decision,
            skip_rag=False,
        )
    )

    # Persist context assembly debug info
    from episodic.db_context_debug import persist_context_assembly_debug

    persist_context_assembly_debug(
        ctx.user_node_id, ctx.context_debug, ctx.reactivation_decision
    )

    # Show context restoration feedback to user
    if ctx.reactivation_applied and ctx.context_debug:
        _show_context_restoration(manager, ctx)


def _show_context_restoration(manager: "ConversationManager", ctx: TurnContext) -> None:
    """Print a one-liner when context is restored from a reactivated topic."""
    from episodic.color_utils import secho_color
    from episodic.configuration import get_system_color

    node_count = len(ctx.context_debug.get("included_node_ids", []))
    if node_count == 0:
        return

    topic_name = manager.current_topic[0] if manager.current_topic else "unknown"
    secho_color(
        f"\n📎 Pulled {node_count} earlier messages about: {topic_name}",
        fg=get_system_color(),
    )


# ---------------------------------------------------------------------------
# Re-exports: phases 7-9 live in conversation_pipeline_llm.py
# Importing from here still works for backward compatibility.
# ---------------------------------------------------------------------------
from episodic.conversation_pipeline_llm import (  # noqa: E402, F401
    phase_message_augmentation,
    phase_llm_query,
    phase_postprocessing,
)
