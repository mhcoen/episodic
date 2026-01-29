"""
Context recovery strategy interface for Episodic.

Defines the protocol and selection logic for context assembly strategies.
"""

from enum import Enum
from typing import Protocol, List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import sqlite3

if TYPE_CHECKING:
    from episodic.recall.reactivation import ReactivationDecision


class ContextRecoveryMode(Enum):
    """Available context recovery strategies."""
    ANCESTRY = "ancestry"           # Traditional DAG ancestry traversal
    TOPIC_LOCAL = "topic_local"     # Topic-isolated context only
    HYBRID = "hybrid"               # Switches based on reactivation
    PACKET_ADAPTIVE = "packet_adaptive"  # Future: adaptive packet-based


@dataclass
class ContextAssemblyResult:
    """Result of context assembly."""
    messages: List[Dict[str, str]]  # role/content dicts for LLM
    debug: Dict[str, Any] = field(default_factory=dict)  # Instrumentation data

    # Debug fields typically include:
    # - mode: ContextRecoveryMode used
    # - topic_start_node_id: Active topic (if any)
    # - included_node_ids: List of node IDs in context
    # - token_counts: Dict of section -> token estimate
    # - truncation_info: What was dropped (if any)
    # - reactivation_fired: Whether topic reactivation triggered
    # - reactivation_reason: Why reactivation fired/didn't


class ContextRecoveryStrategy(Protocol):
    """Protocol for context recovery strategies."""

    def assemble(
        self,
        user_turn_text: str,
        user_node_id: Optional[str],
        active_topic_start_node_id: Optional[str],
        user_embedding: Optional[Any],
        token_budget: int,
        conn: Optional[sqlite3.Connection] = None,
        chroma_collection: Optional[Any] = None,
    ) -> ContextAssemblyResult:
        """
        Assemble context for the LLM.

        Args:
            user_turn_text: Current user message
            user_node_id: ID of the current user node (if already created)
            active_topic_start_node_id: Start node ID of active topic
            user_embedding: Pre-computed embedding for the user message
            token_budget: Maximum tokens for context
            conn: Optional SQLite connection (uses default if not provided)
            chroma_collection: Optional Chroma collection for retrieval

        Returns:
            ContextAssemblyResult with messages and debug info
        """
        ...


def select_strategy(
    mode: ContextRecoveryMode,
    reactivation_decision: Optional['ReactivationDecision'] = None
) -> ContextRecoveryStrategy:
    """
    Select the appropriate context recovery strategy.

    Args:
        mode: The configured context recovery mode
        reactivation_decision: Result of reactivation probe (for hybrid mode)

    Returns:
        An instance of the appropriate strategy
    """
    from .ancestry import AncestryStrategy
    from .topic_local import TopicLocalStrategy

    if mode == ContextRecoveryMode.ANCESTRY:
        return AncestryStrategy()

    elif mode == ContextRecoveryMode.TOPIC_LOCAL:
        return TopicLocalStrategy()

    elif mode == ContextRecoveryMode.HYBRID:
        # Hybrid mode: switch based on reactivation
        if reactivation_decision is not None:
            if reactivation_decision.action == "REACTIVATE":
                return TopicLocalStrategy()
        # Default to ancestry if no reactivation
        return AncestryStrategy()

    elif mode == ContextRecoveryMode.PACKET_ADAPTIVE:
        # Future: return packet-adaptive strategy
        # For now, fall back to ancestry
        return AncestryStrategy()

    else:
        # Unknown mode, fall back to ancestry
        return AncestryStrategy()


def get_mode_from_config() -> ContextRecoveryMode:
    """Get the context recovery mode from config."""
    from episodic.config import config

    mode_str = config.get("context_recovery_mode", "ancestry")

    try:
        return ContextRecoveryMode(mode_str)
    except ValueError:
        # Invalid mode, default to ancestry
        return ContextRecoveryMode.ANCESTRY
