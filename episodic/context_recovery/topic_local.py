"""
Topic-local context recovery strategy.

Assembles context only from the active topic, excluding all other topics.
This enables "year-later" resume without contamination from intervening topics.

Supports explicit cross-topic imports when user references another topic.

Budget Model (Patches A, B, C, D):
- T: Global target for assembled context (config: context_target_T)
- M: Memory budget = T - overhead (system prompt, topic header, summary, user message)
- A: Anchor budget = clamp(a_min, floor(alpha * M), a_max)
- R: Recency budget = M - A

Selection Order (Patch D): Anchors first, then fill recency from non-anchor exchanges.
"""

import sqlite3
import time
from typing import List, Dict, Any, Optional, Tuple
import logging

from .strategy import ContextAssemblyResult, ContextRecoveryMode
from .anchor_retrieval import (
    retrieve_anchors_budgeted,
    build_recency_budgeted,
    emit_assembly_log,
    fallback_to_ancestry,
    cosine_similarity,
    check_import_intent,
)

logger = logging.getLogger(__name__)


class ContaminationError(Exception):
    """Raised when topic_local context contains nodes from other topics."""
    pass


def _assert_no_contamination(
    included_node_ids: List[str],
    active_topic_start_node_id: str,
    conn: sqlite3.Connection
) -> None:
    """
    Assert that all included nodes belong to the active topic.

    Raises ContaminationError in debug mode.
    Logs warning in production.

    Args:
        included_node_ids: List of node IDs included in context
        active_topic_start_node_id: The active topic's start node ID
        conn: Database connection
    """
    from episodic.config import config

    if not active_topic_start_node_id:
        return

    if not included_node_ids:
        return

    from episodic.db_topic_nodes import get_node_topic

    foreign_nodes = []
    for node_id in included_node_ids:
        node_topic = get_node_topic(node_id, conn=conn)
        # Only flag as foreign if it belongs to a different topic
        # Nodes not in any topic (None) are allowed
        if node_topic and node_topic != active_topic_start_node_id:
            foreign_nodes.append((node_id, node_topic))

    if foreign_nodes:
        msg = f"Contamination detected: {len(foreign_nodes)} nodes from foreign topics in topic_local context. "
        msg += f"Active topic: {active_topic_start_node_id[:8]}. "
        msg += f"Foreign: {[(n[:8], t[:8]) for n, t in foreign_nodes[:5]]}"

        if config.get("debug"):
            raise ContaminationError(msg)
        # Don't log at warning level - this is internal diagnostics, not user-actionable


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (chars / 4)."""
    return len(text) // 4 if text else 0


def _truncate_to_tokens(text: str, max_tokens: int) -> Tuple[str, bool]:
    """
    Truncate text to fit within token budget.

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if not text:
        return "", False

    current_tokens = _estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text, False

    # Truncate to approximate char limit
    max_chars = max_tokens * 4
    truncated = text[:max_chars - 3] + "..."
    return truncated, True


# Default number of exchange pairs to include (count-based fallback)
DEFAULT_EXCHANGE_PAIRS = 4  # 4 pairs = 8 messages


class TopicLocalStrategy:
    """
    Topic-local context recovery with budget-based assembly.

    Only includes messages from the active topic, completely excluding
    messages from other topics. Uses topic_nodes and topic_working_set
    tables for efficient retrieval.

    Budget Model:
    - T: Global target (context_target_T config, default 3000)
    - Overhead: System prompt estimate + topic header + summary + current message
    - M: Memory budget = T - overhead
    - A: Anchor budget = clamp(a_min, alpha * M, a_max)
    - R: Recency budget = M - A

    Selection Order (Patch D):
    1. Retrieve anchors first (charged to A)
    2. Fill recency from most recent non-anchor exchanges (charged to R)
    """

    def __init__(
        self,
        exchange_pairs: int = DEFAULT_EXCHANGE_PAIRS,
        anchor_count: Optional[int] = None,
        anchor_similarity_threshold: Optional[float] = None
    ):
        """
        Initialize topic-local strategy.

        Args:
            exchange_pairs: Max exchange pairs for recency (count-based fallback)
            anchor_count: Number of semantic anchors to retrieve (defaults to config)
            anchor_similarity_threshold: Minimum similarity for anchors (defaults to config)
        """
        self.exchange_pairs = exchange_pairs
        self._anchor_count = anchor_count
        self._anchor_similarity_threshold = anchor_similarity_threshold

    @property
    def anchor_count(self) -> int:
        """Get anchor count from config if not set."""
        if self._anchor_count is not None:
            return self._anchor_count
        from episodic.config import config
        return config.get("anchor_count", 3)

    @property
    def anchor_similarity_threshold(self) -> float:
        """Get anchor similarity threshold from config if not set."""
        if self._anchor_similarity_threshold is not None:
            return self._anchor_similarity_threshold
        from episodic.config import config
        return config.get("anchor_similarity_threshold", 0.5)

    def _compute_budgets(self, token_budget: int) -> Dict[str, int]:
        """
        Compute budget allocations using the two-level model.

        Returns dict with keys: T, overhead, M, A, R, s_max, alpha
        """
        from episodic.config import config

        # Global target T (use provided token_budget as override, else config)
        T = token_budget or config.get("context_target_T", 3000)

        # Overhead estimate for fixed costs
        overhead = config.get("context_overhead_estimate", 500)

        # Summary cap
        s_max = config.get("summary_max_tokens", 400)

        # Memory budget M = T - overhead - s_max (summary counted separately)
        M = max(0, T - overhead - s_max)

        # Alpha and anchor bounds
        alpha = config.get("anchor_alpha", 0.25)
        a_min = config.get("anchor_a_min", 200)
        a_max = config.get("anchor_a_max", 1000)

        # A = clamp(a_min, floor(alpha * M), a_max)
        A_raw = int(alpha * M)
        A = max(a_min, min(A_raw, a_max))

        # R = M - A (recency gets remainder)
        R = max(0, M - A)

        return {
            "T": T,
            "overhead": overhead,
            "M": M,
            "A": A,
            "R": R,
            "s_max": s_max,
            "alpha": alpha,
            "a_min": a_min,
            "a_max": a_max,
        }

    def assemble(
        self,
        user_turn_text: str,
        user_node_id: Optional[str],
        active_topic_start_node_id: Optional[str],
        user_embedding: Optional[Any],
        token_budget: int,
        conn: Optional[sqlite3.Connection] = None,
        chroma_collection: Optional[Any] = None,
        force_no_recency: bool = False,
    ) -> ContextAssemblyResult:
        """
        Assemble context from only the active topic using budget-based allocation.

        Order of assembly (Patch D - anchors first):
        1. Compute budgets (T, M, A, R, s_max)
        2. Retrieve anchors (charged to A budget)
        3. Build topic header + truncated summary (charged to overhead + s_max)
        4. Fill recency from non-anchor exchanges (charged to R budget)
        5. Combine and emit structured log

        Args:
            force_no_recency: If True, skip loading recent exchanges (for year-later testing)
        """
        import os
        from episodic.config import config
        from episodic.db_connection import get_connection

        # Guard: force_no_recency is test-only
        if force_no_recency:
            if not (os.environ.get("EPISODIC_TEST_MODE") or config.get("debug")):
                raise ValueError("force_no_recency is only allowed in test/debug mode")

        # Start timing
        assembly_start = time.perf_counter()
        timing = {
            "sqlite_ops_ms": 0.0,
            "chroma_query_ms": 0.0,
            "context_assembly_ms": 0.0,
        }

        # Compute budgets (Patch A)
        budgets = self._compute_budgets(token_budget)

        messages = []
        included_node_ids = []

        # Initialize comprehensive debug struct (Patch E)
        debug = self._init_debug_struct(active_topic_start_node_id, budgets, timing)

        if active_topic_start_node_id is None:
            debug["reactivation_fired"] = False
            debug["timing"]["context_assembly_ms"] = (time.perf_counter() - assembly_start) * 1000
            self._emit_assembly_log(debug)
            return ContextAssemblyResult(messages=messages, debug=debug)

        # Use provided connection or get new one
        if conn is not None:
            result = self._assemble_with_conn(
                conn=conn,
                user_turn_text=user_turn_text,
                user_node_id=user_node_id,
                active_topic_start_node_id=active_topic_start_node_id,
                user_embedding=user_embedding,
                chroma_collection=chroma_collection,
                force_no_recency=force_no_recency,
                budgets=budgets,
                debug=debug,
                timing=timing,
                assembly_start=assembly_start,
            )
        else:
            with get_connection() as c:
                result = self._assemble_with_conn(
                    conn=c,
                    user_turn_text=user_turn_text,
                    user_node_id=user_node_id,
                    active_topic_start_node_id=active_topic_start_node_id,
                    user_embedding=user_embedding,
                    chroma_collection=chroma_collection,
                    force_no_recency=force_no_recency,
                    budgets=budgets,
                    debug=debug,
                    timing=timing,
                    assembly_start=assembly_start,
                )

        return result

    @staticmethod
    def _init_debug_struct(
        active_topic_start_node_id: Optional[str],
        budgets: Dict[str, int],
        timing: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build the initial debug/diagnostics dictionary."""
        return {
            "mode": ContextRecoveryMode.TOPIC_LOCAL.value,
            "topic_start_node_id": active_topic_start_node_id,
            "included_node_ids": [],
            "budgets": budgets,
            "token_counts": {},
            "token_breakdown": {
                "overhead_tokens": 0,
                "summary_tokens": 0,
                "anchor_tokens": 0,
                "recency_tokens": 0,
                "import_tokens": 0,
                "total_tokens": 0,
            },
            "timing": timing,
            "truncation_info": {
                "summary_truncated": False,
                "recency_truncated": False,
                "anchors_dropped": [],
                "recency_dropped": [],
            },
            "reactivation_fired": True,
            "working_set_used": False,
            "summary_included": False,
            "anchors": {
                "retrieved_count": 0,
                "deduped_recency": 0,
                "deduped_similarity": 0,
                "deduped_summary_redundant": 0,
                "included_count": 0,
                "included_node_ids": [],
                "similarities": [],
                "tokens_used": 0,
            },
            "recency": {
                "candidate_count": 0,
                "included_count": 0,
                "included_node_ids": [],
                "tokens_used": 0,
                "skipped_as_anchors": [],
            },
            "import": {
                "detected_intent": False,
                "topic_reference": None,
                "resolved_topic": None,
                "confidence": 0.0,
                "context_tokens": 0,
                "anchors_included": False,
            },
        }

    def _assemble_with_conn(
        self,
        conn: sqlite3.Connection,
        user_turn_text: str,
        user_node_id: Optional[str],
        active_topic_start_node_id: str,
        user_embedding: Optional[Any],
        chroma_collection: Optional[Any],
        force_no_recency: bool,
        budgets: Dict[str, int],
        debug: Dict[str, Any],
        timing: Dict[str, float],
        assembly_start: float,
    ) -> ContextAssemblyResult:
        """Core assembly logic with database connection."""
        from episodic.config import config
        from episodic.db_topic_nodes import (
            get_last_n_exchanges_in_topic,
            get_topic_working_set,
        )

        messages = []

        # Step 1: Get topic working set (summary, name)
        sqlite_start = time.perf_counter()
        working_set = get_topic_working_set(active_topic_start_node_id, conn=conn)

        topic_name = "Unknown Topic"
        summary_text = ""
        if working_set:
            topic_name = working_set.get("topic_name", "Unknown Topic")
            summary_text = working_set.get("summary_md", "") or ""

        # Step 2: ANCHORS FIRST (Patch D) - retrieve before recency
        chroma_start = time.perf_counter()
        anchor_node_ids, anchor_context, anchor_debug = retrieve_anchors_budgeted(
            user_turn_text=user_turn_text,
            user_embedding=user_embedding,
            active_topic_start_node_id=active_topic_start_node_id,
            summary_text=summary_text,
            chroma_collection=chroma_collection,
            anchor_budget=budgets["A"],
            anchor_count=self.anchor_count,
            anchor_similarity_threshold=self.anchor_similarity_threshold,
        )
        timing["chroma_query_ms"] = (time.perf_counter() - chroma_start) * 1000
        debug["anchors"] = anchor_debug

        # Step 3: Get recency candidates (all recent exchanges in topic)
        all_exchanges = []
        if not force_no_recency:
            # Get more than we might need, we'll filter by budget
            all_exchanges = get_last_n_exchanges_in_topic(
                active_topic_start_node_id,
                n=self.exchange_pairs * 2,  # Over-fetch for filtering
                conn=conn
            )

        timing["sqlite_ops_ms"] = (time.perf_counter() - sqlite_start) * 1000

        # Step 4: Fill recency from non-anchor exchanges (Patch B + D)
        recency_messages, recency_node_ids, recency_debug = build_recency_budgeted(
            all_exchanges=all_exchanges,
            anchor_node_ids=anchor_node_ids,
            recency_budget=budgets["R"],
        )
        debug["recency"] = recency_debug

        # Combine all included node IDs
        included_node_ids = list(anchor_node_ids) + list(recency_node_ids)
        debug["included_node_ids"] = included_node_ids

        # Step 5: Build topic context (header + truncated summary - Patch C)
        topic_header = f"# Topic: {topic_name}"
        overhead_tokens = _estimate_tokens(topic_header)

        summary_truncated, summary_injected = "", ""
        if summary_text.strip():
            summary_truncated, was_truncated = _truncate_to_tokens(
                summary_text.strip(),
                budgets["s_max"]
            )
            summary_injected = f"\n## Summary\n{summary_truncated}"
            debug["truncation_info"]["summary_truncated"] = was_truncated
            debug["summary_included"] = True

        summary_tokens = _estimate_tokens(summary_injected)

        # Step 6: Check for cross-topic imports
        import_context = ""
        import_enabled = config.get("import_detection_enabled", True)

        if import_enabled and user_turn_text:
            import_context, import_debug = check_import_intent(
                user_turn_text=user_turn_text,
                user_embedding=user_embedding,
                active_topic_start_node_id=active_topic_start_node_id,
                token_budget=config.get("import_token_budget", 100),
                conn=conn,
                chroma_collection=chroma_collection,
            )
            debug["import"] = import_debug

        import_tokens = _estimate_tokens(import_context)

        # Step 7: Assemble final messages
        # System message with topic context + anchors + imports
        context_parts = [topic_header]
        if summary_injected:
            context_parts.append(summary_injected)
        if anchor_context:
            context_parts.append(f"\n\n## Relevant Past Context\n{anchor_context}")
        if import_context:
            context_parts.append(f"\n\n{import_context}")

        combined_context = "".join(context_parts)

        if combined_context.strip():
            messages.append({
                "role": "system",
                "content": combined_context
            })
            debug["working_set_used"] = True

        # Add recency messages
        for msg in recency_messages:
            messages.append(msg)

        # Step 8: Compute final token breakdown
        anchor_tokens = anchor_debug.get("tokens_used", 0)
        recency_tokens = recency_debug.get("tokens_used", 0)
        total_tokens = overhead_tokens + summary_tokens + anchor_tokens + recency_tokens + import_tokens

        debug["token_breakdown"] = {
            "overhead_tokens": overhead_tokens,
            "summary_tokens": summary_tokens,
            "anchor_tokens": anchor_tokens,
            "recency_tokens": recency_tokens,
            "import_tokens": import_tokens,
            "total_tokens": total_tokens,
        }

        # Legacy token_counts for backwards compatibility
        debug["token_counts"] = {
            "topic_context": overhead_tokens + summary_tokens,
            "anchor_context": anchor_tokens,
            "import_context": import_tokens,
            "conversation_history": recency_tokens,
            "total_estimate": total_tokens,
        }

        # Finalize timing
        timing["context_assembly_ms"] = (time.perf_counter() - assembly_start) * 1000

        # Assert no contamination
        _assert_no_contamination(included_node_ids, active_topic_start_node_id, conn)

        # Check for thin topic_local - fall back to ancestry if insufficient
        has_summary = debug.get("summary_included", False)
        anchor_count_val = anchor_debug.get("included_count", 0)

        min_anchors = config.get("min_anchors_for_topic_local", 2)
        min_tokens = config.get("min_tokens_for_topic_local", 500)

        if (not has_summary and
            anchor_count_val < min_anchors and
            total_tokens < min_tokens):
            debug["fallback_reason"] = "thin_topic_local"
            debug["thin_fallback_details"] = {
                "has_summary": has_summary,
                "anchor_count": anchor_count_val,
                "total_tokens": total_tokens,
                "min_anchors_threshold": min_anchors,
                "min_tokens_threshold": min_tokens,
            }
            logger.debug(
                f"Thin topic_local detected (no summary, {anchor_count_val} anchors, "
                f"{total_tokens} tokens), falling back to ancestry"
            )
            return fallback_to_ancestry(
                user_turn_text=user_turn_text,
                user_node_id=user_node_id,
                active_topic_start_node_id=active_topic_start_node_id,
                user_embedding=user_embedding,
                token_budget=budgets["T"],
                conn=conn,
                chroma_collection=chroma_collection,
                original_debug=debug,
            )

        # Emit structured log (Patch E)
        self._emit_assembly_log(debug)

        return ContextAssemblyResult(messages=messages, debug=debug)

    # -- Backward-compatible delegation methods --
    # Tests and external code may call these as instance methods.

    def _retrieve_anchors_budgeted(self, **kwargs):
        """Delegate to module-level function. See anchor_retrieval.py."""
        return retrieve_anchors_budgeted(
            anchor_count=self.anchor_count,
            anchor_similarity_threshold=self.anchor_similarity_threshold,
            **kwargs,
        )

    @staticmethod
    def _build_recency_budgeted(**kwargs):
        """Delegate to module-level function. See anchor_retrieval.py."""
        return build_recency_budgeted(**kwargs)

    @staticmethod
    def _emit_assembly_log(debug):
        """Delegate to module-level function. See anchor_retrieval.py."""
        return emit_assembly_log(debug)

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Delegate to module-level function. See anchor_retrieval.py."""
        return cosine_similarity(vec1, vec2)

    @staticmethod
    def _fallback_to_ancestry(**kwargs):
        """Delegate to module-level function. See anchor_retrieval.py."""
        return fallback_to_ancestry(**kwargs)

    @staticmethod
    def _check_import_intent(**kwargs):
        """Delegate to module-level function. See anchor_retrieval.py."""
        return check_import_intent(**kwargs)
