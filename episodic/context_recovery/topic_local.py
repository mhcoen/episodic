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

import json
import os
import sqlite3
import time
from typing import List, Dict, Any, Optional, Set, Tuple
import logging

import numpy as np

from .strategy import ContextAssemblyResult, ContextRecoveryMode

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
        from episodic.db_topic_nodes import (
            get_last_n_exchanges_in_topic,
            get_topic_working_set,
        )
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
        debug = {
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
        anchor_node_ids, anchor_context, anchor_debug = self._retrieve_anchors_budgeted(
            user_turn_text=user_turn_text,
            user_embedding=user_embedding,
            active_topic_start_node_id=active_topic_start_node_id,
            summary_text=summary_text,
            chroma_collection=chroma_collection,
            anchor_budget=budgets["A"],
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
        recency_messages, recency_node_ids, recency_debug = self._build_recency_budgeted(
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
            import_context, import_debug = self._check_import_intent(
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
        anchor_count = anchor_debug.get("included_count", 0)

        min_anchors = config.get("min_anchors_for_topic_local", 2)
        min_tokens = config.get("min_tokens_for_topic_local", 500)

        if (not has_summary and
            anchor_count < min_anchors and
            total_tokens < min_tokens):
            debug["fallback_reason"] = "thin_topic_local"
            debug["thin_fallback_details"] = {
                "has_summary": has_summary,
                "anchor_count": anchor_count,
                "total_tokens": total_tokens,
                "min_anchors_threshold": min_anchors,
                "min_tokens_threshold": min_tokens,
            }
            logger.debug(
                f"Thin topic_local detected (no summary, {anchor_count} anchors, "
                f"{total_tokens} tokens), falling back to ancestry"
            )
            return self._fallback_to_ancestry(
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

    def _retrieve_anchors_budgeted(
        self,
        user_turn_text: str,
        user_embedding: Optional[Any],
        active_topic_start_node_id: str,
        summary_text: str,
        chroma_collection: Optional[Any],
        anchor_budget: int,
    ) -> Tuple[Set[str], str, Dict[str, Any]]:
        """
        Retrieve semantic anchors within budget A.

        Returns:
            Tuple of (anchor_node_ids_set, anchor_context_str, anchor_debug_dict)
        """
        anchor_debug = {
            "retrieved_count": 0,
            "deduped_recency": 0,  # No recency dedup here - anchors are first
            "deduped_similarity": 0,
            "deduped_summary_redundant": 0,
            "included_count": 0,
            "included_node_ids": [],
            "similarities": [],
            "tokens_used": 0,
            "budget": anchor_budget,
        }

        anchor_node_ids: Set[str] = set()

        if chroma_collection is None:
            try:
                from episodic.rag_collections import get_multi_collection_rag, CollectionType
                rag = get_multi_collection_rag()
                chroma_collection = rag.get_collection(CollectionType.CONVERSATION)
            except Exception as e:
                logger.debug(f"Could not get Chroma collection for anchors: {e}")
                return anchor_node_ids, "", anchor_debug

        if chroma_collection is None:
            return anchor_node_ids, "", anchor_debug

        try:
            from episodic.config import config

            # Query with topic filter
            results = chroma_collection.query(
                query_texts=[user_turn_text],
                n_results=config.get("anchor_retrieval_count", 10),
                where={"topic_start_node_id": active_topic_start_node_id},
                include=["documents", "embeddings", "distances", "metadatas"],
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                return anchor_node_ids, "", anchor_debug

            anchor_debug["retrieved_count"] = len(results["ids"][0])

            # Compute summary embedding for novelty check
            summary_embedding = None
            if summary_text:
                try:
                    embed_fn = chroma_collection._embedding_function
                    summary_embedding = embed_fn([summary_text])[0]
                except Exception as e:
                    logger.debug(f"Could not compute summary embedding: {e}")

            # Process results with filtering
            filtered_anchors = []
            included_embeddings = []
            threshold = self.anchor_similarity_threshold

            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                document = results["documents"][0][i] if results.get("documents") else ""
                embedding = (
                    results["embeddings"][0][i] if results.get("embeddings") else None
                )

                # Convert distance to similarity
                similarity = max(0, 1 - (distance / 2.0))

                user_node_id = metadata.get("user_id", doc_id)
                asst_node_id = metadata.get("assistant_id")

                # Skip if below similarity threshold
                if similarity < threshold:
                    continue

                if embedding is not None:
                    # Skip near-duplicates
                    is_duplicate = False
                    for seen in included_embeddings:
                        cosine_sim = self._cosine_similarity(embedding, seen)
                        if cosine_sim > 0.95:
                            is_duplicate = True
                            anchor_debug["deduped_similarity"] += 1
                            break
                    if is_duplicate:
                        continue

                    # Novelty check against summary
                    if summary_embedding is not None:
                        summary_sim = self._cosine_similarity(embedding, summary_embedding)
                        if summary_sim > 0.97:
                            anchor_debug["deduped_summary_redundant"] += 1
                            continue

                    included_embeddings.append(embedding)

                filtered_anchors.append({
                    "user_node_id": user_node_id,
                    "assistant_node_id": asst_node_id,
                    "document": document,
                    "similarity": similarity,
                    "user_content": metadata.get("user_content", ""),
                    "assistant_content": metadata.get("assistant_content", ""),
                })

            # Limit to anchor_count
            filtered_anchors = filtered_anchors[:self.anchor_count]

            # Build anchor context within budget A
            anchor_parts = []
            tokens_used = 0

            for anchor in filtered_anchors:
                user_content = anchor.get("user_content", "")
                asst_content = anchor.get("assistant_content", "")

                anchor_text = ""
                if user_content:
                    anchor_text += f"User: {user_content}\n"
                if asst_content:
                    anchor_text += f"Assistant: {asst_content}"

                anchor_tokens = _estimate_tokens(anchor_text)

                # Budget check
                if tokens_used + anchor_tokens > anchor_budget:
                    anchor_debug.setdefault("dropped_by_budget", []).append(anchor["user_node_id"])
                    continue

                anchor_parts.append(anchor_text.strip())
                tokens_used += anchor_tokens
                anchor_node_ids.add(anchor["user_node_id"])
                if anchor["assistant_node_id"]:
                    anchor_node_ids.add(anchor["assistant_node_id"])
                anchor_debug["included_node_ids"].append(anchor["user_node_id"])
                anchor_debug["similarities"].append(anchor["similarity"])

            anchor_debug["included_count"] = len(anchor_debug["included_node_ids"])
            anchor_debug["tokens_used"] = tokens_used

            anchor_context = "\n\n".join(anchor_parts) if anchor_parts else ""
            return anchor_node_ids, anchor_context, anchor_debug

        except Exception as e:
            logger.warning(f"Anchor retrieval failed: {e}")
            return anchor_node_ids, "", anchor_debug

    def _build_recency_budgeted(
        self,
        all_exchanges: List[Dict],
        anchor_node_ids: Set[str],
        recency_budget: int,
    ) -> Tuple[List[Dict], List[str], Dict[str, Any]]:
        """
        Build recency tail from non-anchor exchanges within budget R.

        Patch D: Exchanges that appear in anchors are skipped (charged to A).
        Patch B: Token-based truncation, not just count-based.

        Returns:
            Tuple of (messages_list, node_ids_list, recency_debug_dict)
        """
        recency_debug = {
            "candidate_count": len(all_exchanges),
            "included_count": 0,
            "included_node_ids": [],
            "tokens_used": 0,
            "skipped_as_anchors": [],
            "budget": recency_budget,
        }

        recency_messages = []
        recency_node_ids = []
        tokens_used = 0

        for ex in all_exchanges:
            user_content = ex.get("user_content", "")
            user_node_id = ex.get("user_node_id")
            asst_content = ex.get("assistant_content", "")
            asst_node_id = ex.get("assistant_node_id")

            # Patch D: Skip if this exchange is already in anchors
            if user_node_id and user_node_id in anchor_node_ids:
                recency_debug["skipped_as_anchors"].append(user_node_id)
                continue
            if asst_node_id and asst_node_id in anchor_node_ids:
                recency_debug["skipped_as_anchors"].append(asst_node_id)
                continue

            # Estimate tokens for this exchange
            exchange_tokens = _estimate_tokens(user_content) + _estimate_tokens(asst_content)

            # Patch B: Budget check
            if tokens_used + exchange_tokens > recency_budget:
                recency_debug.setdefault("dropped_by_budget", []).append(user_node_id)
                continue

            # Add user message
            if user_content:
                recency_messages.append({
                    "role": "user",
                    "content": user_content
                })
                if user_node_id:
                    recency_node_ids.append(user_node_id)

            # Add assistant message
            if asst_content:
                recency_messages.append({
                    "role": "assistant",
                    "content": asst_content
                })
                if asst_node_id:
                    recency_node_ids.append(asst_node_id)

            tokens_used += exchange_tokens
            recency_debug["included_node_ids"].append(user_node_id)

        recency_debug["included_count"] = len(recency_debug["included_node_ids"])
        recency_debug["tokens_used"] = tokens_used

        return recency_messages, recency_node_ids, recency_debug

    def _emit_assembly_log(self, debug: Dict[str, Any]) -> None:
        """
        Emit structured log for assembly (Patch E).

        Single structured log record per assembly with:
        - strategy (topic-local vs ancestry)
        - topic_id
        - anchor exchange_ids (ordered)
        - recency exchange_ids (ordered)
        - per-component token counts
        - drop reasons
        """
        from episodic.config import config

        if not config.get("debug"):
            return

        log_record = {
            "event": "context_assembly",
            "strategy": debug.get("mode", "unknown"),
            "topic_id": debug.get("topic_start_node_id", "none"),
            "budgets": debug.get("budgets", {}),
            "anchors": {
                "exchange_ids": debug.get("anchors", {}).get("included_node_ids", []),
                "tokens": debug.get("anchors", {}).get("tokens_used", 0),
            },
            "recency": {
                "exchange_ids": debug.get("recency", {}).get("included_node_ids", []),
                "tokens": debug.get("recency", {}).get("tokens_used", 0),
                "skipped_as_anchors": debug.get("recency", {}).get("skipped_as_anchors", []),
            },
            "token_breakdown": debug.get("token_breakdown", {}),
            "truncations": debug.get("truncation_info", {}),
            "timing_ms": debug.get("timing", {}).get("context_assembly_ms", 0),
        }

        # Log as structured JSON
        logger.info(f"ASSEMBLY: {json.dumps(log_record)}")

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _fallback_to_ancestry(
        self,
        user_turn_text: str,
        user_node_id: Optional[str],
        active_topic_start_node_id: Optional[str],
        user_embedding: Optional[Any],
        token_budget: int,
        conn: Optional[sqlite3.Connection] = None,
        chroma_collection: Optional[Any] = None,
        original_debug: Optional[Dict] = None,
    ) -> ContextAssemblyResult:
        """
        Fall back to ancestry strategy when topic_local context is too thin.
        """
        from .ancestry import AncestryStrategy

        ancestry = AncestryStrategy()
        result = ancestry.assemble(
            user_turn_text=user_turn_text,
            user_node_id=user_node_id,
            active_topic_start_node_id=active_topic_start_node_id,
            user_embedding=user_embedding,
            token_budget=token_budget,
            conn=conn,
            chroma_collection=chroma_collection,
        )

        if original_debug:
            result.debug["fallback_reason"] = original_debug.get("fallback_reason")
            result.debug["thin_fallback_details"] = original_debug.get("thin_fallback_details")
            result.debug["original_mode"] = original_debug.get("mode")
        result.debug["mode"] = "ancestry_fallback"

        return result

    def _check_import_intent(
        self,
        user_turn_text: str,
        user_embedding: Optional[np.ndarray],
        active_topic_start_node_id: str,
        token_budget: int,
        conn: Optional[sqlite3.Connection] = None,
        chroma_collection: Optional[Any] = None,
    ) -> tuple:
        """
        Check for cross-topic import intent and fetch context if needed.
        """
        from .imports import (
            detect_import_intent,
            resolve_import_target,
            fetch_import_context,
        )
        from episodic.db_connection import get_connection

        import_debug = {
            "detected_intent": False,
            "topic_reference": None,
            "resolved_topic": None,
            "confidence": 0.0,
            "context_tokens": 0,
            "anchors_included": False,
        }

        intent = detect_import_intent(user_turn_text)
        import_debug["detected_intent"] = intent.has_intent
        import_debug["topic_reference"] = intent.topic_reference

        if not intent.has_intent or not intent.topic_reference:
            return "", import_debug

        def _resolve_and_fetch(c: sqlite3.Connection) -> tuple:
            target = resolve_import_target(
                topic_reference=intent.topic_reference,
                active_topic_start_node_id=active_topic_start_node_id,
                user_embedding=user_embedding,
                conn=c
            )

            if not target:
                return "", import_debug

            import_debug["resolved_topic"] = target.topic_name
            import_debug["confidence"] = target.confidence
            import_debug["match_method"] = target.match_method

            if target.topic_start_node_id == active_topic_start_node_id:
                import_debug["skipped_reason"] = "same_as_active"
                return "", import_debug

            import_result = fetch_import_context(
                source_topic_start_node_id=target.topic_start_node_id,
                user_input=user_turn_text,
                user_embedding=user_embedding,
                token_budget=token_budget,
                conn=c,
                chroma_collection=chroma_collection
            )

            import_debug["context_tokens"] = import_result.debug.get("estimated_tokens", 0)
            import_debug["anchors_included"] = import_result.debug.get("anchors_included", False)
            import_debug["summary_included"] = import_result.debug.get("summary_included", False)

            return import_result.context_block, import_debug

        if conn is not None:
            return _resolve_and_fetch(conn)

        with get_connection() as c:
            return _resolve_and_fetch(c)
