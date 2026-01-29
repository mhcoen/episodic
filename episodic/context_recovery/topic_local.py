"""
Topic-local context recovery strategy.

Assembles context only from the active topic, excluding all other topics.
This enables "year-later" resume without contamination from intervening topics.

Supports explicit cross-topic imports when user references another topic.
"""

import sqlite3
import time
from typing import List, Dict, Any, Optional
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
        else:
            logger.warning(msg)

# Default number of exchange pairs to include
DEFAULT_EXCHANGE_PAIRS = 4  # 4 pairs = 8 messages


class TopicLocalStrategy:
    """
    Topic-local context recovery.

    Only includes messages from the active topic, completely excluding
    messages from other topics. Uses topic_nodes and topic_working_set
    tables for efficient retrieval.

    Also retrieves semantic anchors from Chroma filtered by topic_start_node_id.
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
            exchange_pairs: Number of exchange pairs (user+assistant) to include
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
        Assemble context from only the active topic.

        Order of assembly:
        1. System prompt (added later by caller)
        2. Global scratchpad (if any)
        3. Topic context block:
           - Topic name
           - Summary (if available)
           - Anchors (semantic retrieval within topic)
           - Last N exchanges from topic
        4. Current user message (added later by caller)

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
            "context_assembly_ms": 0.0,  # Total assembly time (excluding embedding)
        }

        messages = []
        included_node_ids = []
        debug = {
            "mode": ContextRecoveryMode.TOPIC_LOCAL.value,
            "topic_start_node_id": active_topic_start_node_id,
            "included_node_ids": [],
            "token_counts": {},
            "token_breakdown": {
                "summary_tokens": 0,
                "recency_tokens": 0,
                "anchor_tokens": 0,
                "scratchpad_tokens": 0,
                "import_tokens": 0,
                "total_tokens": 0,
            },
            "timing": timing,
            "truncation_info": None,
            "reactivation_fired": True,  # Topic-local implies reactivation
            "working_set_used": False,
            "summary_included": False,
            "anchors": {
                "retrieved_count": 0,
                "deduped_count": 0,
                "included_count": 0,
                "included_node_ids": [],
                "similarities": [],
                "tokens_used": 0,
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
            # No active topic, return empty context
            debug["reactivation_fired"] = False
            debug["timing"]["context_assembly_ms"] = (time.perf_counter() - assembly_start) * 1000
            return ContextAssemblyResult(messages=messages, debug=debug)

        # Time SQLite operations: build topic context
        sqlite_start = time.perf_counter()
        if conn is not None:
            topic_context, topic_messages, node_ids = self._build_topic_context(
                active_topic_start_node_id, conn, force_no_recency=force_no_recency
            )
        else:
            with get_connection() as c:
                topic_context, topic_messages, node_ids = self._build_topic_context(
                    active_topic_start_node_id, c, force_no_recency=force_no_recency
                )

        included_node_ids = list(node_ids)  # Copy to avoid mutation issues

        # Get summary text for novelty filtering
        summary_text = None
        if conn is not None:
            ws = get_topic_working_set(active_topic_start_node_id, conn=conn)
            summary_text = ws.get("summary_md") if ws else None
        else:
            with get_connection() as c:
                ws = get_topic_working_set(active_topic_start_node_id, conn=c)
                summary_text = ws.get("summary_md") if ws else None
        timing["sqlite_ops_ms"] = (time.perf_counter() - sqlite_start) * 1000

        # Retrieve semantic anchors within the topic (includes Chroma timing)
        recency_node_ids = set(node_ids)
        chroma_start = time.perf_counter()
        anchor_context, anchor_debug = self._retrieve_anchors(
            user_turn_text=user_turn_text,
            user_embedding=user_embedding,
            active_topic_start_node_id=active_topic_start_node_id,
            recency_node_ids=recency_node_ids,
            summary_text=summary_text,
            chroma_collection=chroma_collection,
            token_budget=token_budget,
        )
        timing["chroma_query_ms"] = (time.perf_counter() - chroma_start) * 1000

        # Add anchor node IDs to included list
        for anchor_id in anchor_debug.get("included_node_ids", []):
            if anchor_id not in included_node_ids:
                included_node_ids.append(anchor_id)

        debug["included_node_ids"] = included_node_ids
        debug["anchors"] = anchor_debug

        # Check for cross-topic import intent
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

        # Build messages in order:
        # 1. Topic context block (as system message)
        if topic_context or anchor_context or import_context:
            combined_context = topic_context
            if anchor_context:
                if combined_context:
                    combined_context += f"\n\n## Relevant Past Context\n{anchor_context}"
                else:
                    combined_context = f"## Relevant Past Context\n{anchor_context}"

            # Add imported context from other topic (if any)
            if import_context:
                if combined_context:
                    combined_context += f"\n\n{import_context}"
                else:
                    combined_context = import_context

            messages.append({
                "role": "system",
                "content": combined_context
            })
            debug["working_set_used"] = True
            if "## Summary" in combined_context:
                debug["summary_included"] = True

        # 2. Recent exchanges from topic
        for msg in topic_messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Estimate token counts - compute breakdown by section
        # Extract summary tokens from topic_context (if summary section exists)
        summary_chars = 0
        if topic_context and "## Summary" in topic_context:
            # Find the summary section
            summary_start = topic_context.find("## Summary")
            summary_end = topic_context.find("\n##", summary_start + 10)  # Next section
            if summary_end == -1:
                summary_end = len(topic_context)
            summary_chars = summary_end - summary_start

        topic_context_chars = len(topic_context) if topic_context else 0
        non_summary_topic_chars = topic_context_chars - summary_chars
        anchor_context_chars = len(anchor_context) if anchor_context else 0
        import_context_chars = len(import_context) if import_context else 0
        conversation_chars = sum(len(msg["content"]) for msg in topic_messages)

        # Legacy token_counts (for backwards compatibility)
        debug["token_counts"] = {
            "topic_context": topic_context_chars // 4,
            "anchor_context": anchor_context_chars // 4,
            "import_context": import_context_chars // 4,
            "conversation_history": conversation_chars // 4,
            "total_estimate": (topic_context_chars + anchor_context_chars + import_context_chars + conversation_chars) // 4,
        }
        debug["anchors"]["tokens_used"] = anchor_context_chars // 4

        # New token_breakdown (structured by section type)
        debug["token_breakdown"] = {
            "summary_tokens": summary_chars // 4,
            "recency_tokens": conversation_chars // 4,
            "anchor_tokens": anchor_context_chars // 4,
            "scratchpad_tokens": non_summary_topic_chars // 4,  # Topic name + metadata
            "import_tokens": import_context_chars // 4,
            "total_tokens": (topic_context_chars + anchor_context_chars + import_context_chars + conversation_chars) // 4,
        }

        # Finalize timing
        timing["context_assembly_ms"] = (time.perf_counter() - assembly_start) * 1000

        # Assert no contamination from foreign topics (debug mode only)
        if conn is not None:
            _assert_no_contamination(included_node_ids, active_topic_start_node_id, conn)
        else:
            with get_connection() as c:
                _assert_no_contamination(included_node_ids, active_topic_start_node_id, c)

        # Check for thin topic_local - fall back to ancestry if context is insufficient
        has_summary = debug.get("summary_included", False)
        anchor_count_retrieved = debug["anchors"].get("included_count", 0)
        total_tokens = debug["token_counts"].get("total_estimate", 0)

        min_anchors = config.get("min_anchors_for_topic_local", 2)
        min_tokens = config.get("min_tokens_for_topic_local", 500)

        if (not has_summary and
            anchor_count_retrieved < min_anchors and
            total_tokens < min_tokens):
            debug["fallback_reason"] = "thin_topic_local"
            debug["thin_fallback_details"] = {
                "has_summary": has_summary,
                "anchor_count": anchor_count_retrieved,
                "total_tokens": total_tokens,
                "min_anchors_threshold": min_anchors,
                "min_tokens_threshold": min_tokens,
            }
            logger.debug(
                f"Thin topic_local detected (no summary, {anchor_count_retrieved} anchors, "
                f"{total_tokens} tokens), falling back to ancestry"
            )
            return self._fallback_to_ancestry(
                user_turn_text=user_turn_text,
                user_node_id=user_node_id,
                active_topic_start_node_id=active_topic_start_node_id,
                user_embedding=user_embedding,
                token_budget=token_budget,
                conn=conn,
                chroma_collection=chroma_collection,
                original_debug=debug,
            )

        return ContextAssemblyResult(messages=messages, debug=debug)

    def _build_topic_context(
        self,
        topic_start_node_id: str,
        conn: sqlite3.Connection,
        force_no_recency: bool = False,
    ) -> tuple:
        """
        Build the topic context block.

        Args:
            topic_start_node_id: The topic's start node ID
            conn: Database connection
            force_no_recency: If True, skip loading recent exchanges (for year-later testing)

        Returns:
            Tuple of (topic_context_str, messages_list, node_ids_list)
        """
        from episodic.db_topic_nodes import (
            get_last_n_exchanges_in_topic,
            get_topic_working_set,
        )

        context_parts = []
        topic_messages = []
        node_ids = []

        # Get topic working set (summary, etc.)
        working_set = get_topic_working_set(topic_start_node_id, conn=conn)

        if working_set:
            topic_name = working_set.get("topic_name", "Unknown Topic")
            context_parts.append(f"# Topic: {topic_name}")

            summary = working_set.get("summary_md", "")
            if summary and summary.strip():
                context_parts.append(f"\n## Summary\n{summary.strip()}")

            # Future: add decisions, open_loops, entities from working_set

        # Skip recency loading for year-later testing scenarios
        if force_no_recency:
            topic_context = "\n".join(context_parts) if context_parts else ""
            return topic_context, topic_messages, node_ids

        # Get last N exchanges from topic
        exchanges = get_last_n_exchanges_in_topic(
            topic_start_node_id,
            n=self.exchange_pairs,
            conn=conn
        )

        if exchanges:
            for ex in exchanges:
                # User message
                user_content = ex.get("user_content", "")
                user_node_id = ex.get("user_node_id")
                if user_content:
                    topic_messages.append({
                        "role": "user",
                        "content": user_content
                    })
                    if user_node_id:
                        node_ids.append(user_node_id)

                # Assistant message
                asst_content = ex.get("assistant_content", "")
                asst_node_id = ex.get("assistant_node_id")
                if asst_content:
                    topic_messages.append({
                        "role": "assistant",
                        "content": asst_content
                    })
                    if asst_node_id:
                        node_ids.append(asst_node_id)

        # Build final context string
        topic_context = "\n".join(context_parts) if context_parts else ""

        return topic_context, topic_messages, node_ids

    def _retrieve_anchors(
        self,
        user_turn_text: str,
        user_embedding: Optional[Any],
        active_topic_start_node_id: str,
        recency_node_ids: set,
        summary_text: Optional[str] = None,
        chroma_collection: Optional[Any] = None,
        token_budget: int = 4000,
    ) -> tuple:
        """
        Retrieve semantic anchors from Chroma within the active topic.

        Rules (in order):
        1. Filter by topic_start_node_id in Chroma metadata
        2. Filter by similarity threshold
        3. Deduplicate against recency slice (don't repeat)
        4. Deduplicate near-duplicates (embedding cosine > 0.95)
        5. Novelty check against summary (reject if > 0.97 similarity)
        6. Enforce max anchor budget

        Args:
            user_turn_text: Current user input for query
            user_embedding: Pre-computed embedding of user input (optional)
            active_topic_start_node_id: Topic to filter by
            recency_node_ids: Node IDs already in recency window (for dedup)
            summary_text: Summary text for novelty filtering (optional)
            chroma_collection: Optional Chroma collection (uses default if None)
            token_budget: Max tokens for anchor context

        Returns:
            Tuple of (anchor_context_str, anchor_debug_dict)
        """
        anchor_debug = {
            "retrieved_count": 0,
            "deduped_recency": 0,
            "deduped_similarity": 0,
            "deduped_summary_redundant": 0,
            "included_count": 0,
            "included_node_ids": [],
            "similarities": [],
            "dropped_by_truncation": [],
            "tokens_used": 0,
        }

        # Get Chroma collection if not provided
        if chroma_collection is None:
            try:
                from episodic.rag_collections import get_multi_collection_rag, CollectionType
                rag = get_multi_collection_rag()
                chroma_collection = rag.get_collection(CollectionType.CONVERSATION)
            except Exception as e:
                logger.debug(f"Could not get Chroma collection for anchors: {e}")
                return "", anchor_debug

        if chroma_collection is None:
            return "", anchor_debug

        try:
            from episodic.config import config

            # Query with topic filter and include embeddings for dedup
            results = chroma_collection.query(
                query_texts=[user_turn_text],
                n_results=config.get("anchor_retrieval_count", 10),
                where={"topic_start_node_id": active_topic_start_node_id},
                include=["documents", "embeddings", "distances", "metadatas"],
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                return "", anchor_debug

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

                # Convert distance to similarity (Chroma uses L2 distance)
                similarity = max(0, 1 - (distance / 2.0))

                user_node_id = metadata.get("user_id", doc_id)
                asst_node_id = metadata.get("assistant_id")

                # Rule 2: Skip if below similarity threshold
                if similarity < threshold:
                    continue

                # Rule 3: Skip if in recency slice
                if user_node_id in recency_node_ids:
                    anchor_debug["deduped_recency"] += 1
                    continue
                if asst_node_id and asst_node_id in recency_node_ids:
                    anchor_debug["deduped_recency"] += 1
                    continue

                if embedding is not None:
                    # Rule 4: Skip near-duplicates of already-included anchors
                    is_duplicate = False
                    for seen in included_embeddings:
                        cosine_sim = self._cosine_similarity(embedding, seen)
                        if cosine_sim > 0.95:
                            is_duplicate = True
                            anchor_debug["deduped_similarity"] += 1
                            break
                    if is_duplicate:
                        continue

                    # Rule 5: Novelty check against summary
                    if summary_embedding is not None:
                        summary_sim = self._cosine_similarity(embedding, summary_embedding)
                        if summary_sim > 0.97:
                            anchor_debug["deduped_summary_redundant"] += 1
                            continue

                    included_embeddings.append(embedding)

                filtered_anchors.append(
                    {
                        "user_node_id": user_node_id,
                        "assistant_node_id": asst_node_id,
                        "document": document,
                        "similarity": similarity,
                        "user_content": metadata.get("user_content", ""),
                        "assistant_content": metadata.get("assistant_content", ""),
                    }
                )

            # Rule 6: Limit to anchor_count
            filtered_anchors = filtered_anchors[: self.anchor_count]

            # Build anchor context string with token budget check
            anchor_parts = []
            included_tokens = 0
            max_anchor_tokens = token_budget // 4  # Reserve ~25% for anchors

            for anchor in filtered_anchors:
                user_content = anchor.get("user_content", "")
                asst_content = anchor.get("assistant_content", "")

                anchor_text = ""
                if user_content:
                    anchor_text += f"User: {user_content}\n"
                if asst_content:
                    anchor_text += f"Assistant: {asst_content}"

                anchor_tokens = len(anchor_text) // 4

                if included_tokens + anchor_tokens > max_anchor_tokens:
                    anchor_debug["dropped_by_truncation"].append(anchor["user_node_id"])
                    continue

                anchor_parts.append(anchor_text.strip())
                included_tokens += anchor_tokens
                anchor_debug["included_node_ids"].append(anchor["user_node_id"])
                anchor_debug["similarities"].append(anchor["similarity"])

            anchor_debug["included_count"] = len(anchor_debug["included_node_ids"])
            anchor_debug["tokens_used"] = included_tokens

            anchor_context = "\n\n".join(anchor_parts) if anchor_parts else ""
            return anchor_context, anchor_debug

        except Exception as e:
            logger.warning(f"Anchor retrieval failed: {e}")
            return "", anchor_debug

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

        Args:
            user_turn_text: Current user message
            user_node_id: ID of the current user node
            active_topic_start_node_id: Start node ID of active topic
            user_embedding: Pre-computed embedding
            token_budget: Maximum tokens for context
            conn: Optional SQLite connection
            chroma_collection: Optional Chroma collection
            original_debug: Debug info from topic_local attempt (preserved)

        Returns:
            ContextAssemblyResult from ancestry strategy with merged debug info
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

        # Merge debug info - preserve fallback reason and thin details
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

        Args:
            user_turn_text: Current user input
            user_embedding: Pre-computed embedding
            active_topic_start_node_id: Currently active topic
            token_budget: Max tokens for imported context
            conn: Optional database connection
            chroma_collection: Optional Chroma collection

        Returns:
            Tuple of (import_context_str, import_debug_dict)
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

        # Detect import intent
        intent = detect_import_intent(user_turn_text)
        import_debug["detected_intent"] = intent.has_intent
        import_debug["topic_reference"] = intent.topic_reference

        if not intent.has_intent or not intent.topic_reference:
            return "", import_debug

        # Resolve the target topic
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

            # Don't import from active topic
            if target.topic_start_node_id == active_topic_start_node_id:
                import_debug["skipped_reason"] = "same_as_active"
                return "", import_debug

            # Fetch context from the resolved topic
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
