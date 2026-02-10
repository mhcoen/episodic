"""
Anchor retrieval and recency assembly for topic-local context recovery.

Extracted from topic_local.py to keep file sizes under the 500-line target.

Functions:
- retrieve_anchors_budgeted(): Semantic anchor retrieval within budget A
- build_recency_budgeted(): Recency tail from non-anchor exchanges within budget R
- cosine_similarity(): Vector similarity helper
- fallback_to_ancestry(): Fall back when topic_local context is too thin
- emit_assembly_log(): Structured debug logging for context assembly
- check_import_intent(): Cross-topic import detection and context fetching
"""

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def retrieve_anchors_budgeted(
    user_turn_text: str,
    user_embedding: Optional[Any],
    active_topic_start_node_id: str,
    summary_text: str,
    chroma_collection: Optional[Any],
    anchor_budget: int,
    anchor_count: int,
    anchor_similarity_threshold: float,
) -> Tuple[Set[str], str, Dict[str, Any]]:
    """
    Retrieve semantic anchors within budget A.

    Args:
        user_turn_text: The user's current input text
        user_embedding: Pre-computed embedding of user text (optional)
        active_topic_start_node_id: The active topic's start node ID
        summary_text: Topic summary text for novelty filtering
        chroma_collection: ChromaDB collection for similarity search
        anchor_budget: Token budget for anchors (A)
        anchor_count: Maximum number of anchors to include
        anchor_similarity_threshold: Minimum similarity threshold

    Returns:
        Tuple of (anchor_node_ids_set, anchor_context_str, anchor_debug_dict)
    """
    from .topic_local import _estimate_tokens

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
        threshold = anchor_similarity_threshold

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
                    cosine_sim = cosine_similarity(embedding, seen)
                    if cosine_sim > 0.95:
                        is_duplicate = True
                        anchor_debug["deduped_similarity"] += 1
                        break
                if is_duplicate:
                    continue

                # Novelty check against summary
                if summary_embedding is not None:
                    summary_sim = cosine_similarity(embedding, summary_embedding)
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
        filtered_anchors = filtered_anchors[:anchor_count]

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


def build_recency_budgeted(
    all_exchanges: List[Dict],
    anchor_node_ids: Set[str],
    recency_budget: int,
) -> Tuple[List[Dict], List[str], Dict[str, Any]]:
    """
    Build recency tail from non-anchor exchanges within budget R.

    Patch D: Exchanges that appear in anchors are skipped (charged to A).
    Patch B: Token-based truncation, not just count-based.

    Args:
        all_exchanges: List of exchange dicts from the topic
        anchor_node_ids: Set of node IDs already included as anchors
        recency_budget: Token budget for recency (R)

    Returns:
        Tuple of (messages_list, node_ids_list, recency_debug_dict)
    """
    from .topic_local import _estimate_tokens

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


def emit_assembly_log(debug: Dict[str, Any]) -> None:
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


def fallback_to_ancestry(
    user_turn_text: str,
    user_node_id: Optional[str],
    active_topic_start_node_id: Optional[str],
    user_embedding: Optional[Any],
    token_budget: int,
    conn: Any = None,
    chroma_collection: Optional[Any] = None,
    original_debug: Optional[Dict] = None,
):
    """
    Fall back to ancestry strategy when topic_local context is too thin.

    Returns:
        ContextAssemblyResult with ancestry-based context
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


def check_import_intent(
    user_turn_text: str,
    user_embedding: Optional[np.ndarray],
    active_topic_start_node_id: str,
    token_budget: int,
    conn: Optional[sqlite3.Connection] = None,
    chroma_collection: Optional[Any] = None,
) -> tuple:
    """
    Check for cross-topic import intent and fetch context if needed.

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
