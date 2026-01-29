"""
Implicit topic reactivation probe.

Detects when a user message relates to a previously inactive topic
and returns a decision about whether to reactivate that topic.
"""

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from episodic.db_connection import get_connection

logger = logging.getLogger(__name__)

# Default parameters
K_TOPICS = 7           # Number of topics to consider in ANN
M_EXCHANGES = 12       # Exchanges to check for support
S_SUPPORT = 2          # Minimum support count
DELTA_BAND = 0.15      # Similarity band for support check (widened for sparse data)
COOLDOWN_TURNS = 3     # Turns to wait after reactivation
DORMANCY_MIN = 4       # Minimum turns inactive before reactivation


@dataclass
class DisambiguationOption:
    """Option for disambiguation when multiple topics match."""
    topic_name: str
    topic_start_node_id: str
    similarity: float
    support_count: int
    preview: str = ""  # Short preview of topic content
    turns_ago: int = 0  # How many turns since last activity
    snippets: List[str] = field(default_factory=list)  # Evidence snippets


@dataclass
class ReactivationDecision:
    """Result of reactivation probe."""
    action: Literal["CONTINUE", "REACTIVATE", "DISAMBIGUATE"]
    topic_name: Optional[str] = None
    topic_start_node_id: Optional[str] = None
    options: Optional[List[DisambiguationOption]] = None
    debug: Dict[str, Any] = field(default_factory=dict)


def _get_current_turn_idx(conn: sqlite3.Connection) -> int:
    """Get the current maximum turn index."""
    cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
    row = cursor.fetchone()
    return row[0] if row and row[0] else 0


def _get_topic_info(conn: sqlite3.Connection, start_node_id: str) -> Optional[Dict[str, Any]]:
    """Get topic information including name and centroid."""
    cursor = conn.execute("""
        SELECT t.name, t.start_node_id, t.end_node_id,
               tc.centroid_medoid_exchange_id, tc.exchange_count, tc.last_active_turn_idx
        FROM topics t
        LEFT JOIN topic_centroids tc ON t.start_node_id = tc.start_node_id
        WHERE t.start_node_id = ?
    """, (start_node_id,))
    row = cursor.fetchone()
    if row:
        return {
            'name': row[0],
            'start_node_id': row[1],
            'end_node_id': row[2],
            'centroid_medoid_exchange_id': row[3],
            'exchange_count': row[4] or 0,
            'last_active_turn_idx': row[5] or 0
        }
    return None


def _get_all_topic_centroids(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all topics with their centroid information."""
    cursor = conn.execute("""
        SELECT t.name, t.start_node_id, t.end_node_id,
               tc.centroid_medoid_exchange_id, tc.exchange_count, tc.last_active_turn_idx
        FROM topics t
        LEFT JOIN topic_centroids tc ON t.start_node_id = tc.start_node_id
        WHERE tc.centroid_medoid_exchange_id IS NOT NULL
        ORDER BY tc.last_active_turn_idx DESC
    """)

    topics = []
    for row in cursor.fetchall():
        topics.append({
            'name': row[0],
            'start_node_id': row[1],
            'end_node_id': row[2],
            'centroid_medoid_exchange_id': row[3],
            'exchange_count': row[4] or 0,
            'last_active_turn_idx': row[5] or 0
        })
    return topics


def _compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def _get_embeddings_for_nodes(node_ids: List[str]) -> Dict[str, np.ndarray]:
    """Fetch embeddings from Chroma for given node IDs."""
    if not node_ids:
        return {}

    try:
        from episodic.rag_collections import get_multi_collection_rag, CollectionType

        rag = get_multi_collection_rag()
        collection = rag.get_collection(CollectionType.CONVERSATION)

        result = collection.get(ids=node_ids, include=['embeddings'])

        embeddings = {}
        if result is None:
            return embeddings

        result_ids = result.get('ids')
        result_embeddings = result.get('embeddings')

        # Check for empty results (handle both list and numpy array)
        if result_ids is None or result_embeddings is None:
            return embeddings

        ids_len = len(result_ids) if hasattr(result_ids, '__len__') else 0
        embs_len = len(result_embeddings) if hasattr(result_embeddings, '__len__') else 0

        if ids_len == 0 or embs_len == 0:
            return embeddings

        for i in range(min(ids_len, embs_len)):
            node_id = result_ids[i]
            emb = result_embeddings[i]
            # emb could be numpy array, so check via 'is None'
            if emb is None:
                continue
            embeddings[node_id] = np.array(emb)

        return embeddings
    except Exception as e:
        logger.warning(f"Error fetching embeddings: {e}")
        return {}


def _get_topic_exchanges(
    conn: sqlite3.Connection,
    start_node_id: str,
    end_node_id: Optional[str],
    limit: int = M_EXCHANGES
) -> List[str]:
    """Get user node IDs for exchanges in a topic."""
    # Get turn indices
    cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (start_node_id,))
    row = cursor.fetchone()
    if not row:
        return []
    start_idx = row[0]

    if end_node_id:
        cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (end_node_id,))
        row = cursor.fetchone()
        end_idx = row[0] if row else None
    else:
        cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
        end_idx = cursor.fetchone()[0]

    if end_idx is None:
        return []

    # Get user nodes in range (most recent first for sampling)
    cursor = conn.execute("""
        SELECT id FROM nodes
        WHERE rowid >= ? AND rowid <= ?
        AND role = 'user'
        AND COALESCE(is_meta_query, 0) = 0
        ORDER BY rowid DESC
        LIMIT ?
    """, (start_idx, end_idx, limit))

    return [row[0] for row in cursor.fetchall()]


def _check_support(
    user_embedding: np.ndarray,
    topic_exchanges: List[str],
    embeddings: Dict[str, np.ndarray],
    delta: float = DELTA_BAND
) -> Tuple[int, float]:
    """
    Check support for a topic by counting exchanges within similarity band.

    Returns: (support_count, best_similarity)
    """
    if not topic_exchanges:
        return 0, 0.0

    similarities = []
    for node_id in topic_exchanges:
        if node_id in embeddings:
            sim = _compute_similarity(user_embedding, embeddings[node_id])
            similarities.append(sim)

    if not similarities:
        return 0, 0.0

    best_sim = max(similarities)
    # Count how many are within delta of the best
    support_count = sum(1 for s in similarities if s >= best_sim - delta)

    return support_count, best_sim


def _get_topic_preview(conn: sqlite3.Connection, start_node_id: str) -> str:
    """Get a short preview of the topic's content."""
    cursor = conn.execute("""
        SELECT content FROM nodes
        WHERE id = ?
    """, (start_node_id,))
    row = cursor.fetchone()
    if row and row[0]:
        content = row[0][:100]
        if len(row[0]) > 100:
            content += "..."
        return content
    return ""


def _get_topic_snippets(
    conn: sqlite3.Connection,
    start_node_id: str,
    max_snippets: int = 2
) -> List[str]:
    """Get representative snippets from a topic for disambiguation display."""
    # Try to get user messages from topic_nodes first
    cursor = conn.execute("""
        SELECT n.content
        FROM topic_nodes tn
        JOIN nodes n ON tn.node_id = n.id
        WHERE tn.topic_start_node_id = ?
          AND n.role = 'user'
        ORDER BY n.rowid DESC
        LIMIT ?
    """, (start_node_id, max_snippets * 2))  # Fetch more to filter
    rows = cursor.fetchall()

    if not rows:
        # Fallback: get any nodes from the topic range
        cursor = conn.execute("""
            SELECT n.content
            FROM nodes n
            WHERE n.rowid >= (SELECT rowid FROM nodes WHERE id = ?)
              AND n.role = 'user'
            ORDER BY n.rowid DESC
            LIMIT ?
        """, (start_node_id, max_snippets * 2))
        rows = cursor.fetchall()

    snippets = []
    for row in rows:
        if row[0]:
            content = row[0].strip()
            # Take first meaningful sentence/question
            if content:
                # Truncate long content
                if len(content) > 60:
                    content = content[:57] + "..."
                snippets.append(content)
                if len(snippets) >= max_snippets:
                    break

    return snippets


def _get_topic_summary(conn: sqlite3.Connection, start_node_id: str) -> Optional[str]:
    """Get the compression summary for a topic if it exists."""
    cursor = conn.execute("""
        SELECT c.compressed_content
        FROM compressions_v2 c
        JOIN compression_nodes cn ON c.compressed_node_id = cn.compression_id
        WHERE cn.original_node_id IN (
            SELECT n.id FROM nodes n
            WHERE n.rowid >= (SELECT rowid FROM nodes WHERE id = ?)
            AND n.rowid <= COALESCE(
                (SELECT rowid FROM nodes WHERE id = (
                    SELECT end_node_id FROM topics WHERE start_node_id = ?
                )),
                (SELECT MAX(rowid) FROM nodes)
            )
        )
        LIMIT 1
    """, (start_node_id, start_node_id))
    row = cursor.fetchone()
    return row[0] if row else None


def _get_recent_exchanges(
    conn: sqlite3.Connection,
    start_node_id: str,
    end_node_id: Optional[str],
    limit: int = 2
) -> List[Tuple[str, str]]:
    """Get the most recent exchanges from a topic."""
    # Get turn indices
    cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (start_node_id,))
    row = cursor.fetchone()
    if not row:
        return []
    start_idx = row[0]

    if end_node_id:
        cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (end_node_id,))
        row = cursor.fetchone()
        end_idx = row[0] if row else None
    else:
        cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
        end_idx = cursor.fetchone()[0]

    if end_idx is None:
        return []

    # Get user nodes (most recent first)
    cursor = conn.execute("""
        SELECT id, rowid FROM nodes
        WHERE rowid >= ? AND rowid <= ?
        AND role = 'user'
        AND COALESCE(is_meta_query, 0) = 0
        ORDER BY rowid DESC
        LIMIT ?
    """, (start_idx, end_idx, limit))

    exchanges = []
    for user_id, user_rowid in cursor.fetchall():
        # Get assistant response
        asst_cursor = conn.execute("""
            SELECT id, content FROM nodes
            WHERE rowid > ? AND role = 'assistant'
            ORDER BY rowid
            LIMIT 1
        """, (user_rowid,))
        asst_row = asst_cursor.fetchone()

        # Get user content
        user_cursor = conn.execute("SELECT content FROM nodes WHERE id = ?", (user_id,))
        user_row = user_cursor.fetchone()

        if asst_row and user_row:
            exchanges.append((user_row[0], asst_row[1]))

    # Return in chronological order
    return list(reversed(exchanges))


def _get_anchor_exchange(
    conn: sqlite3.Connection,
    start_node_id: str,
    end_node_id: Optional[str],
    user_embedding: np.ndarray
) -> Optional[Tuple[str, str]]:
    """Get the most relevant exchange based on embedding similarity."""
    exchanges = _get_topic_exchanges(conn, start_node_id, end_node_id, limit=M_EXCHANGES)
    if not exchanges:
        return None

    embeddings = _get_embeddings_for_nodes(exchanges)
    if not embeddings:
        return None

    # Find best matching exchange
    best_id = None
    best_sim = -1
    for node_id in exchanges:
        if node_id in embeddings:
            sim = _compute_similarity(user_embedding, embeddings[node_id])
            if sim > best_sim:
                best_sim = sim
                best_id = node_id

    if not best_id:
        return None

    # Get the exchange content
    cursor = conn.execute("""
        SELECT content FROM nodes WHERE id = ?
    """, (best_id,))
    user_row = cursor.fetchone()
    if not user_row:
        return None

    # Get assistant response
    cursor = conn.execute("""
        SELECT content FROM nodes
        WHERE rowid > (SELECT rowid FROM nodes WHERE id = ?)
        AND role = 'assistant'
        ORDER BY rowid
        LIMIT 1
    """, (best_id,))
    asst_row = cursor.fetchone()

    if asst_row:
        return (user_row[0], asst_row[0])
    return None


def _truncate_to_budget(text: str, token_budget: int) -> str:
    """Truncate text to approximately fit token budget (rough estimate: 4 chars per token)."""
    char_budget = token_budget * 4
    if len(text) <= char_budget:
        return text
    return text[:char_budget - 3] + "..."


def assemble_reactivation_packet(
    topic_start_node_id: str,
    user_embedding: np.ndarray,
    token_budget: int = 150,
    conn: Optional[sqlite3.Connection] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Assemble a context packet for topic reactivation.

    Rules:
    - If topic compressed: summary + 1 anchor exchange
    - If not compressed: last 2 exchanges from topic

    Args:
        topic_start_node_id: Start node ID of the topic to reactivate
        user_embedding: Embedding of current user input (for anchor selection)
        token_budget: Maximum tokens for the packet
        conn: Optional database connection

    Returns:
        Tuple of (packet_text, debug_info)
    """
    debug_info: Dict[str, Any] = {
        'topic_start_node_id': topic_start_node_id,
        'token_budget': token_budget,
    }

    def _assemble(c: sqlite3.Connection) -> Tuple[str, Dict[str, Any]]:
        # Get topic info
        topic_info = _get_topic_info(c, topic_start_node_id)
        if not topic_info:
            debug_info['error'] = 'topic_not_found'
            return "", debug_info

        topic_name = topic_info['name']
        end_node_id = topic_info.get('end_node_id')
        debug_info['topic_name'] = topic_name

        # Check if topic has a compression summary
        summary = _get_topic_summary(c, topic_start_node_id)

        if summary:
            debug_info['has_summary'] = True
            # Use summary + 1 anchor exchange
            anchor = _get_anchor_exchange(c, topic_start_node_id, end_node_id, user_embedding)

            parts = [f"[Previous context from '{topic_name}':]"]
            parts.append(summary)

            if anchor:
                user_content, asst_content = anchor
                parts.append("")
                parts.append(f"User: {user_content}")
                parts.append(f"Assistant: {asst_content}")
                debug_info['anchor_included'] = True

            packet = "\n".join(parts)

        else:
            debug_info['has_summary'] = False
            # Use last 2 exchanges
            exchanges = _get_recent_exchanges(c, topic_start_node_id, end_node_id, limit=2)
            debug_info['exchanges_count'] = len(exchanges)

            if not exchanges:
                debug_info['error'] = 'no_exchanges_found'
                return "", debug_info

            parts = [f"[Previous context from '{topic_name}':]"]
            for user_content, asst_content in exchanges:
                parts.append("")
                parts.append(f"User: {user_content}")
                parts.append(f"Assistant: {asst_content}")

            packet = "\n".join(parts)

        # Truncate to budget
        packet = _truncate_to_budget(packet, token_budget)
        debug_info['packet_length'] = len(packet)

        return packet, debug_info

    if conn is not None:
        return _assemble(conn)

    with get_connection() as c:
        return _assemble(c)


def probe_reactivation(
    user_input: str,
    user_embedding: np.ndarray,
    active_topic_start_node_id: Optional[str],
    cooldown_turns: int,
    now: datetime,
    recent_nodes: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None
) -> ReactivationDecision:
    """
    Probe whether to reactivate a dormant topic based on user input.

    Args:
        user_input: The user's message text
        user_embedding: Pre-computed embedding for user_input
        active_topic_start_node_id: Start node ID of currently active topic (if any)
        cooldown_turns: Number of turns remaining in cooldown period
        now: Current timestamp
        recent_nodes: Recent conversation nodes for context
        conn: Optional database connection

    Returns:
        ReactivationDecision with action and optional topic info
    """
    debug_info: Dict[str, Any] = {
        'cooldown_turns': cooldown_turns,
        'active_topic': active_topic_start_node_id,
        # Feature logging for calibration
        'candidates': [],
        'best_vs_active_gap': None,
        'support_counts': {},
        'gates_passed': [],
        'gates_failed': [],
        'confidence': 0.0,
    }

    # Early exit: cooldown active
    if cooldown_turns > 0:
        debug_info['exit_reason'] = 'cooldown_active'
        debug_info['gates_failed'].append('cooldown')
        return ReactivationDecision(action="CONTINUE", debug=debug_info)

    # Early exit: input too short
    if len(user_input.split()) < 4:
        debug_info['exit_reason'] = 'input_too_short'
        debug_info['gates_failed'].append('input_length')
        return ReactivationDecision(action="CONTINUE", debug=debug_info)

    def _probe(c: sqlite3.Connection) -> ReactivationDecision:
        current_turn_idx = _get_current_turn_idx(c)
        debug_info['current_turn_idx'] = current_turn_idx

        # Get all topics with centroids
        all_topics = _get_all_topic_centroids(c)
        if not all_topics:
            debug_info['exit_reason'] = 'no_topics_with_centroids'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        debug_info['total_topics'] = len(all_topics)

        # Filter for dormant topics (inactive for >= DORMANCY_MIN turns)
        dormant_topics = []
        for topic in all_topics:
            dormancy = current_turn_idx - topic['last_active_turn_idx']
            # Skip active topic
            if topic['start_node_id'] == active_topic_start_node_id:
                continue
            # Skip recently active topics
            if dormancy < DORMANCY_MIN:
                continue
            topic['dormancy'] = dormancy
            dormant_topics.append(topic)

        if not dormant_topics:
            debug_info['exit_reason'] = 'no_dormant_topics'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        debug_info['dormant_topics'] = len(dormant_topics)

        # Get centroid embeddings for dormant topics
        centroid_node_ids = [t['centroid_medoid_exchange_id'] for t in dormant_topics]
        centroid_embeddings = _get_embeddings_for_nodes(centroid_node_ids)

        if not centroid_embeddings:
            debug_info['exit_reason'] = 'no_centroid_embeddings'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        # Compute similarity to each topic centroid
        topic_similarities = []
        for topic in dormant_topics:
            centroid_id = topic['centroid_medoid_exchange_id']
            if centroid_id in centroid_embeddings:
                sim = _compute_similarity(user_embedding, centroid_embeddings[centroid_id])
                topic_similarities.append((topic, sim))

        if not topic_similarities:
            debug_info['exit_reason'] = 'no_similarities_computed'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        # Sort by similarity descending
        topic_similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top K topics
        top_k = topic_similarities[:K_TOPICS]
        debug_info['top_k_similarities'] = [(t['name'], s) for t, s in top_k]

        # Build detailed candidates list for feature logging
        for rank, (topic, sim) in enumerate(top_k):
            debug_info['candidates'].append({
                'topic': topic['name'],
                'topic_start_node_id': topic['start_node_id'],
                'sim': sim,
                'rank': rank,
                'dormancy': topic.get('dormancy', 0),
            })

        # Percentile gate: check if best similarity is reasonable
        # Use P25 threshold (simple heuristic: similarity > 0.3)
        best_topic, best_sim = top_k[0]
        if best_sim < 0.3:  # P25 approximation
            debug_info['exit_reason'] = 'similarity_below_threshold'
            debug_info['best_similarity'] = best_sim
            debug_info['gates_failed'].append('similarity_threshold')
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        debug_info['best_similarity'] = best_sim
        debug_info['best_topic'] = best_topic['name']
        debug_info['gates_passed'].append('similarity_threshold')

        # Support check for best candidate (and close contenders for ambiguity)
        # First, identify candidates within rank_gap of best
        rank_gap = max(2, min(5, math.ceil(0.1 * K_TOPICS)))
        rank_gap_threshold = best_sim - 0.05  # Within 0.05 of best

        # Candidates to check: best + any within rank_gap
        candidates_to_check = [(best_topic, best_sim)]
        for topic, sim in top_k[1:]:
            if sim >= rank_gap_threshold:
                candidates_to_check.append((topic, sim))

        # Gather exchange embeddings for candidates
        exchange_ids_needed = []
        for topic, _ in candidates_to_check:
            exchanges = _get_topic_exchanges(
                c, topic['start_node_id'], topic.get('end_node_id'), M_EXCHANGES
            )
            topic['_exchange_ids'] = exchanges
            exchange_ids_needed.extend(exchanges)

        exchange_embeddings = _get_embeddings_for_nodes(list(set(exchange_ids_needed)))

        # Check support for best candidate first
        best_exchanges = best_topic.get('_exchange_ids', [])
        best_support_count, best_exchange_sim = _check_support(
            user_embedding, best_exchanges, exchange_embeddings, DELTA_BAND
        )

        debug_info['best_support_count'] = best_support_count
        debug_info['best_exchange_sim'] = best_exchange_sim
        debug_info['support_counts']['best'] = best_support_count

        # Best candidate must pass support threshold
        if best_support_count < S_SUPPORT:
            debug_info['exit_reason'] = 'best_candidate_insufficient_support'
            debug_info['gates_failed'].append('support')
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        debug_info['gates_passed'].append('support')

        # Check if there are close contenders with support (for ambiguity)
        close_with_support = []
        for topic, centroid_sim in candidates_to_check[1:]:
            exchanges = topic.get('_exchange_ids', [])
            support_count, exchange_sim = _check_support(
                user_embedding, exchanges, exchange_embeddings, DELTA_BAND
            )
            debug_info['support_counts'][topic['name']] = support_count
            if support_count >= S_SUPPORT:
                close_with_support.append({
                    'topic': topic,
                    'centroid_sim': centroid_sim,
                    'support_count': support_count,
                    'best_exchange_sim': exchange_sim,
                })

        debug_info['close_contenders_with_support'] = len(close_with_support)
        if len(close_with_support) > 0:
            debug_info['support_counts']['second'] = close_with_support[0]['support_count'] if close_with_support else 0

        # Check for ambiguity
        if close_with_support:
            # Multiple topics match - disambiguate
            debug_info['ambiguity_detected'] = True
            debug_info['rank_gap'] = rank_gap

            options = [
                DisambiguationOption(
                    topic_name=best_topic['name'],
                    topic_start_node_id=best_topic['start_node_id'],
                    similarity=best_sim,
                    support_count=best_support_count,
                    preview=_get_topic_preview(c, best_topic['start_node_id']),
                    turns_ago=best_topic.get('dormancy', 0),
                    snippets=_get_topic_snippets(c, best_topic['start_node_id']),
                )
            ]
            for cand in close_with_support:
                topic = cand['topic']
                options.append(DisambiguationOption(
                    topic_name=topic['name'],
                    topic_start_node_id=topic['start_node_id'],
                    similarity=cand['centroid_sim'],
                    support_count=cand['support_count'],
                    preview=_get_topic_preview(c, topic['start_node_id']),
                    turns_ago=topic.get('dormancy', 0),
                    snippets=_get_topic_snippets(c, topic['start_node_id']),
                ))

            return ReactivationDecision(
                action="DISAMBIGUATE",
                options=options,
                debug=debug_info
            )

        # Clear winner - reactivate best candidate
        debug_info['support_count'] = best_support_count
        debug_info['rank_gap_passes'] = True
        debug_info['dormancy_turns'] = best_topic.get('dormancy', 0)
        debug_info['gates_passed'].append('rank_gap')
        debug_info['gates_passed'].append('dormancy')

        # Compute confidence score (higher = more confident)
        # Factors: similarity, support count, dormancy, no close contenders
        confidence = min(1.0, (
            (best_sim - 0.3) / 0.4 * 0.4 +  # Similarity contribution (0.3-0.7 → 0-0.4)
            min(best_support_count / 4.0, 1.0) * 0.3 +  # Support contribution (up to 0.3)
            (1.0 if not close_with_support else 0.5) * 0.2 +  # Uniqueness contribution
            min(best_topic.get('dormancy', 0) / 10.0, 1.0) * 0.1  # Dormancy contribution
        ))
        debug_info['confidence'] = confidence

        return ReactivationDecision(
            action="REACTIVATE",
            topic_name=best_topic['name'],
            topic_start_node_id=best_topic['start_node_id'],
            debug=debug_info
        )

    # Execute with connection
    if conn is not None:
        return _probe(conn)

    with get_connection() as c:
        return _probe(c)
