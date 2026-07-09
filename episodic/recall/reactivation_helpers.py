"""Low-level helpers and tuning constants for topic reactivation.

Leaf module split out of reactivation.py. Both the probe and the packet
assembler depend on these, and they depend on each other, so they live in one
leaf that neither high-level function imports back (no cycle). reactivation.py
re-exports the names so external imports are unchanged.
"""

import logging
import math
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from episodic.config import config
from episodic.db_connection import get_connection
from episodic.recall.resume_cues import has_resume_cues
from episodic.recall.topic_aliases import compute_alias_score, get_topic_aliases_batch

logger = logging.getLogger(__name__)

# Default parameters
K_TOPICS = 7           # Number of topics to consider in ANN
M_EXCHANGES = 12       # Exchanges to check for support
S_SUPPORT = 2          # Minimum support count (for channel A)
S_SUPPORT_ALIAS = 1    # Minimum support for channel B (alias matching) - lower for short topics
DELTA_BAND = 0.15      # Similarity band for support check (widened for sparse data)
COOLDOWN_TURNS = 3     # Turns to wait after reactivation
DORMANCY_MIN = 2       # Minimum turns inactive before reactivation (lowered for quick switches)

# Two-channel gate thresholds
SIM_THRESHOLD_NORMAL = 0.30      # Similarity threshold for non-resume-cued turns
SIM_THRESHOLD_RESUME_CUE = 0.25  # Lower threshold when resume cues detected
ALIAS_HITS_MIN = 2               # Minimum alias hits for channel B to pass


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


def _get_dormant_topic_centroids(
    conn: sqlite3.Connection,
    current_turn_idx: int,
    active_topic_start_node_id: Optional[str],
    dormancy_min: int,
    limit: int,
) -> List[Dict[str, Any]]:
    """Get up to `limit` most-recently-active DORMANT topics with centroids.

    The dormancy filter and the recency bound are pushed into SQL (indexed by
    idx_topic_centroids_turn) so this stays O(limit) per turn instead of
    scanning every topic centroid as the topic count grows.
    """
    cutoff = current_turn_idx - dormancy_min
    cursor = conn.execute("""
        SELECT t.name, t.start_node_id, t.end_node_id,
               tc.centroid_medoid_exchange_id, tc.exchange_count, tc.last_active_turn_idx
        FROM topics t
        JOIN topic_centroids tc ON t.start_node_id = tc.start_node_id
        WHERE tc.centroid_medoid_exchange_id IS NOT NULL
          AND tc.last_active_turn_idx <= ?
          AND (? IS NULL OR t.start_node_id != ?)
        ORDER BY tc.last_active_turn_idx DESC
        LIMIT ?
    """, (cutoff, active_topic_start_node_id, active_topic_start_node_id, limit))

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


