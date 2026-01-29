"""
Centroid/medoid maintenance for topic reactivation.

This module maintains topic centroid information in the topic_centroids table.
Centroids are computed lazily at checkpoint intervals (1, 2, 4, 8, 16... exchanges).
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np

from episodic.db_connection import get_connection

logger = logging.getLogger(__name__)

# Checkpoint intervals for centroid updates (powers of 2)
CHECKPOINT_INTERVALS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Maximum exchanges to sample for medoid computation
MAX_SAMPLE_SIZE = 50


def is_checkpoint(exchange_count: int) -> bool:
    """Check if exchange_count is a checkpoint for centroid update."""
    return exchange_count in CHECKPOINT_INTERVALS


def get_turn_idx(conn: sqlite3.Connection, node_id: str) -> Optional[int]:
    """Get the rowid (turn index) for a node."""
    cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    return row[0] if row else None


def get_topic_exchanges(
    conn: sqlite3.Connection,
    start_node_id: str,
    end_node_id: Optional[str] = None,
    limit: int = MAX_SAMPLE_SIZE
) -> List[Tuple[str, str]]:
    """
    Get exchange node IDs for a topic.

    Returns list of (user_node_id, assistant_node_id) tuples.
    Only returns user nodes with their assistant response pairs.
    """
    # Get turn indices
    start_idx = get_turn_idx(conn, start_node_id)
    if start_idx is None:
        return []

    if end_node_id:
        end_idx = get_turn_idx(conn, end_node_id)
    else:
        # Open topic - get current max rowid
        cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
        end_idx = cursor.fetchone()[0]

    if end_idx is None:
        return []

    # Get user nodes in range (most recent first for sampling)
    cursor = conn.execute("""
        SELECT n.id, n.rowid
        FROM nodes n
        WHERE n.rowid >= ? AND n.rowid <= ?
        AND n.role = 'user'
        AND COALESCE(n.is_meta_query, 0) = 0
        ORDER BY n.rowid DESC
        LIMIT ?
    """, (start_idx, end_idx, limit))

    user_nodes = cursor.fetchall()

    # For each user node, find the assistant response
    exchanges = []
    for user_id, user_rowid in user_nodes:
        # Get the next assistant node after this user node
        cursor = conn.execute("""
            SELECT id FROM nodes
            WHERE rowid > ? AND role = 'assistant'
            ORDER BY rowid
            LIMIT 1
        """, (user_rowid,))
        asst_row = cursor.fetchone()
        if asst_row:
            exchanges.append((user_id, asst_row[0]))

    # Reverse to chronological order (oldest first)
    return list(reversed(exchanges))


def compute_medoid(
    conn: sqlite3.Connection,
    exchanges: List[Tuple[str, str]],
    embeddings_cache: Optional[dict] = None
) -> Optional[str]:
    """
    Compute the medoid (most central) exchange from a list of exchanges.

    The medoid is the exchange whose embedding has minimum average distance
    to all other exchange embeddings.

    Args:
        conn: Database connection
        exchanges: List of (user_node_id, assistant_node_id) tuples
        embeddings_cache: Optional dict of node_id -> embedding for efficiency

    Returns:
        The user_node_id of the medoid exchange, or None if can't compute
    """
    if not exchanges:
        return None

    if len(exchanges) == 1:
        return exchanges[0][0]  # Return the only user node

    try:
        # Get embeddings from Chroma
        from episodic.rag_collections import get_multi_collection_rag, CollectionType

        rag = get_multi_collection_rag()
        collection = rag.get_collection(CollectionType.CONVERSATION)

        # Get user node IDs
        user_node_ids = [ex[0] for ex in exchanges]

        # Fetch embeddings
        if embeddings_cache:
            embeddings = []
            valid_ids = []
            missing_ids = []
            for nid in user_node_ids:
                if nid in embeddings_cache:
                    embeddings.append(embeddings_cache[nid])
                    valid_ids.append(nid)
                else:
                    missing_ids.append(nid)

            if missing_ids:
                # Fetch missing from Chroma
                result = collection.get(ids=missing_ids, include=['embeddings'])
                result_embeddings = result.get('embeddings', []) if result else []
                result_ids = result.get('ids', []) if result else []
                if len(result_embeddings) > 0 and len(result_ids) > 0:
                    for i, nid in enumerate(result_ids):
                        emb = result_embeddings[i]
                        # Check for None explicitly using identity check
                        if emb is None or (hasattr(emb, '__len__') and len(emb) == 0):
                            continue
                        embeddings_cache[nid] = emb
                        embeddings.append(emb)
                        valid_ids.append(nid)
            user_node_ids = valid_ids
        else:
            # Fetch all from Chroma
            result = collection.get(ids=user_node_ids, include=['embeddings'])
            result_embeddings = result.get('embeddings', []) if result else []
            result_ids = result.get('ids', []) if result else []

            if len(result_embeddings) == 0:
                logger.warning("No embeddings found for topic exchanges")
                return exchanges[-1][0]  # Fallback: return most recent

            # Filter out None embeddings
            embeddings = []
            valid_ids = []
            for i in range(len(result_embeddings)):
                emb = result_embeddings[i]
                # Check for None explicitly using identity check
                if emb is None or (hasattr(emb, '__len__') and len(emb) == 0):
                    continue
                if i < len(result_ids) and result_ids[i]:
                    embeddings.append(emb)
                    valid_ids.append(result_ids[i])
            user_node_ids = valid_ids

        # Check if we have enough embeddings (handle both list and numpy array)
        emb_count = len(embeddings) if hasattr(embeddings, '__len__') else 0
        if emb_count < 2:
            return exchanges[-1][0]

        # Convert to numpy array
        emb_matrix = np.array(embeddings)

        # Compute pairwise distances using cosine distance
        # Normalize embeddings
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = emb_matrix / norms

        # Cosine similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)

        # Convert to distance (1 - similarity)
        dist_matrix = 1 - sim_matrix

        # Find medoid: node with minimum average distance to others
        avg_distances = np.mean(dist_matrix, axis=1)
        medoid_idx = np.argmin(avg_distances)

        # Return the user_node_id of the medoid
        return user_node_ids[medoid_idx]

    except Exception as e:
        logger.warning(f"Error computing medoid: {e}")
        # Fallback: return most recent exchange
        return exchanges[-1][0] if exchanges else None


def _update_topic_centroid_impl(
    conn: sqlite3.Connection,
    topic_start_node_id: str,
    topic_end_node_id: Optional[str] = None,
    force: bool = False
) -> bool:
    """Internal implementation of update_topic_centroid with an active connection."""
    # Get current exchange count
    exchanges = get_topic_exchanges(conn, topic_start_node_id, topic_end_node_id)
    exchange_count = len(exchanges)

    if exchange_count == 0:
        return False

    # Check if we should update
    if not force and not is_checkpoint(exchange_count):
        # Not at a checkpoint - update exchange count, turn_idx, and ensure medoid exists
        cursor = conn.execute(
            "SELECT exchange_count, centroid_medoid_exchange_id FROM topic_centroids WHERE start_node_id = ?",
            (topic_start_node_id,)
        )
        row = cursor.fetchone()
        existing_count = row[0] if row else 0
        existing_medoid = row[1] if row else None

        if exchange_count != existing_count or existing_medoid is None:
            # Get current turn index
            if topic_end_node_id:
                turn_idx = get_turn_idx(conn, topic_end_node_id)
            else:
                cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
                turn_idx = cursor.fetchone()[0]

            # If no medoid exists, use the most recent exchange as fallback
            # This ensures topics are discoverable by the reactivation probe
            medoid_node_id = existing_medoid
            if medoid_node_id is None and exchanges:
                medoid_node_id = exchanges[-1][0]  # Most recent user node

            # Update count, turn_idx, and medoid (if needed)
            conn.execute("""
                INSERT INTO topic_centroids
                    (start_node_id, centroid_medoid_exchange_id, exchange_count,
                     last_active_turn_idx, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(start_node_id) DO UPDATE SET
                    centroid_medoid_exchange_id = COALESCE(
                        topic_centroids.centroid_medoid_exchange_id,
                        excluded.centroid_medoid_exchange_id
                    ),
                    exchange_count = excluded.exchange_count,
                    last_active_turn_idx = excluded.last_active_turn_idx,
                    updated_at = excluded.updated_at
            """, (topic_start_node_id, medoid_node_id, exchange_count, turn_idx, datetime.utcnow()))
            conn.commit()

        return False

    # At a checkpoint - compute medoid
    logger.debug(f"Computing medoid for topic {topic_start_node_id[:8]}... "
                f"(checkpoint at {exchange_count} exchanges)")

    # Sample last W exchanges for medoid computation
    sample_exchanges = exchanges[-MAX_SAMPLE_SIZE:]
    medoid_node_id = compute_medoid(conn, sample_exchanges)

    # Get current turn index
    if topic_end_node_id:
        turn_idx = get_turn_idx(conn, topic_end_node_id)
    else:
        cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
        turn_idx = cursor.fetchone()[0]

    # Upsert centroid record
    conn.execute("""
        INSERT INTO topic_centroids
            (start_node_id, centroid_medoid_exchange_id, exchange_count,
             last_active_turn_idx, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(start_node_id) DO UPDATE SET
            centroid_medoid_exchange_id = excluded.centroid_medoid_exchange_id,
            exchange_count = excluded.exchange_count,
            last_active_turn_idx = excluded.last_active_turn_idx,
            updated_at = excluded.updated_at
    """, (topic_start_node_id, medoid_node_id, exchange_count,
          turn_idx, datetime.utcnow()))

    conn.commit()

    logger.debug(f"Updated centroid for topic {topic_start_node_id[:8]}...: "
                f"medoid={medoid_node_id[:8] if medoid_node_id else 'None'}...")

    return True


def update_topic_centroid(
    topic_start_node_id: str,
    topic_end_node_id: Optional[str] = None,
    force: bool = False,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Update the centroid for a topic if at a checkpoint.

    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        topic_end_node_id: Optional end_node_id (None for open topics)
        force: If True, update regardless of checkpoint
        conn: Optional existing connection

    Returns:
        True if centroid was updated, False otherwise
    """
    try:
        if conn is not None:
            return _update_topic_centroid_impl(
                conn, topic_start_node_id, topic_end_node_id, force
            )

        with get_connection() as c:
            return _update_topic_centroid_impl(
                c, topic_start_node_id, topic_end_node_id, force
            )

    except Exception as e:
        logger.error(f"Error updating topic centroid: {e}")
        return False


def get_topic_centroid(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[dict]:
    """
    Get centroid info for a topic.

    Returns dict with: centroid_medoid_exchange_id, exchange_count,
                      last_active_turn_idx, updated_at
    """
    def _query(c: sqlite3.Connection) -> Optional[dict]:
        cursor = c.execute("""
            SELECT centroid_medoid_exchange_id, exchange_count,
                   last_active_turn_idx, updated_at
            FROM topic_centroids
            WHERE start_node_id = ?
        """, (topic_start_node_id,))

        row = cursor.fetchone()
        if row:
            return {
                'centroid_medoid_exchange_id': row[0],
                'exchange_count': row[1],
                'last_active_turn_idx': row[2],
                'updated_at': row[3]
            }
        return None

    if conn is not None:
        return _query(conn)

    with get_connection() as c:
        return _query(c)


def backfill_centroids() -> int:
    """
    Backfill centroids for all existing topics.

    Returns the number of topics updated.
    """
    from episodic.db import get_recent_topics

    topics = get_recent_topics(limit=1000)
    updated = 0

    with get_connection() as conn:
        for topic in topics:
            if update_topic_centroid(
                topic['start_node_id'],
                topic.get('end_node_id'),
                force=True,
                conn=conn
            ):
                updated += 1

    logger.info(f"Backfilled centroids for {updated} topics")
    return updated
