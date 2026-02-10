"""
Topic deletion operations for Episodic database.

This module handles cascade deletion of topics and all related data:
- topic_centroids
- topic_nodes
- topic_working_set
- topics
- ChromaDB embeddings (via rag_collections)

Note: The nodes table is NOT deleted - conversation history is preserved.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

from .db_connection import get_connection

logger = logging.getLogger(__name__)


def get_topics_by_name(name: str) -> List[dict]:
    """
    Get topics matching an exact name.

    Args:
        name: Exact topic name to match

    Returns:
        List of topic dicts with start_node_id, name, created_at
    """
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT t.name, t.start_node_id, t.end_node_id, n.created_at
            FROM topics t
            LEFT JOIN nodes n ON t.start_node_id = n.id
            WHERE t.name = ?
            ORDER BY n.created_at DESC
        """, (name,))

        topics = []
        for row in cursor.fetchall():
            topics.append({
                'name': row[0],
                'start_node_id': row[1],
                'end_node_id': row[2],
                'created_at': row[3]
            })
        return topics


def get_topics_by_pattern(pattern: str) -> List[dict]:
    """
    Get topics matching a name pattern (case-insensitive substring match).

    Args:
        pattern: Pattern to search for in topic names

    Returns:
        List of topic dicts with start_node_id, name, created_at
    """
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT t.name, t.start_node_id, t.end_node_id, n.created_at
            FROM topics t
            LEFT JOIN nodes n ON t.start_node_id = n.id
            WHERE LOWER(t.name) LIKE ?
            ORDER BY n.created_at DESC
        """, (f'%{pattern.lower()}%',))

        topics = []
        for row in cursor.fetchall():
            topics.append({
                'name': row[0],
                'start_node_id': row[1],
                'end_node_id': row[2],
                'created_at': row[3]
            })
        return topics


def get_topics_by_time_range(
    start_utc: Optional[datetime],
    end_utc: Optional[datetime]
) -> List[dict]:
    """
    Get topics created within a time range.

    Args:
        start_utc: Start of range (UTC), or None for open start
        end_utc: End of range (UTC), or None for open end

    Returns:
        List of topic dicts with start_node_id, name, created_at
    """
    with get_connection() as conn:
        # Build query based on which bounds are provided
        if start_utc and end_utc:
            cursor = conn.execute("""
                SELECT t.name, t.start_node_id, t.end_node_id, n.created_at
                FROM topics t
                LEFT JOIN nodes n ON t.start_node_id = n.id
                WHERE n.created_at >= ? AND n.created_at < ?
                ORDER BY n.created_at DESC
            """, (start_utc.isoformat(), end_utc.isoformat()))
        elif start_utc:
            cursor = conn.execute("""
                SELECT t.name, t.start_node_id, t.end_node_id, n.created_at
                FROM topics t
                LEFT JOIN nodes n ON t.start_node_id = n.id
                WHERE n.created_at >= ?
                ORDER BY n.created_at DESC
            """, (start_utc.isoformat(),))
        elif end_utc:
            cursor = conn.execute("""
                SELECT t.name, t.start_node_id, t.end_node_id, n.created_at
                FROM topics t
                LEFT JOIN nodes n ON t.start_node_id = n.id
                WHERE n.created_at < ?
                ORDER BY n.created_at DESC
            """, (end_utc.isoformat(),))
        else:
            cursor = conn.execute("""
                SELECT t.name, t.start_node_id, t.end_node_id, n.created_at
                FROM topics t
                LEFT JOIN nodes n ON t.start_node_id = n.id
                ORDER BY n.created_at DESC
            """)

        topics = []
        for row in cursor.fetchall():
            topics.append({
                'name': row[0],
                'start_node_id': row[1],
                'end_node_id': row[2],
                'created_at': row[3]
            })
        return topics


def delete_topic_cascade(start_node_id: str, delete_embeddings: bool = True) -> dict:
    """
    Delete a topic and all related data in the correct order.

    Deletion order (to respect foreign key constraints):
    1. ChromaDB embeddings (if enabled)
    2. topic_working_set
    3. topic_nodes
    4. topic_centroids
    5. topics

    Args:
        start_node_id: The start_node_id that uniquely identifies the topic
        delete_embeddings: Whether to delete ChromaDB embeddings

    Returns:
        Dict with counts of deleted records from each table
    """
    deleted = {
        'embeddings': 0,
        'working_set': 0,
        'topic_nodes': 0,
        'centroids': 0,
        'topics': 0
    }

    # Get topic name for logging
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM topics WHERE start_node_id = ?",
            (start_node_id,)
        )
        row = cursor.fetchone()
        topic_name = row[0] if row else "unknown"

    logger.info(f"Deleting topic '{topic_name}' (start_node_id={start_node_id})")

    # 1. Delete ChromaDB embeddings for nodes in this topic
    if delete_embeddings:
        deleted['embeddings'] = _delete_topic_embeddings(start_node_id)

    with get_connection() as conn:
        # 2. Delete from topic_working_set
        cursor = conn.execute(
            "DELETE FROM topic_working_set WHERE topic_start_node_id = ?",
            (start_node_id,)
        )
        deleted['working_set'] = cursor.rowcount

        # 3. Delete from topic_nodes
        cursor = conn.execute(
            "DELETE FROM topic_nodes WHERE topic_start_node_id = ?",
            (start_node_id,)
        )
        deleted['topic_nodes'] = cursor.rowcount

        # 4. Delete from topic_centroids
        cursor = conn.execute(
            "DELETE FROM topic_centroids WHERE start_node_id = ?",
            (start_node_id,)
        )
        deleted['centroids'] = cursor.rowcount

        # 5. Delete from topics
        cursor = conn.execute(
            "DELETE FROM topics WHERE start_node_id = ?",
            (start_node_id,)
        )
        deleted['topics'] = cursor.rowcount

        conn.commit()

    logger.info(f"Deleted topic '{topic_name}': {deleted}")
    return deleted


def _delete_topic_embeddings(start_node_id: str) -> int:
    """
    Delete ChromaDB embeddings for nodes in a topic.

    Args:
        start_node_id: The topic's start_node_id

    Returns:
        Number of embeddings deleted
    """
    try:
        from episodic.rag_collections import get_multi_collection_rag, CollectionType

        # Get node IDs in this topic
        with get_connection() as conn:
            cursor = conn.execute("""
                SELECT node_id FROM topic_nodes
                WHERE topic_start_node_id = ?
            """, (start_node_id,))
            node_ids = [row[0] for row in cursor.fetchall()]

        if not node_ids:
            return 0

        # Delete from ChromaDB
        rag = get_multi_collection_rag()
        collection = rag.get_collection(CollectionType.CONVERSATION)

        # ChromaDB delete expects a list of IDs
        collection.delete(ids=node_ids)

        logger.debug(f"Deleted {len(node_ids)} embeddings for topic {start_node_id}")
        return len(node_ids)

    except Exception as e:
        logger.warning(f"Could not delete embeddings for topic {start_node_id}: {e}")
        return 0


def delete_topics_batch(
    topics: List[dict],
    delete_embeddings: bool = True
) -> Tuple[int, dict]:
    """
    Delete multiple topics.

    Args:
        topics: List of topic dicts (must have 'start_node_id')
        delete_embeddings: Whether to delete ChromaDB embeddings

    Returns:
        Tuple of (count_deleted, total_deleted_by_table)
    """
    total_deleted = {
        'embeddings': 0,
        'working_set': 0,
        'topic_nodes': 0,
        'centroids': 0,
        'topics': 0
    }

    for topic in topics:
        deleted = delete_topic_cascade(
            topic['start_node_id'],
            delete_embeddings=delete_embeddings
        )
        for key in total_deleted:
            total_deleted[key] += deleted[key]

    return len(topics), total_deleted


def check_tables_exist() -> dict:
    """
    Check which topic-related tables exist in the database.

    Returns:
        Dict mapping table names to boolean (exists or not)
    """
    tables_to_check = [
        'topics',
        'topic_nodes',
        'topic_centroids',
        'topic_working_set'
    ]

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN (?, ?, ?, ?)
        """, tables_to_check)

        existing = {row[0] for row in cursor.fetchall()}
        return {table: table in existing for table in tables_to_check}
