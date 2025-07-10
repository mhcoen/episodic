"""
Topic operations for Episodic database.

This module handles all topic-related database operations including
storing, retrieving, and updating conversation topics.
"""

import logging
from typing import Optional, List, Dict, Any

from .db_connection import get_connection

# Set up logging
logger = logging.getLogger(__name__)


def store_topic(name: str, start_node_id: str, end_node_id: Optional[str] = None, confidence: str = None):
    """
    Store a topic in the database.
    
    Args:
        name: Topic name
        start_node_id: ID of the first node in the topic
        end_node_id: ID of the last node in the topic (optional)
        confidence: Confidence level (e.g., 'detected', 'initial', 'manual')
    """
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO topics (name, start_node_id, end_node_id, confidence)
            VALUES (?, ?, ?, ?)
        """, (name, start_node_id, end_node_id, confidence))
        conn.commit()


def get_recent_topics(limit: int = 10):
    """
    Get recent topics from the database.
    
    Args:
        limit: Maximum number of topics to return
        
    Returns:
        List of topic dictionaries ordered by creation time (oldest first)
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # First check if the confidence column exists
        c.execute("PRAGMA table_info(topics)")
        columns = [column[1] for column in c.fetchall()]
        has_confidence = 'confidence' in columns
        
        if has_confidence:
            c.execute("""
                SELECT name, start_node_id, end_node_id, confidence 
                FROM topics 
                ORDER BY ROWID DESC 
                LIMIT ?
            """, (limit,))
        else:
            c.execute("""
                SELECT name, start_node_id, end_node_id, NULL as confidence 
                FROM topics 
                ORDER BY ROWID DESC 
                LIMIT ?
            """, (limit,))
        
        topics = []
        for row in c.fetchall():
            topics.append({
                'name': row[0],
                'start_node_id': row[1],
                'end_node_id': row[2],
                'confidence': row[3]
            })
        
        # Return in chronological order (oldest first)
        return list(reversed(topics))


def get_all_topics():
    """Get all topics from the database."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM topics ORDER BY ROWID ASC")
        columns = [description[0] for description in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]


def update_topic_end_node(topic_name: str, start_node_id: str, new_end_node_id: str):
    """
    Update the end node of a topic.
    
    Args:
        topic_name: Name of the topic
        start_node_id: Start node ID (for uniquely identifying the topic)
        new_end_node_id: New end node ID
        
    Returns:
        Number of rows updated
    """
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE topics 
            SET end_node_id = ? 
            WHERE name = ? AND start_node_id = ?
        """, (new_end_node_id, topic_name, start_node_id))
        return c.rowcount


def update_topic_name(old_name: str, start_node_id: str, new_name: str):
    """
    Update the name of a topic.
    
    Args:
        old_name: Current topic name
        start_node_id: Start node ID (for uniquely identifying the topic)
        new_name: New topic name
        
    Returns:
        Number of rows updated
    """
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE topics 
            SET name = ? 
            WHERE name = ? AND start_node_id = ?
        """, (new_name, old_name, start_node_id))
        rows_updated = c.rowcount
        
        if rows_updated > 0:
            logger.info(f"Updated topic name from '{old_name}' to '{new_name}' (start_node: {start_node_id})")
        else:
            logger.warning(f"No topic found with name '{old_name}' and start_node_id '{start_node_id}'")
            
        return rows_updated