"""
Topic detection scoring operations for Episodic database.

This module handles storing and retrieving topic detection scores
and manual indexing scores.
"""

import json
import logging
from typing import Optional, List, Dict, Any

from .db_connection import get_connection

# Set up logging
logger = logging.getLogger(__name__)


def store_topic_detection_scores(
    user_node_short_id: str,
    detection_method: str,
    current_topic: Optional[str] = None,
    messages_in_topic: int = 0,
    drift_score: float = 0.0,
    keyword_score: float = 0.0,
    combined_score: float = 0.0,
    effective_threshold: float = 0.9,
    topic_changed: bool = False,
    detection_response: Optional[str] = None
):
    """
    Store topic detection scores and metadata for analysis.
    
    Args:
        user_node_short_id: Short ID of the user node
        detection_method: Method used for detection (llm, sliding_window, hybrid, etc.)
        current_topic: Current topic name
        messages_in_topic: Number of messages in current topic
        drift_score: Semantic drift score
        keyword_score: Keyword-based score
        combined_score: Combined/final score
        effective_threshold: Threshold used for this detection
        topic_changed: Whether a topic change was detected
        detection_response: Raw response from detection (if applicable)
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # Check if the table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topic_detection_scores'")
        if not c.fetchone():
            logger.warning("topic_detection_scores table does not exist")
            return
            
        c.execute("""
            INSERT INTO topic_detection_scores (
                user_node_short_id, detection_method, current_topic,
                messages_in_topic, drift_score, keyword_score,
                combined_score, effective_threshold, topic_changed,
                detection_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_node_short_id, detection_method, current_topic,
            messages_in_topic, drift_score, keyword_score,
            combined_score, effective_threshold, topic_changed,
            detection_response
        ))
        
        conn.commit()


def get_topic_detection_scores(user_node_id: str = None, limit: int = 100):
    """
    Get topic detection scores from the database.
    
    Args:
        user_node_id: Optional user node ID to filter by
        limit: Maximum number of records to return
        
    Returns:
        List of score dictionaries
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # Check if the table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topic_detection_scores'")
        if not c.fetchone():
            return []
            
        if user_node_id:
            c.execute("""
                SELECT * FROM topic_detection_scores 
                WHERE user_node_short_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_node_id, limit))
        else:
            c.execute("""
                SELECT * FROM topic_detection_scores 
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
        columns = [description[0] for description in c.description]
        scores = []
        
        for row in c.fetchall():
            score_dict = dict(zip(columns, row))
            scores.append(score_dict)
            
        return scores


def store_manual_index_score(
    user_node_short_id: str,
    window_size: int,
    window_a_start_short_id: str,
    window_a_end_short_id: str,
    window_a_size: int,
    window_b_start_short_id: str,
    window_b_end_short_id: str,
    window_b_size: int,
    drift_score: float,
    keyword_score: float = 0.0,
    combined_score: float = 0.0,
    is_boundary: bool = False,
    transition_phrase: Optional[str] = None,
    threshold_used: float = 0.9
):
    """
    Store manual index scoring data (window-based detection).
    
    This is used by the sliding window detection system to store
    detailed window comparison data.
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO manual_index_scores (
                user_node_short_id, window_size,
                window_a_start_short_id, window_a_end_short_id, window_a_size,
                window_b_start_short_id, window_b_end_short_id, window_b_size,
                drift_score, keyword_score, combined_score,
                is_boundary, transition_phrase, threshold_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_node_short_id, window_size,
            window_a_start_short_id, window_a_end_short_id, window_a_size,
            window_b_start_short_id, window_b_end_short_id, window_b_size,
            drift_score, keyword_score, combined_score,
            is_boundary, transition_phrase, threshold_used
        ))
        
        conn.commit()


def get_manual_index_scores(window_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get manual index scores from the database.
    
    Args:
        window_size: Optional window size to filter by
        
    Returns:
        List of score dictionaries
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        if window_size is not None:
            c.execute("""
                SELECT * FROM manual_index_scores 
                WHERE window_size = ?
                ORDER BY created_at DESC
            """, (window_size,))
        else:
            c.execute("""
                SELECT * FROM manual_index_scores 
                ORDER BY created_at DESC
            """)
            
        columns = [description[0] for description in c.description]
        scores = []
        
        for row in c.fetchall():
            score_dict = dict(zip(columns, row))
            scores.append(score_dict)
            
        return scores


def clear_manual_index_scores() -> None:
    """Clear all manual index scores from the database."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM manual_index_scores")
        conn.commit()