"""
Database wrapper functions for backward compatibility during schema migration.

These functions provide a compatibility layer that works with both the old
table names (manual_index_scores) and new table names (topic_detection_scores).
"""

import sqlite3
from typing import List, Dict, Any, Optional
from episodic.db import get_connection


def get_detection_scores_table_name(conn: sqlite3.Connection) -> str:
    """Determine which table name to use for topic detection scores."""
    cursor = conn.cursor()
    
    # Check for new table
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='topic_detection_scores'
    """)
    if cursor.fetchone():
        return 'topic_detection_scores'
    
    # Fall back to old table
    return 'manual_index_scores'


def store_topic_detection_score(
    user_node_short_id: str,
    window_size: int,
    window_a_start_short_id: str,
    window_a_end_short_id: str,
    window_a_size: int,
    window_b_start_short_id: str,
    window_b_end_short_id: str,
    window_b_size: int,
    drift_score: float,
    keyword_score: float,
    combined_score: float,
    is_boundary: bool,
    transition_phrase: Optional[str] = None,
    threshold_used: Optional[float] = None,
    detection_method: str = 'sliding_window'
) -> None:
    """Store topic detection scores (works with both old and new schema)."""
    with get_connection() as conn:
        table_name = get_detection_scores_table_name(conn)
        cursor = conn.cursor()
        
        # Check if detection_method column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        has_method = 'detection_method' in columns
        
        if has_method:
            # New schema with detection_method
            cursor.execute(f"""
                INSERT OR REPLACE INTO {table_name} (
                    user_node_short_id, window_size, detection_method,
                    window_a_start_short_id, window_a_end_short_id, window_a_size,
                    window_b_start_short_id, window_b_end_short_id, window_b_size,
                    drift_score, keyword_score, combined_score,
                    is_boundary, transition_phrase, threshold_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_node_short_id, window_size, detection_method,
                window_a_start_short_id, window_a_end_short_id, window_a_size,
                window_b_start_short_id, window_b_end_short_id, window_b_size,
                drift_score, keyword_score, combined_score,
                is_boundary, transition_phrase, threshold_used
            ))
        else:
            # Old schema without detection_method
            cursor.execute(f"""
                INSERT OR REPLACE INTO {table_name} (
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


def get_topic_detection_scores(
    window_size: Optional[int] = None,
    detection_method: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get topic detection scores (works with both old and new schema)."""
    with get_connection() as conn:
        table_name = get_detection_scores_table_name(conn)
        cursor = conn.cursor()
        
        # Check if detection_method column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        has_method = 'detection_method' in columns
        
        # Build query
        query = f"""
            SELECT t.*, n.content
            FROM {table_name} t
            JOIN nodes n ON n.short_id = t.user_node_short_id
        """
        
        conditions = []
        params = []
        
        if window_size:
            conditions.append("t.window_size = ?")
            params.append(window_size)
        
        if detection_method and has_method:
            conditions.append("t.detection_method = ?")
            params.append(detection_method)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY n.ROWID"
        
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            record = {}
            for i, col in enumerate(columns):
                record[col] = row[i]
            # Add default detection_method if not present
            if not has_method:
                record['detection_method'] = 'sliding_window'
            results.append(record)
        
        return results


def clear_topic_detection_scores() -> None:
    """Clear all topic detection scores (works with both old and new schema)."""
    with get_connection() as conn:
        table_name = get_detection_scores_table_name(conn)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()


# Backward compatibility aliases
store_manual_index_score = store_topic_detection_score
get_manual_index_scores = get_topic_detection_scores
clear_manual_index_scores = clear_topic_detection_scores