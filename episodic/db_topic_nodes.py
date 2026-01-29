"""
Topic node membership operations for Episodic database.

This module manages the topic_nodes table which provides fast O(1) lookup
for "which nodes belong to topic X" without scanning ancestry.

Required for topic-local context assembly where resuming topic A
excludes topic B from the prompt entirely.
"""

import logging
import sqlite3
from typing import Optional, List, Dict, Any

from .db_connection import get_connection

logger = logging.getLogger(__name__)


def add_node_to_topic(
    topic_start_node_id: str,
    node_id: str,
    role: str,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Add a node to a topic's membership set.
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        node_id: The node to add
        role: 'user' or 'assistant'
        conn: Optional existing connection
        
    Returns:
        True if inserted, False if already existed or error
    """
    def _add(c: sqlite3.Connection) -> bool:
        cursor = c.cursor()
        
        # Get turn_idx (rowid) for the node
        cursor.execute("SELECT rowid FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Node {node_id[:8]}... not found, cannot add to topic")
            return False
        turn_idx = row[0]
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO topic_nodes 
                (topic_start_node_id, node_id, turn_idx, role)
                VALUES (?, ?, ?, ?)
            """, (topic_start_node_id, node_id, turn_idx, role))
            c.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error adding node to topic: {e}")
            return False
    
    if conn is not None:
        return _add(conn)
    
    with get_connection() as c:
        return _add(c)


def add_nodes_to_topic_range(
    topic_start_node_id: str,
    from_node_id: str,
    to_node_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Add all nodes in a range to a topic's membership set.
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        from_node_id: Start of range (inclusive)
        to_node_id: End of range (inclusive), or None for "to current head"
        conn: Optional existing connection
        
    Returns:
        Number of nodes added
    """
    def _add_range(c: sqlite3.Connection) -> int:
        cursor = c.cursor()
        
        # Get turn_idx for from_node
        cursor.execute("SELECT rowid FROM nodes WHERE id = ?", (from_node_id,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"From node {from_node_id[:8]}... not found")
            return 0
        from_idx = row[0]
        
        # Get turn_idx for to_node (or max)
        if to_node_id:
            cursor.execute("SELECT rowid FROM nodes WHERE id = ?", (to_node_id,))
            row = cursor.fetchone()
            to_idx = row[0] if row else None
        else:
            cursor.execute("SELECT MAX(rowid) FROM nodes")
            to_idx = cursor.fetchone()[0]
        
        if to_idx is None:
            return 0
        
        # Get all nodes in range
        cursor.execute("""
            SELECT id, rowid, role FROM nodes
            WHERE rowid >= ? AND rowid <= ?
            AND role IN ('user', 'assistant')
        """, (from_idx, to_idx))
        
        nodes = cursor.fetchall()
        inserted = 0
        
        for node_id, turn_idx, role in nodes:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO topic_nodes 
                    (topic_start_node_id, node_id, turn_idx, role)
                    VALUES (?, ?, ?, ?)
                """, (topic_start_node_id, node_id, turn_idx, role))
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.Error:
                pass
        
        c.commit()
        return inserted
    
    if conn is not None:
        return _add_range(conn)
    
    with get_connection() as c:
        return _add_range(c)


def get_topic_nodes(
    topic_start_node_id: str,
    limit: Optional[int] = None,
    role: Optional[str] = None,
    order: str = "DESC",
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get nodes belonging to a topic.
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        limit: Maximum number of nodes to return
        role: Filter by role ('user', 'assistant'), or None for all
        order: 'ASC' for oldest first, 'DESC' for newest first
        conn: Optional existing connection
        
    Returns:
        List of dicts with node_id, turn_idx, role
    """
    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.cursor()
        
        query = """
            SELECT node_id, turn_idx, role 
            FROM topic_nodes
            WHERE topic_start_node_id = ?
        """
        params: List[Any] = [topic_start_node_id]
        
        if role:
            query += " AND role = ?"
            params.append(role)
        
        query += f" ORDER BY turn_idx {order}"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        
        return [
            {'node_id': row[0], 'turn_idx': row[1], 'role': row[2]}
            for row in cursor.fetchall()
        ]
    
    if conn is not None:
        return _get(conn)
    
    with get_connection() as c:
        return _get(c)


def get_last_n_exchanges_in_topic(
    topic_start_node_id: str,
    n: int = 2,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get the last N user+assistant exchange pairs from a topic.
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        n: Number of exchanges (user+assistant pairs)
        conn: Optional existing connection
        
    Returns:
        List of dicts with user_node_id, user_content, assistant_node_id, assistant_content
        Ordered oldest to newest
    """
    def _get_exchanges(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.cursor()
        
        # Get the last 2*n nodes (to ensure we have n complete exchanges)
        cursor.execute("""
            SELECT tn.node_id, tn.turn_idx, tn.role, n.content
            FROM topic_nodes tn
            JOIN nodes n ON tn.node_id = n.id
            WHERE tn.topic_start_node_id = ?
            AND tn.role IN ('user', 'assistant')
            ORDER BY tn.turn_idx DESC
            LIMIT ?
        """, (topic_start_node_id, n * 3))  # Extra buffer for mismatched pairs
        
        rows = cursor.fetchall()
        
        # Build exchanges by pairing user/assistant
        exchanges = []
        i = 0
        while i < len(rows) and len(exchanges) < n:
            # Look for assistant followed by user (since we're going backwards)
            if rows[i][2] == 'assistant':
                asst_node_id, asst_turn_idx, _, asst_content = rows[i]
                # Find the preceding user message
                if i + 1 < len(rows) and rows[i + 1][2] == 'user':
                    user_node_id, user_turn_idx, _, user_content = rows[i + 1]
                    exchanges.append({
                        'user_node_id': user_node_id,
                        'user_content': user_content,
                        'assistant_node_id': asst_node_id,
                        'assistant_content': asst_content,
                        'turn_idx': user_turn_idx
                    })
                    i += 2
                    continue
            i += 1
        
        # Return in chronological order
        return list(reversed(exchanges))
    
    if conn is not None:
        return _get_exchanges(conn)
    
    with get_connection() as c:
        return _get_exchanges(c)


def get_node_topic(
    node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[str]:
    """
    Get the topic that contains a node.
    
    Args:
        node_id: The node to look up
        conn: Optional existing connection
        
    Returns:
        topic_start_node_id if found, None otherwise
    """
    def _get(c: sqlite3.Connection) -> Optional[str]:
        cursor = c.cursor()
        cursor.execute("""
            SELECT topic_start_node_id FROM topic_nodes
            WHERE node_id = ?
            LIMIT 1
        """, (node_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    if conn is not None:
        return _get(conn)
    
    with get_connection() as c:
        return _get(c)


def count_topic_nodes(
    topic_start_node_id: str,
    role: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Count nodes in a topic.
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        role: Filter by role, or None for all
        conn: Optional existing connection
        
    Returns:
        Number of nodes
    """
    def _count(c: sqlite3.Connection) -> int:
        cursor = c.cursor()
        
        if role:
            cursor.execute("""
                SELECT COUNT(*) FROM topic_nodes
                WHERE topic_start_node_id = ? AND role = ?
            """, (topic_start_node_id, role))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM topic_nodes
                WHERE topic_start_node_id = ?
            """, (topic_start_node_id,))
        
        return cursor.fetchone()[0]
    
    if conn is not None:
        return _count(conn)
    
    with get_connection() as c:
        return _count(c)


def ensure_topic_working_set(
    topic_start_node_id: str,
    topic_name: str,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Ensure a topic has a working set entry (creates if missing).
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        topic_name: Name of the topic
        conn: Optional existing connection
        
    Returns:
        True if created or already existed
    """
    def _ensure(c: sqlite3.Connection) -> bool:
        cursor = c.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO topic_working_set 
                (topic_start_node_id, topic_name)
                VALUES (?, ?)
            """, (topic_start_node_id, topic_name))
            c.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error ensuring topic working set: {e}")
            return False
    
    if conn is not None:
        return _ensure(conn)
    
    with get_connection() as c:
        return _ensure(c)


def get_topic_working_set(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a topic's working set state.
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        conn: Optional existing connection
        
    Returns:
        Dict with summary_md, decisions_json, open_loops_json, etc. or None
    """
    def _get(c: sqlite3.Connection) -> Optional[Dict[str, Any]]:
        cursor = c.cursor()
        cursor.execute("""
            SELECT topic_name, summary_md, decisions_json, open_loops_json,
                   entities_json, last_summarized_turn_idx, last_updated_at,
                   summary_version
            FROM topic_working_set
            WHERE topic_start_node_id = ?
        """, (topic_start_node_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'topic_start_node_id': topic_start_node_id,
            'topic_name': row[0],
            'summary_md': row[1],
            'decisions_json': row[2],
            'open_loops_json': row[3],
            'entities_json': row[4],
            'last_summarized_turn_idx': row[5],
            'last_updated_at': row[6],
            'summary_version': row[7]
        }
    
    if conn is not None:
        return _get(conn)
    
    with get_connection() as c:
        return _get(c)


def update_topic_summary(
    topic_start_node_id: str,
    summary_md: str,
    last_summarized_turn_idx: int,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Update a topic's summary in its working set.
    
    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        summary_md: The new summary text
        last_summarized_turn_idx: Turn index up to which summary covers
        conn: Optional existing connection
        
    Returns:
        True if updated successfully
    """
    def _update(c: sqlite3.Connection) -> bool:
        cursor = c.cursor()
        try:
            cursor.execute("""
                UPDATE topic_working_set
                SET summary_md = ?,
                    last_summarized_turn_idx = ?,
                    last_updated_at = CURRENT_TIMESTAMP,
                    summary_version = summary_version + 1
                WHERE topic_start_node_id = ?
            """, (summary_md, last_summarized_turn_idx, topic_start_node_id))
            c.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating topic summary: {e}")
            return False
    
    if conn is not None:
        return _update(conn)
    
    with get_connection() as c:
        return _update(c)
