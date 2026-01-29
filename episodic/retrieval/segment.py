"""
Segment membership and caching.

Implements v1.1 spec section 6.
"""
import sqlite3
import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)

# Module-level cache
_segment_cache: Dict[int, 'SegmentCacheEntry'] = {}


@dataclass
class SegmentCacheEntry:
    """Cache entry for segment nodes."""
    effective_end: str
    nodes_list: List[str]
    nodes_set: Set[str]


def get_head(conn: sqlite3.Connection) -> Optional[str]:
    """Get current head node ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT head_id FROM state WHERE name = 'head'")
    row = cursor.fetchone()
    return row['head_id'] if row else None


def get_topic(conn: sqlite3.Connection, segment_id: int) -> Optional[Dict]:
    """Get topic by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM topics WHERE id = ?", (segment_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_all_topics(conn: sqlite3.Connection) -> List[Dict]:
    """Get all topics ordered by id ASC (required for overlap resolution)."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM topics ORDER BY id ASC")
    return [dict(row) for row in cursor.fetchall()]


def get_node(conn: sqlite3.Connection, node_id: str) -> Optional[Dict]:
    """Get node by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def build_ancestry_map(conn: sqlite3.Connection, end_id: str) -> Dict[str, Optional[str]]:
    """
    Build ancestry map from end_id to root in single recursive CTE.
    
    Returns:
        Dict mapping node_id -> parent_id for all ancestors
    """
    cursor = conn.cursor()
    cursor.execute("""
        WITH RECURSIVE ancestors AS (
            SELECT id, parent_id FROM nodes WHERE id = ?
            UNION ALL
            SELECT n.id, n.parent_id FROM nodes n
            JOIN ancestors a ON n.id = a.parent_id
        )
        SELECT id, parent_id FROM ancestors
    """, (end_id,))
    return {row['id']: row['parent_id'] for row in cursor.fetchall()}


def compute_segment_nodes(
    conn: sqlite3.Connection,
    segment_id: int,
    effective_end: str
) -> Tuple[List[str], Set[str]]:
    """
    Compute segment nodes via batched ancestry traversal.
    
    Returns:
        (ordered_list, membership_set) or ([], set()) on error
    """
    topic = get_topic(conn, segment_id)
    if not topic:
        logger.debug(f"AUDIT: Segment {segment_id} not found")
        return [], set()
    
    start_id = topic['start_node_id']
    ancestry_map = build_ancestry_map(conn, effective_end)
    
    nodes = []
    current_id = effective_end
    
    while current_id is not None:
        if current_id not in ancestry_map:
            logger.debug(f"AUDIT: Segment {segment_id} node {current_id} not in ancestry")
            return [], set()
        
        nodes.append(current_id)
        if current_id == start_id:
            break
        current_id = ancestry_map[current_id]
    
    if not nodes or nodes[-1] != start_id:
        logger.debug(f"AUDIT: Segment {segment_id} start_node not reached")
        return [], set()
    
    # Reverse to get oldest->newest order
    ordered = list(reversed(nodes))
    return ordered, set(ordered)


def get_cached_segment_nodes(
    conn: sqlite3.Connection,
    segment_id: int
) -> Tuple[List[str], Set[str]]:
    """
    Get segment nodes with caching.
    
    Cache invalidates when effective_end changes (ongoing segments).
    """
    topic = get_topic(conn, segment_id)
    if not topic:
        return [], set()
    
    effective_end = topic['end_node_id'] or get_head(conn)
    if not effective_end:
        logger.debug(f"AUDIT: Segment {segment_id} has no effective end (no head)")
        return [], set()
    
    cached = _segment_cache.get(segment_id)
    if cached and cached.effective_end == effective_end:
        return cached.nodes_list, cached.nodes_set
    
    nodes_list, nodes_set = compute_segment_nodes(conn, segment_id, effective_end)
    _segment_cache[segment_id] = SegmentCacheEntry(effective_end, nodes_list, nodes_set)
    return nodes_list, nodes_set
