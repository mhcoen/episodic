"""
Lexical retrieval using SQLite FTS5.

Implements v1.1 spec section 8.
"""
import sqlite3
import logging
import re
import uuid
from typing import List, Dict, Optional, Tuple

from .segment_filter import SegmentFilter, FilterKind, plan_sql_filter

logger = logging.getLogger(__name__)


def execute_lexical_search(
    conn: sqlite3.Connection,
    target: str,
    segment_filter: SegmentFilter,
    speaker: Optional[str],
    temporal: Optional[Tuple[str, str]],
    limit: int,
    config: dict
) -> List[Dict]:
    """
    Execute FTS5 lexical search with all filters.

    Args:
        conn: Database connection (must have FTS5 migrated)
        target: Search query text
        segment_filter: Segment scope filter
        speaker: Optional role filter ('user' or 'assistant')
        temporal: Optional (start_utc, end_utc) half-open interval
        limit: Maximum results
        config: Must have segment_filter_in_clause_max, sqlite_max_variable_number

    Returns:
        List of result dicts with id, content, role, parent_id, created_at,
        parent_role, bm25_score
    """
    # Ensure row_factory is set for dict conversion
    conn.row_factory = sqlite3.Row

    # Handle EMPTY immediately
    if segment_filter.kind == FilterKind.EMPTY:
        return []
    
    # Build query parts
    conditions = ["nodes_fts MATCH ?"]
    params = [target]

    # Always exclude meta-queries from retrieval
    conditions.append("(n.is_meta_query IS NULL OR n.is_meta_query = FALSE)")

    if speaker:
        conditions.append("n.role = ?")
        params.append(speaker)
        # When filtering to user nodes, only include complete exchanges (with assistant response)
        if speaker == 'user':
            conditions.append("EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n.id AND c.role = 'assistant')")

    if temporal:
        start_utc, end_utc = temporal
        conditions.append("n.created_at >= ?")
        conditions.append("n.created_at < ?")
        params.extend([start_utc, end_utc])
    
    # Count params before segment filter (include LIMIT)
    other_param_count = len(params) + 1
    
    # Plan segment filter SQL form
    planned_filter = plan_sql_filter(segment_filter, other_param_count, config)
    
    # Build segment clause
    segment_join = ""
    temp_table_name = None
    
    try:
        if planned_filter.kind == FilterKind.NONE:
            pass  # No segment clause
        elif planned_filter.kind == FilterKind.IN_CLAUSE:
            placeholders = ', '.join(['?'] * len(planned_filter.node_ids))
            conditions.append(f"n.id IN ({placeholders})")
            params.extend(planned_filter.node_ids)
        elif planned_filter.kind == FilterKind.PENDING_IDS:
            # Budget exceeded, use temp table
            temp_table_name = _create_temp_table(conn, planned_filter.node_ids)
            segment_join = f"JOIN {temp_table_name} sf ON n.id = sf.node_id"
        elif planned_filter.kind == FilterKind.TEMP_TABLE:
            segment_join = f"JOIN {planned_filter.table_name} sf ON n.id = sf.node_id"
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
            SELECT n.id, n.content, n.role, n.parent_id, n.created_at,
                   p.role as parent_role,
                   -bm25(nodes_fts) as bm25_score
            FROM nodes_fts
            JOIN nodes n ON nodes_fts.rowid = n.rowid
            LEFT JOIN nodes p ON n.parent_id = p.id
            {segment_join}
            WHERE {where_clause}
            ORDER BY bm25_score DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    
    finally:
        if temp_table_name:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
            except Exception as e:
                logger.warning(f"Failed to drop temp table {temp_table_name}: {e}")


def _create_temp_table(conn: sqlite3.Connection, node_ids: List[str]) -> str:
    """Create temp table with safe name."""
    suffix = uuid.uuid4().hex[:8]
    name = f"seg_filter_{suffix}"
    assert re.match(r'^[a-zA-Z0-9_]+$', name)
    
    cursor = conn.cursor()
    cursor.execute(f"CREATE TEMP TABLE {name} (node_id TEXT PRIMARY KEY)")
    cursor.executemany(
        f"INSERT INTO {name} (node_id) VALUES (?)",
        [(nid,) for nid in node_ids]
    )
    return name


def get_recent_exchanges(
    conn: sqlite3.Connection,
    limit: int,
    segment_filter: SegmentFilter,
    temporal: Optional[Tuple[str, str]]
) -> List[Dict]:
    """
    Get recent exchanges (user nodes) for browse mode with empty target.

    Returns user nodes ordered by created_at DESC.
    """
    # Ensure row_factory is set for dict conversion
    conn.row_factory = sqlite3.Row

    if segment_filter.kind == FilterKind.EMPTY:
        return []
    
    conditions = ["n.role = 'user'"]
    params = []

    # Always exclude meta-queries from retrieval
    conditions.append("(n.is_meta_query IS NULL OR n.is_meta_query = FALSE)")

    # Only include user nodes that have an assistant response (complete exchanges)
    conditions.append("EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n.id AND c.role = 'assistant')")

    if temporal:
        start_utc, end_utc = temporal
        conditions.append("n.created_at >= ?")
        conditions.append("n.created_at < ?")
        params.extend([start_utc, end_utc])
    
    segment_join = ""
    temp_table_name = None
    
    try:
        if segment_filter.kind == FilterKind.IN_CLAUSE:
            placeholders = ', '.join(['?'] * len(segment_filter.node_ids))
            conditions.append(f"n.id IN ({placeholders})")
            params.extend(segment_filter.node_ids)
        elif segment_filter.kind in (FilterKind.PENDING_IDS, FilterKind.TEMP_TABLE):
            if segment_filter.kind == FilterKind.PENDING_IDS:
                temp_table_name = _create_temp_table(conn, segment_filter.node_ids)
                segment_join = f"JOIN {temp_table_name} sf ON n.id = sf.node_id"
            else:
                segment_join = f"JOIN {segment_filter.table_name} sf ON n.id = sf.node_id"
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
            SELECT n.id, n.content, n.role, n.parent_id, n.created_at,
                   n.id as exchange_id
            FROM nodes n
            {segment_join}
            WHERE {where_clause}
            ORDER BY n.created_at DESC, n.id ASC
            LIMIT ?
        """
        params.append(limit)
        
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    
    finally:
        if temp_table_name:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
            except Exception as e:
                logger.warning(f"Failed to drop temp table: {e}")


def node_to_exchange_id(node: Dict, parent_role: Optional[str] = None) -> Optional[str]:
    """
    Map node to exchange_id. Pure function, no DB calls.
    
    Rules per spec 5.2:
    - user -> node.id
    - assistant -> node.parent_id if parent_role == 'user'
    - system/unknown -> None
    """
    role = node.get('role')
    
    if role == 'user':
        return node['id']
    elif role == 'assistant':
        parent_id = node.get('parent_id')
        if parent_id and parent_role == 'user':
            return parent_id
        return None
    else:
        return None
