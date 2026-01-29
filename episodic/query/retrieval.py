"""
MQL Retrieval Execution

Executes ResolvedQuery against the database and returns matching nodes.

This module bridges the query understanding pipeline (parser/resolver)
with the database layer, applying temporal, speaker, and segment filters.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from .types import ResolvedQuery


@dataclass
class RetrievedNode:
    """A node retrieved from the database with metadata."""
    
    node_id: str
    short_id: str
    content: str
    role: str  # "user" or "assistant"
    created_at: datetime
    parent_id: Optional[str]
    topic_name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Serialize for debugging/logging."""
        return {
            "node_id": self.node_id,
            "short_id": self.short_id,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "topic_name": self.topic_name,
        }


@dataclass
class RetrievalResult:
    """Result of query execution."""
    
    nodes: List[RetrievedNode]
    total_matches: int
    query_applied: ResolvedQuery
    filters_applied: List[str]  # Human-readable filter descriptions
    execution_time_ms: float
    
    def is_empty(self) -> bool:
        """Check if no results were found."""
        return len(self.nodes) == 0
    
    def to_context_string(self, max_nodes: int = 10) -> str:
        """
        Format results as context string for LLM injection.
        
        Args:
            max_nodes: Maximum nodes to include
            
        Returns:
            Formatted context string
        """
        if self.is_empty():
            return ""
        
        lines = []
        for node in self.nodes[:max_nodes]:
            role_label = "You" if node.role == "user" else "Assistant"
            date_str = node.created_at.strftime("%Y-%m-%d") if node.created_at else "unknown"
            lines.append(f"[{date_str}] {role_label}: {node.content}")
        
        if len(self.nodes) > max_nodes:
            lines.append(f"... and {len(self.nodes) - max_nodes} more messages")
        
        return "\n\n".join(lines)


class QueryExecutor:
    """
    Executes ResolvedQuery against the database.
    
    Builds SQL with appropriate WHERE clauses for:
    - Temporal filtering (date ranges)
    - Speaker filtering (role)
    - Segment filtering (topic node cache)
    - Content search (LIKE matching)
    """
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def execute(
        self,
        query: ResolvedQuery,
        limit: int = 50,
        include_meta_queries: bool = False
    ) -> RetrievalResult:
        """
        Execute a ResolvedQuery and return matching nodes.
        
        Args:
            query: The resolved query to execute
            limit: Maximum nodes to return
            include_meta_queries: Whether to include meta-query nodes
            
        Returns:
            RetrievalResult with matching nodes
        """
        import time
        start_time = time.perf_counter()
        
        # Build SQL query
        sql, params, filters = self._build_sql(query, limit, include_meta_queries)
        
        # Execute
        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        # Convert to RetrievedNode objects
        nodes = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            nodes.append(self._row_to_node(row_dict))
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            nodes=nodes,
            total_matches=len(nodes),
            query_applied=query,
            filters_applied=filters,
            execution_time_ms=execution_time
        )
    
    def _build_sql(
        self,
        query: ResolvedQuery,
        limit: int,
        include_meta_queries: bool
    ) -> Tuple[str, List, List[str]]:
        """
        Build SQL query with filters.
        
        Returns:
            Tuple of (sql_string, parameters, filter_descriptions)
        """
        # Base query
        select = """
            SELECT DISTINCT n.id, n.short_id, n.content, n.role, n.created_at, n.parent_id
        """
        
        from_clause = "FROM nodes n"
        joins = []
        where_clauses = ["1=1"]  # Always true base
        params = []
        filters = []
        
        # Exclude meta-queries by default
        if not include_meta_queries:
            where_clauses.append("(n.is_meta_query IS NULL OR n.is_meta_query = 0)")
        
        # --- Temporal filter ---
        if query.temporal:
            start_utc, end_utc = query.temporal
            # SQLite stores timestamps as strings, need to compare appropriately
            where_clauses.append("n.created_at >= ?")
            where_clauses.append("n.created_at < ?")
            params.append(start_utc.strftime("%Y-%m-%d %H:%M:%S"))
            params.append(end_utc.strftime("%Y-%m-%d %H:%M:%S"))
            filters.append(f"temporal: {start_utc.date()} to {end_utc.date()}")
        
        # --- Speaker filter ---
        if query.speaker:
            if query.speaker == "user":
                where_clauses.append("n.role = 'user'")
                filters.append("speaker: user only")
            elif query.speaker == "assistant":
                where_clauses.append("n.role = 'assistant'")
                filters.append("speaker: assistant only")
            # "both" means no filter
        
        # --- Segment filter ---
        if query.segment_explicit and query.segment_resolved_ids:
            # segment_resolved_ids contains node IDs (resolved by the segment resolver)
            # Filter directly by node ID - no need to join topic_node_cache
            placeholders = ",".join("?" * len(query.segment_resolved_ids))
            where_clauses.append(f"n.id IN ({placeholders})")
            params.extend(query.segment_resolved_ids)
            filters.append(f"segment: {query.segment_query}")
        
        # --- Target/content filter ---
        if query.target:
            # Simple LIKE search - could be enhanced with FTS5
            where_clauses.append("n.content LIKE ?")
            params.append(f"%{query.target}%")
            filters.append(f"content contains: {query.target}")
        
        # Build final SQL
        sql_parts = [select, from_clause]
        if joins:
            sql_parts.extend(joins)
        sql_parts.append("WHERE " + " AND ".join(where_clauses))
        sql_parts.append("ORDER BY n.created_at DESC")
        sql_parts.append(f"LIMIT {limit}")
        
        sql = "\n".join(sql_parts)
        
        return sql, params, filters
    
    def _row_to_node(self, row: dict) -> RetrievedNode:
        """Convert database row to RetrievedNode."""
        created_at = None
        if row.get("created_at"):
            # Parse SQLite timestamp
            try:
                created_at = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
                created_at = created_at.replace(tzinfo=ZoneInfo("UTC"))
            except (ValueError, TypeError):
                pass
        
        return RetrievedNode(
            node_id=row["id"],
            short_id=row.get("short_id", ""),
            content=row.get("content", ""),
            role=row.get("role", "unknown"),
            created_at=created_at,
            parent_id=row.get("parent_id"),
            topic_name=row.get("topic_name"),
        )


def execute_query(
    query: ResolvedQuery,
    conn: sqlite3.Connection,
    limit: int = 50
) -> RetrievalResult:
    """
    Convenience function to execute a query.
    
    Args:
        query: The resolved query
        conn: Database connection
        limit: Maximum results
        
    Returns:
        RetrievalResult
    """
    executor = QueryExecutor(conn)
    return executor.execute(query, limit=limit)


def format_retrieval_for_context(
    result: RetrievalResult,
    max_tokens: int = 2000,
    include_header: bool = True
) -> str:
    """
    Format retrieval results for injection into LLM context.
    
    Args:
        result: The retrieval result
        max_tokens: Approximate token limit (rough estimate: 4 chars/token)
        include_header: Whether to include a context header
        
    Returns:
        Formatted string for context injection
    """
    if result.is_empty():
        return ""
    
    max_chars = max_tokens * 4
    lines = []
    
    if include_header:
        filter_desc = ", ".join(result.filters_applied) if result.filters_applied else "none"
        lines.append(f"[Retrieved {result.total_matches} relevant messages (filters: {filter_desc})]")
        lines.append("")
    
    current_chars = sum(len(line) for line in lines)
    
    for node in result.nodes:
        role_label = "User" if node.role == "user" else "Assistant"
        date_str = node.created_at.strftime("%b %d") if node.created_at else ""
        
        entry = f"[{date_str}] {role_label}: {node.content}"
        
        if current_chars + len(entry) > max_chars:
            lines.append(f"... ({result.total_matches - len(lines) + 2} more messages)")
            break
        
        lines.append(entry)
        current_chars += len(entry) + 2  # +2 for newlines
    
    return "\n".join(lines)
