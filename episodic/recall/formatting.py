"""
Block formatting for recall context.

Formats topic expansions and statement hits as labeled blocks for LLM context.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .expansion import TopicExpansion, ExpandedExchange
from .ranking import PromotedHit
from .budget import RecallBudget


@dataclass
class ConversationBlock:
    """A conversation block from a topic."""
    topic_id: int
    topic_name: str
    date_range: str  # e.g., "Jan 19-20" or "Jan 19"
    hit_count: int
    best_score: float
    is_compressed: bool
    summary: Optional[str]
    exchanges: List[ExpandedExchange]


@dataclass
class StatementBlock:
    """A single statement block (exchange pair)."""
    exchange_id: str
    topic_name: Optional[str]  # May be None if unassigned
    timestamp: str
    score: float
    user_content: str
    assistant_content: str


@dataclass
class FormattedRecall:
    """Complete formatted recall result."""
    conversation_blocks: List[ConversationBlock]
    statement_blocks: List[StatementBlock]
    total_exchanges: int
    
    def to_context_string(self, budget: Optional[RecallBudget] = None) -> str:
        """Format as string for LLM context injection."""
        lines = []
        
        # Conversation blocks first
        for block in self.conversation_blocks:
            lines.append(_format_conversation_block(block, budget))
        
        # Then statement blocks
        for block in self.statement_blocks:
            lines.append(_format_statement_block(block))
        
        if not lines:
            return ""
        
        return "\n\n".join(lines)


def format_recall_result(
    conn: sqlite3.Connection,
    topic_expansions: List[TopicExpansion],
    statement_hits: List[PromotedHit],
    topic_scores: dict,  # topic_id -> (best_score, hit_count)
    budget: RecallBudget
) -> FormattedRecall:
    """
    Format topic expansions and statement hits into blocks.
    
    Args:
        conn: SQLite connection
        topic_expansions: Expanded topics from expansion module
        statement_hits: Statement candidates from ranking module
        topic_scores: Scores for provenance display
        budget: Budget for formatting decisions
    
    Returns:
        FormattedRecall with conversation and statement blocks
    """
    conversation_blocks = []
    statement_blocks = []
    total_exchanges = 0
    
    # Format conversation blocks
    for expansion in topic_expansions:
        score_info = topic_scores.get(expansion.topic_id, (0.0, 0))
        best_score, hit_count = score_info
        
        # Get date range from exchanges
        date_range = _get_date_range(conn, expansion)
        
        block = ConversationBlock(
            topic_id=expansion.topic_id,
            topic_name=expansion.topic_name,
            date_range=date_range,
            hit_count=hit_count,
            best_score=best_score,
            is_compressed=expansion.is_compressed,
            summary=expansion.summary,
            exchanges=expansion.exchanges
        )
        conversation_blocks.append(block)
        total_exchanges += len(expansion.exchanges)
    
    # Format statement blocks - use Chroma metadata which has full exchange
    for hit in statement_hits:
        metadata = hit.metadata or {}
        
        # Get content from metadata (preferred) or SQLite (fallback)
        user_content = metadata.get('user_content', '')
        assistant_content = metadata.get('assistant_content', '')
        
        # Fallback to SQLite if metadata is empty
        if not user_content and not assistant_content:
            node = _get_node_from_conn(conn, hit.exchange_id)
            if node:
                user_content = node.get('content', '') if node.get('role') == 'user' else ''
                # Try to get assistant response
                # This is a fallback - ideally metadata should have it
            if not user_content:
                continue
        
        # Get timestamp from metadata or node
        timestamp = ''
        ts_str = metadata.get('timestamp')
        if ts_str:
            ts = _parse_timestamp(ts_str)
            if ts:
                timestamp = ts.strftime("%b %d")
        
        block = StatementBlock(
            exchange_id=hit.exchange_id,
            topic_name=None,  # Could look up from promotion.topic_info
            timestamp=timestamp,
            score=hit.similarity,
            user_content=user_content,
            assistant_content=assistant_content
        )
        statement_blocks.append(block)
        total_exchanges += 1
    
    return FormattedRecall(
        conversation_blocks=conversation_blocks,
        statement_blocks=statement_blocks,
        total_exchanges=total_exchanges
    )


def _format_conversation_block(block: ConversationBlock, budget: Optional[RecallBudget]) -> str:
    """Format a conversation block as string."""
    lines = []
    
    # Header with provenance
    header_parts = [f"[Conversation: {block.topic_name}"]
    
    if budget and budget.emphasize_timestamps and block.date_range:
        header_parts.append(f", {block.date_range}")
    
    header_parts.append(f", {block.hit_count} matches")
    
    if block.best_score > 0:
        header_parts.append(f", best={block.best_score:.2f}")
    
    header_parts.append("]")
    lines.append("".join(header_parts))
    
    # Summary for compressed topics
    if block.is_compressed and block.summary:
        lines.append(f"Summary: {block.summary}")
        if block.exchanges:
            lines.append("Relevant exchanges:")
    
    # Group exchanges into pairs (user + assistant)
    i = 0
    while i < len(block.exchanges):
        exchange = block.exchanges[i]
        role_label = "User" if exchange.role == "user" else "Assistant"
        anchor_marker = " *" if exchange.is_anchor else ""
        lines.append(f"{role_label}{anchor_marker}: {exchange.content}")
        i += 1
    
    return "\n".join(lines)


def _format_statement_block(block: StatementBlock) -> str:
    """Format a statement block as string."""
    lines = []
    
    # Header with provenance
    header_parts = ["[Statement"]
    
    if block.topic_name:
        header_parts.append(f": from {block.topic_name}")
    
    if block.timestamp:
        header_parts.append(f", {block.timestamp}")
    
    if block.score > 0:
        header_parts.append(f", score={block.score:.2f}")
    
    header_parts.append("]")
    lines.append("".join(header_parts))
    
    # Show both user question and assistant response
    if block.user_content:
        lines.append(f"User: {block.user_content}")
    if block.assistant_content:
        lines.append(f"Assistant: {block.assistant_content}")
    
    return "\n".join(lines)


def _get_date_range(conn: sqlite3.Connection, expansion: TopicExpansion) -> str:
    """Get date range string from expansion's exchanges."""
    if not expansion.exchanges:
        return ""
    
    # Get timestamps from first and last exchanges
    first_id = expansion.exchanges[0].node_id
    last_id = expansion.exchanges[-1].node_id
    
    first_node = _get_node_from_conn(conn, first_id)
    last_node = _get_node_from_conn(conn, last_id)
    
    first_ts = _parse_timestamp(first_node.get('created_at') if first_node else None)
    last_ts = _parse_timestamp(last_node.get('created_at') if last_node else None)
    
    if not first_ts:
        return ""
    
    first_str = first_ts.strftime("%b %d")
    
    if not last_ts or first_ts.date() == last_ts.date():
        return first_str
    
    last_str = last_ts.strftime("%d") if first_ts.month == last_ts.month else last_ts.strftime("%b %d")
    return f"{first_str}-{last_str}"


def _get_node_timestamp(node: Optional[dict]) -> str:
    """Get formatted timestamp from node."""
    if not node:
        return ""
    
    ts = _parse_timestamp(node.get('created_at'))
    if not ts:
        return ""
    
    return ts.strftime("%b %d")


def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse SQLite timestamp string."""
    if not ts_str:
        return None
    
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except:
            return None


def _get_node_from_conn(conn: sqlite3.Connection, node_id: str) -> Optional[dict]:
    """Get node by ID using the provided connection (not global)."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, short_id, parent_id, content, role, created_at FROM nodes WHERE id = ? OR short_id = ?",
            (node_id, node_id)
        )
        row = cursor.fetchone()
        if row:
            # Handle both dict-like Row and tuple
            if hasattr(row, 'keys'):
                return dict(row)
            else:
                return {
                    'id': row[0],
                    'short_id': row[1],
                    'parent_id': row[2],
                    'content': row[3],
                    'role': row[4],
                    'created_at': row[5]
                }
        return None
    except Exception:
        return None
