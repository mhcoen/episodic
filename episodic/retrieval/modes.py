"""
Mode-specific response formatting.

Implements v1.1 spec section 12.
"""
import sqlite3
from typing import List, Dict

from .display import get_exchange_for_display
from .segment import get_all_topics, get_cached_segment_nodes


def format_answer_response(results: List[Dict]) -> str:
    """
    Format answer mode response.
    
    Per spec 12.2:
    - Empty retrieval: "I don't have that in our conversation history."
    - Non-empty: Return results for LLM synthesis
    """
    if not results:
        return "I don't have that in our conversation history."
    
    # Non-empty: return excerpts (actual LLM call happens at higher level)
    return None  # Indicates results available for synthesis


def format_summarize_response(results: List[Dict]) -> str:
    """
    Format summarize mode response.
    
    Per spec 12.3:
    - Empty retrieval: "No conversations found to summarize."
    - Non-empty: Return results for LLM summarization
    """
    if not results:
        return "No conversations found to summarize."
    
    return None  # Indicates results available for summarization


def format_browse_response(
    conn: sqlite3.Connection,
    results: List[Dict]
) -> Dict:
    """
    Format browse mode response with segment grouping.
    
    Per spec 12.1:
    - Display full exchange even with speaker scope
    - Group by segment
    """
    if not results:
        return {'groups': []}
    
    # Build segment membership map
    topics = get_all_topics(conn)
    node_to_segment = {}
    
    for topic in topics:
        nodes_list, nodes_set = get_cached_segment_nodes(conn, topic['id'])
        for node_id in nodes_set:
            if node_id not in node_to_segment:
                node_to_segment[node_id] = topic['name']
    
    # Group results
    groups = {}
    for r in results:
        exchange_id = r.get('exchange_id')
        segment_name = node_to_segment.get(exchange_id, 'Other')
        
        if segment_name not in groups:
            groups[segment_name] = []
        
        exchange = get_exchange_for_display(conn, exchange_id, r.get('metadata'))
        groups[segment_name].append(exchange)
    
    return {
        'groups': [
            {'segment': name, 'exchanges': exchanges}
            for name, exchanges in groups.items()
        ]
    }
