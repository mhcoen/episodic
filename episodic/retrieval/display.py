"""
Exchange display pairing.

Implements v1.1 spec section 5.3.
"""
import sqlite3
import logging
from typing import Dict, Optional

from .segment import get_node, get_head, build_ancestry_map

logger = logging.getLogger(__name__)


def get_exchange_for_display(
    conn: sqlite3.Connection,
    exchange_id: str,
    metadata: Optional[Dict]
) -> Dict:
    """
    Get full exchange for display with correct assistant pairing.
    
    Per spec 5.3:
    1. If metadata.assistant_id exists and valid, use it
    2. Otherwise fallback: earliest assistant child on ancestry, else earliest overall
    
    Returns:
        Dict with exchange_id, user_content, assistant_id, assistant_content
    """
    user_node = get_node(conn, exchange_id)
    if not user_node:
        return {'exchange_id': exchange_id, 'user_content': None, 'assistant_id': None}
    
    result = {
        'exchange_id': exchange_id,
        'user_content': user_node.get('content'),
        'assistant_id': None,
        'assistant_content': None,
    }
    
    # Try metadata.assistant_id first
    if metadata and metadata.get('assistant_id'):
        candidate_id = metadata['assistant_id']
        candidate = get_node(conn, candidate_id)
        
        if candidate:
            # Validate: must be assistant and parent must be exchange_id
            if (candidate.get('role') == 'assistant' and 
                candidate.get('parent_id') == exchange_id):
                result['assistant_id'] = candidate_id
                result['assistant_content'] = candidate.get('content')
                return result
            else:
                logger.debug(f"AUDIT: Invalid assistant_id {candidate_id} for exchange {exchange_id}")
    
    # Fallback: find assistant children
    assistant = _find_fallback_assistant(conn, exchange_id)
    if assistant:
        result['assistant_id'] = assistant['id']
        result['assistant_content'] = assistant.get('content')
    
    return result


def _find_fallback_assistant(
    conn: sqlite3.Connection,
    user_node_id: str
) -> Optional[Dict]:
    """
    Find fallback assistant for user node.
    
    Prefers assistant on current head ancestry, else earliest by created_at.
    """
    cursor = conn.cursor()
    
    # Get all assistant children
    cursor.execute("""
        SELECT * FROM nodes 
        WHERE parent_id = ? AND role = 'assistant'
        ORDER BY created_at ASC
    """, (user_node_id,))
    
    children = [dict(row) for row in cursor.fetchall()]
    if not children:
        return None
    
    if len(children) == 1:
        return children[0]
    
    # Multiple children - prefer one on ancestry
    head_id = get_head(conn)
    if head_id:
        ancestry_map = build_ancestry_map(conn, head_id)
        ancestry_set = set(ancestry_map.keys())
        
        for child in children:
            if child['id'] in ancestry_set:
                return child
    
    # No child on ancestry, return earliest
    return children[0]
