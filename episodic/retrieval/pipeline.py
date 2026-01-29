"""
End-to-end retrieval pipeline.

Implements v1.1 spec section 10.
"""
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Set

from .segment_filter import SegmentFilter, FilterKind, build_segment_filter
from .segment import get_cached_segment_nodes
from .lexical import execute_lexical_search, get_recent_exchanges, node_to_exchange_id
from .semantic import execute_semantic_search, filter_semantic_by_segment, filter_semantic_by_temporal
from .fusion import fuse_results

logger = logging.getLogger(__name__)


def retrieve(
    conn: sqlite3.Connection,
    chroma,
    target: str,
    segment_scope: Optional[List[str]],
    temporal: Optional[Tuple[str, str]],
    speaker: Optional[str],
    mode: str,
    max_results: int,
    config: dict
) -> List[Dict]:
    """
    End-to-end retrieval pipeline.
    
    Args:
        conn: Database connection
        chroma: Chroma collection or FakeChroma
        target: Search query text
        segment_scope: Tri-state segment scope (None, [], or [ids])
        temporal: Optional (start_utc, end_utc) half-open interval
        speaker: Optional role filter ('user' or 'assistant')
        mode: 'answer', 'browse', or 'summarize'
        max_results: Maximum results to return
        config: Configuration dict
    
    Returns:
        List of result dicts with exchange_id and metadata
    """
    # Build segment filter from tri-state
    segment_filter = build_segment_filter(segment_scope)
    
    # Fail-safe: empty scope returns empty
    if segment_filter.kind == FilterKind.EMPTY:
        return []
    
    # Handle empty target per spec 8.3
    if not target or not target.strip():
        if mode == 'browse':
            return get_recent_exchanges(conn, max_results, segment_filter, temporal)
        else:
            return []  # answer/summarize cannot proceed without target
    
    # Get segment set for semantic filtering
    segment_set: Optional[Set[str]] = None
    if segment_scope is not None and segment_scope:
        segment_set = set(segment_scope)
    
    # Weights
    w_sem = config.get('semantic_weight', 0.6)
    w_lex = config.get('lexical_weight', 0.4)
    over_fetch = config.get('over_fetch_multiplier', 3)
    fetch_limit = max_results * over_fetch
    
    # Speaker routing per spec 10.2
    if speaker:
        # Disable semantic, lexical only with role filter
        lexical_results = execute_lexical_search(
            conn=conn,
            target=target,
            segment_filter=segment_filter,
            speaker=speaker,
            temporal=temporal,
            limit=fetch_limit,
            config=config
        )
        
        # Convert to exchange format
        results = _lexical_to_exchanges(lexical_results)
        return results[:max_results]
    
    # Normal path: both channels
    
    # Lexical retrieval
    lexical_results = execute_lexical_search(
        conn=conn,
        target=target,
        segment_filter=segment_filter,
        speaker=None,
        temporal=temporal,
        limit=fetch_limit,
        config=config
    )
    lexical_exchanges = _lexical_to_exchanges(lexical_results)
    
    # Semantic retrieval
    semantic_results = execute_semantic_search(chroma, target, fetch_limit)
    
    # Apply semantic filters
    if segment_set:
        semantic_results = filter_semantic_by_segment(semantic_results, segment_set)
    
    if temporal:
        start_str, end_str = temporal
        start_utc = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end_utc = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        semantic_results = filter_semantic_by_temporal(semantic_results, start_utc, end_utc)
    
    # Fusion
    fused = fuse_results(semantic_results, lexical_exchanges, w_sem, w_lex)
    
    return fused[:max_results]


def _lexical_to_exchanges(lexical_results: List[Dict]) -> List[Dict]:
    """Convert lexical results to exchange format with bm25_score."""
    exchanges = []
    seen = set()
    
    for r in lexical_results:
        exchange_id = node_to_exchange_id(r, r.get('parent_role'))
        if exchange_id and exchange_id not in seen:
            seen.add(exchange_id)
            exchanges.append({
                'exchange_id': exchange_id,
                'bm25_score': r.get('bm25_score', 0),
                'metadata': {}
            })
    
    return exchanges
