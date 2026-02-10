"""
Semantic retrieval using Chroma.

Implements v1.1 spec section 9.
"""
import logging
from datetime import datetime, timezone
from typing import List, Dict, Set

logger = logging.getLogger(__name__)


def adapt_chroma_results(raw: Dict) -> List[Dict]:
    """
    Convert Chroma response to List[Dict].
    
    Strict drop on missing distance per spec 9.1.
    
    Returns:
        List of dicts with exchange_id, distance, metadata
    """
    if not raw or not raw.get('ids') or not raw['ids'][0]:
        return []
    
    ids = raw['ids'][0]
    metadatas = raw.get('metadatas', [[]])[0]
    distances = raw.get('distances', [[]])[0]
    
    results = []
    for i, doc_id in enumerate(ids):
        if i >= len(distances) or distances[i] is None:
            logger.debug(f"AUDIT: Dropping Chroma result {doc_id} - missing distance")
            continue
        
        metadata = metadatas[i] if i < len(metadatas) else {}
        
        results.append({
            'exchange_id': doc_id,
            'distance': distances[i],
            'metadata': metadata
        })
    
    return results


def filter_semantic_by_segment(
    results: List[Dict],
    segment_set: Set[str]
) -> List[Dict]:
    """Filter semantic results by segment membership using exchange_id."""
    return [r for r in results if r['exchange_id'] in segment_set]


def filter_semantic_by_temporal(
    results: List[Dict],
    start_utc: datetime,
    end_utc: datetime
) -> List[Dict]:
    """
    Filter by temporal range. Strict drop on missing/unparseable timestamp.
    
    Uses half-open interval: start_utc <= ts < end_utc
    """
    filtered = []
    for r in results:
        ts_str = r.get('metadata', {}).get('timestamp')
        if not ts_str:
            logger.debug(f"AUDIT: Dropping {r['exchange_id']} - missing timestamp")
            continue
        
        try:
            # Handle Z suffix and various ISO formats
            ts_str_clean = ts_str.replace('Z', '+00:00')
            ts = datetime.fromisoformat(ts_str_clean)
            # Ensure timezone aware
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError) as e:
            logger.debug(f"AUDIT: Dropping {r['exchange_id']} - unparseable timestamp: {e}")
            continue
        
        if start_utc <= ts < end_utc:
            filtered.append(r)
    
    return filtered


def execute_semantic_search(
    chroma,
    target: str,
    n_results: int
) -> List[Dict]:
    """
    Execute Chroma semantic search.
    
    Args:
        chroma: Chroma collection or FakeChroma
        target: Query text
        n_results: Number of results to fetch
    
    Returns:
        Adapted results list
    """
    if not target or not target.strip():
        return []
    
    raw = chroma.query(query_texts=[target], n_results=n_results)
    return adapt_chroma_results(raw)
