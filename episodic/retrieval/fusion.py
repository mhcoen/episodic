"""
Score normalization and fusion.

Implements v1.1 spec section 10.4-10.6.
"""
from typing import List, Dict, Tuple


def prepare_for_fusion(
    semantic: List[Dict],
    lexical: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Sort candidates deterministically before normalization.
    
    Per spec 10.5:
    - Semantic: (distance ASC, exchange_id ASC)
    - Lexical: (bm25_score DESC, exchange_id ASC)
    """
    semantic_sorted = sorted(
        semantic,
        key=lambda r: (r.get('distance', float('inf')), r.get('exchange_id', ''))
    )
    lexical_sorted = sorted(
        lexical,
        key=lambda r: (-r.get('bm25_score', 0), r.get('exchange_id', ''))
    )
    return semantic_sorted, lexical_sorted


def normalize_scores(
    results: List[Dict],
    score_key: str,
    invert: bool
) -> Dict[str, float]:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        results: List of result dicts
        score_key: Key to extract score from
        invert: If True, lower raw = higher normalized
    
    Returns:
        Dict mapping exchange_id -> normalized score
    """
    if not results:
        return {}
    
    scores = [(r.get('exchange_id'), r.get(score_key, 0)) for r in results]
    values = [s[1] for s in scores]
    min_v, max_v = min(values), max(values)
    
    normalized = {}
    for doc_id, score in scores:
        if max_v == min_v:
            # Single value or all equal: give full score (1.0), not partial (0.5)
            # Per spec: the item IS the best (and worst), so it should be treated as best
            norm = 1.0
        else:
            norm = (score - min_v) / (max_v - min_v)
            if invert:
                norm = 1.0 - norm
        normalized[doc_id] = norm

    return normalized


def fuse_results(
    semantic: List[Dict],
    lexical: List[Dict],
    w_sem: float,
    w_lex: float
) -> List[Dict]:
    """
    Fuse semantic and lexical results.
    
    Per spec 10.6:
    - Semantic: invert=True (lower distance = higher norm)
    - Lexical: invert=False (higher bm25 = higher norm)
    - Missing channel: norm = 0.0
    - Final sort: (final_score DESC, exchange_id ASC)
    """
    semantic, lexical = prepare_for_fusion(semantic, lexical)
    
    sem_by_id = {r['exchange_id']: r for r in semantic}
    lex_by_id = {r['exchange_id']: r for r in lexical}
    
    sem_scores = normalize_scores(semantic, 'distance', invert=True)
    lex_scores = normalize_scores(lexical, 'bm25_score', invert=False)
    
    all_ids = set(sem_scores.keys()) | set(lex_scores.keys())
    
    fused = []
    for doc_id in all_ids:
        norm_sem = sem_scores.get(doc_id, 0.0)
        norm_lex = lex_scores.get(doc_id, 0.0)
        final_score = w_sem * norm_sem + w_lex * norm_lex
        
        # Get full result from whichever channel has it
        result = sem_by_id.get(doc_id) or lex_by_id.get(doc_id)
        result = dict(result)  # Copy to avoid mutation
        result['final_score'] = final_score
        result['exchange_id'] = doc_id
        fused.append(result)
    
    # Sort by final_score DESC, exchange_id ASC for stability
    fused.sort(key=lambda x: (-x['final_score'], x['exchange_id']))
    return fused
