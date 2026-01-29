"""
Exchange-to-topic promotion.

Promotes exchange hits to topic membership using SQLite as authoritative source.
Implements deterministic first-match-wins by topic id ASC.
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from episodic.retrieval.segment import get_cached_segment_nodes, get_all_topics

logger = logging.getLogger(__name__)


@dataclass
class PromotedHit:
    """An exchange hit with its topic assignment."""
    exchange_id: str
    similarity: float
    topic_id: Optional[int]  # None if no topic membership found
    metadata: Dict = field(default_factory=dict)


@dataclass
class PromotionResult:
    """Result of promoting exchange hits to topics."""
    # Hits grouped by topic_id (None key = unassigned)
    by_topic: Dict[Optional[int], List[PromotedHit]]
    # Topic metadata for assigned topics
    topic_info: Dict[int, Dict]  # topic_id -> {name, start_node_id, end_node_id, ...}
    # Audit log entries
    audit_entries: List[str]


def promote_hits_to_topics(
    conn: sqlite3.Connection,
    hits: List[Dict],
    similarity_key: str = 'similarity'
) -> PromotionResult:
    """
    Promote exchange hits to topic membership.
    
    Args:
        conn: SQLite connection
        hits: List of hit dicts with exchange_id and similarity score
        similarity_key: Key for similarity score in hit dict
            (may be 'similarity', 'distance', 'relevance_score', etc.)
    
    Returns:
        PromotionResult with hits grouped by topic_id
    
    Determinism:
        - Topics iterated in id ASC order
        - First matching topic wins
        - Overlaps logged to audit
    """
    audit_entries = []
    
    # Extract exchange_ids from hits
    exchange_ids = set()
    hit_lookup = {}  # exchange_id -> hit dict
    for hit in hits:
        eid = _extract_exchange_id(hit)
        if eid:
            exchange_ids.add(eid)
            hit_lookup[eid] = hit
    
    if not exchange_ids:
        return PromotionResult(by_topic={}, topic_info={}, audit_entries=audit_entries)
    
    # Get all topics in id ASC order
    topics = get_all_topics(conn)
    topic_info = {t['id']: t for t in topics}
    
    # Build exchange_id -> topic_id mapping
    # First match wins (topics in id ASC)
    exchange_to_topic: Dict[str, Optional[int]] = {eid: None for eid in exchange_ids}
    
    for topic in topics:
        topic_id = topic['id']
        _, nodes_set = get_cached_segment_nodes(conn, topic_id)
        
        if not nodes_set:
            continue
        
        for eid in exchange_ids:
            if eid in nodes_set:
                if exchange_to_topic[eid] is None:
                    # First assignment
                    exchange_to_topic[eid] = topic_id
                else:
                    # Overlap detected - already assigned to another topic
                    prev_topic = exchange_to_topic[eid]
                    audit_entries.append(
                        f"OVERLAP: exchange_id={eid} matches topic_id={prev_topic} (kept) "
                        f"and topic_id={topic_id} (discarded)"
                    )
                    logger.debug(audit_entries[-1])
    
    # Group hits by topic_id
    by_topic: Dict[Optional[int], List[PromotedHit]] = {}
    
    for eid, hit in hit_lookup.items():
        topic_id = exchange_to_topic.get(eid)
        
        # Extract similarity score
        sim = _extract_similarity(hit, similarity_key)
        
        promoted = PromotedHit(
            exchange_id=eid,
            similarity=sim,
            topic_id=topic_id,
            metadata=hit.get('metadata', {})
        )
        
        if topic_id not in by_topic:
            by_topic[topic_id] = []
        by_topic[topic_id].append(promoted)
    
    # Sort hits within each topic by similarity descending
    for topic_id in by_topic:
        by_topic[topic_id].sort(key=lambda h: -h.similarity)
    
    return PromotionResult(
        by_topic=by_topic,
        topic_info=topic_info,
        audit_entries=audit_entries
    )


def _extract_exchange_id(hit: Dict) -> Optional[str]:
    """Extract exchange_id from various hit formats."""
    # Try common keys
    for key in ('exchange_id', 'user_id', 'id', 'node_id'):
        if key in hit:
            return hit[key]
    
    # Try nested in metadata
    metadata = hit.get('metadata', {})
    for key in ('exchange_id', 'user_id', 'id', 'node_id'):
        if key in metadata:
            return metadata[key]
    
    return None


def _extract_similarity(hit: Dict, similarity_key: str) -> float:
    """Extract similarity score from hit dict."""
    # Direct key
    if similarity_key in hit:
        val = hit[similarity_key]
        # Handle distance (lower is better) vs similarity (higher is better)
        if similarity_key == 'distance':
            return max(0.0, 1.0 - val) if val is not None else 0.0
        return float(val) if val is not None else 0.0
    
    # Common alternatives
    for key in ('similarity', 'relevance_score', 'score', 'distance'):
        if key in hit:
            val = hit[key]
            if key == 'distance':
                return max(0.0, 1.0 - val) if val is not None else 0.0
            return float(val) if val is not None else 0.0
    
    return 0.0


def get_unassigned_hits(result: PromotionResult) -> List[PromotedHit]:
    """Get hits that didn't match any topic (statement-only candidates)."""
    return result.by_topic.get(None, [])


def get_topic_hits(result: PromotionResult, topic_id: int) -> List[PromotedHit]:
    """Get hits for a specific topic."""
    return result.by_topic.get(topic_id, [])


def get_assigned_topic_ids(result: PromotionResult) -> List[int]:
    """Get list of topic_ids that have at least one hit, sorted by id ASC."""
    return sorted([tid for tid in result.by_topic.keys() if tid is not None])
