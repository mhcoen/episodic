"""
Topic ranking by concentration.

Ranks topics by hit quality, not length. Uses best hit + top-k mass + weak count term.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from .promotion import PromotedHit, PromotionResult


@dataclass
class RankedTopic:
    """A topic with its ranking score and evidence."""
    topic_id: int
    score: float
    best_hit: float  # Highest similarity among hits
    top_k_mass: float  # Sum of top-k similarities
    hit_count: int
    hits: List[PromotedHit]
    topic_info: dict  # name, start_node_id, etc.


@dataclass
class RankingResult:
    """Result of topic ranking."""
    ranked_topics: List[RankedTopic]  # Sorted by score descending
    unassigned_hits: List[PromotedHit]  # Hits with no topic (statement candidates)


def rank_topics(
    promotion_result: PromotionResult,
    w_best: float = 0.5,
    w_mass: float = 0.3,
    w_count: float = 0.2,
    top_k: int = 3
) -> RankingResult:
    """
    Rank topics by hit concentration.
    
    Score = w_best * best_hit + w_mass * top_k_mass + w_count * log(1 + count)
    
    Args:
        promotion_result: Result from promote_hits_to_topics
        w_best: Weight for best hit similarity
        w_mass: Weight for top-k hit mass
        w_count: Weight for log count term
        top_k: Number of top hits to sum for mass
    
    Returns:
        RankingResult with topics sorted by score descending
    """
    ranked_topics = []
    
    for topic_id, hits in promotion_result.by_topic.items():
        if topic_id is None:
            # Skip unassigned - handled separately
            continue
        
        if not hits:
            continue
        
        # Hits are already sorted by similarity descending (from promotion)
        best_hit = hits[0].similarity
        
        # Top-k mass
        top_k_hits = hits[:top_k]
        top_k_mass = sum(h.similarity for h in top_k_hits)
        
        # Score
        hit_count = len(hits)
        score = (
            w_best * best_hit +
            w_mass * top_k_mass +
            w_count * math.log(1 + hit_count)
        )
        
        # Get topic info
        topic_info = promotion_result.topic_info.get(topic_id, {})
        
        ranked_topics.append(RankedTopic(
            topic_id=topic_id,
            score=score,
            best_hit=best_hit,
            top_k_mass=top_k_mass,
            hit_count=hit_count,
            hits=hits,
            topic_info=topic_info
        ))
    
    # Sort by score descending, then topic_id ascending for determinism
    ranked_topics.sort(key=lambda t: (-t.score, t.topic_id))
    
    # Get unassigned hits
    unassigned_hits = promotion_result.by_topic.get(None, [])
    
    return RankingResult(
        ranked_topics=ranked_topics,
        unassigned_hits=unassigned_hits
    )


def get_top_topics(ranking_result: RankingResult, n: int = 2) -> List[RankedTopic]:
    """Get top N ranked topics."""
    return ranking_result.ranked_topics[:n]


def get_top_statements(
    ranking_result: RankingResult,
    n: int = 3,
    exclude_topic_ids: Optional[List[int]] = None
) -> List[PromotedHit]:
    """
    Get top N statement candidates.
    
    Returns unassigned hits plus optionally hits from non-top topics,
    sorted by similarity descending.
    
    Args:
        ranking_result: Result from rank_topics
        n: Maximum statements to return
        exclude_topic_ids: Topic IDs whose hits should not be included
            (typically the top topics being expanded as conversation blocks)
    
    Returns:
        List of PromotedHit sorted by similarity descending
    """
    if exclude_topic_ids is None:
        exclude_topic_ids = []
    
    exclude_set = set(exclude_topic_ids)
    
    # Start with unassigned hits
    candidates = list(ranking_result.unassigned_hits)
    
    # Add hits from non-excluded topics (spillover)
    for topic in ranking_result.ranked_topics:
        if topic.topic_id not in exclude_set:
            candidates.extend(topic.hits)
    
    # Sort by similarity descending, then exchange_id for determinism
    candidates.sort(key=lambda h: (-h.similarity, h.exchange_id))
    
    return candidates[:n]
