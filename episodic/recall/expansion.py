"""
Expansion tiers for topic recall.

Expands topics around anchor hits with configurable window sizes.
Handles both ongoing (no summary) and compressed (has summary) topics.
"""

import sqlite3
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from episodic.retrieval.segment import get_cached_segment_nodes

from .ranking import RankedTopic


class Tier(Enum):
    """Expansion tier levels."""
    A = auto()  # Minimal: anchors only (ongoing) or summary only (compressed)
    B = auto()  # Medium: anchors ± small window or summary + anchors
    C = auto()  # Full: diverse anchors ± large window, merged


@dataclass
class ExpandedExchange:
    """A single expanded exchange."""
    node_id: str
    content: str
    role: str  # 'user' or 'assistant'
    is_anchor: bool  # True if this was a hit anchor
    position: int  # Position in topic's node list


@dataclass 
class TopicExpansion:
    """Expanded content for a topic."""
    topic_id: int
    topic_name: str
    is_compressed: bool
    summary: Optional[str]  # Only for compressed topics
    exchanges: List[ExpandedExchange]
    anchor_count: int
    total_exchanges: int
    tier: Tier


@dataclass
class ExpansionConfig:
    """Configuration for expansion behavior."""
    # Anchor selection
    anchor_diversity_distance: int = 2  # Skip candidates within ±d of selected anchors
    
    # Tier A (ongoing): anchors only
    tier_a_max_anchors: int = 2
    tier_a_window: int = 0
    
    # Tier B (ongoing): anchors ± small window
    tier_b_max_anchors: int = 2
    tier_b_window: int = 1
    tier_b_max_exchanges: int = 6
    
    # Tier C (ongoing): diverse anchors ± large window
    tier_c_max_anchors: int = 3
    tier_c_window: int = 3
    tier_c_max_exchanges: int = 12
    
    # Compressed topics
    compressed_tier_a_max_anchors: int = 0  # Summary only
    compressed_tier_b_max_anchors: int = 2
    compressed_tier_b_max_exchanges: int = 6
    compressed_tier_c_max_anchors: int = 3
    compressed_tier_c_window: int = 2
    compressed_tier_c_max_exchanges: int = 12


DEFAULT_CONFIG = ExpansionConfig()


def expand_topic(
    conn: sqlite3.Connection,
    ranked_topic: RankedTopic,
    tier: Tier,
    config: ExpansionConfig = DEFAULT_CONFIG
) -> TopicExpansion:
    """
    Expand a topic to exchanges based on tier.
    
    Args:
        conn: SQLite connection
        ranked_topic: Topic with hits to expand
        tier: Expansion tier (A, B, or C)
        config: Expansion configuration
    
    Returns:
        TopicExpansion with exchanges and metadata
    """
    topic_id = ranked_topic.topic_id
    topic_info = ranked_topic.topic_info
    topic_name = topic_info.get('name', f'topic-{topic_id}')
    
    # Check if compressed
    is_compressed = _is_topic_compressed(conn, topic_id)
    summary = _get_topic_summary(conn, topic_id) if is_compressed else None
    
    # Get topic's node list in order
    nodes_list, nodes_set = get_cached_segment_nodes(conn, topic_id)
    
    if not nodes_list:
        return TopicExpansion(
            topic_id=topic_id,
            topic_name=topic_name,
            is_compressed=is_compressed,
            summary=summary,
            exchanges=[],
            anchor_count=0,
            total_exchanges=0,
            tier=tier
        )
    
    # Build position map for anchor selection
    position_map = {nid: i for i, nid in enumerate(nodes_list)}
    
    # Get anchor positions from hits
    anchor_positions = []
    for hit in ranked_topic.hits:
        pos = position_map.get(hit.exchange_id)
        if pos is not None:
            anchor_positions.append((pos, hit.exchange_id, hit.similarity))
    
    # Sort by similarity descending for selection
    anchor_positions.sort(key=lambda x: -x[2])
    
    # Select diverse anchors
    if is_compressed:
        selected_anchors = _select_diverse_anchors(
            anchor_positions,
            config.anchor_diversity_distance,
            _get_max_anchors_compressed(tier, config)
        )
    else:
        selected_anchors = _select_diverse_anchors(
            anchor_positions,
            config.anchor_diversity_distance,
            _get_max_anchors_ongoing(tier, config)
        )
    
    # Expand around anchors
    window = _get_window_size(tier, is_compressed, config)
    max_exchanges = _get_max_exchanges(tier, is_compressed, config)
    
    expanded_positions = _expand_around_anchors(
        selected_anchors,
        window,
        max_exchanges,
        len(nodes_list)
    )
    
    # Fetch exchange content
    anchor_set = {pos for pos, _, _ in selected_anchors}
    exchanges = []
    
    for pos in sorted(expanded_positions):
        node_id = nodes_list[pos]
        node = _get_node_from_conn(conn, node_id)
        if node:
            exchanges.append(ExpandedExchange(
                node_id=node_id,
                content=node.get('content', ''),
                role=node.get('role', 'unknown'),
                is_anchor=pos in anchor_set,
                position=pos
            ))
    
    return TopicExpansion(
        topic_id=topic_id,
        topic_name=topic_name,
        is_compressed=is_compressed,
        summary=summary,
        exchanges=exchanges,
        anchor_count=len(selected_anchors),
        total_exchanges=len(exchanges),
        tier=tier
    )


def _select_diverse_anchors(
    candidates: List[Tuple[int, str, float]],  # (position, exchange_id, similarity)
    diversity_distance: int,
    max_anchors: int
) -> List[Tuple[int, str, float]]:
    """
    Select diverse anchors, skipping candidates within ±d of already-selected.
    
    Candidates should be pre-sorted by similarity descending.
    """
    if max_anchors <= 0:
        return []
    
    selected = []
    selected_positions = set()
    
    for pos, eid, sim in candidates:
        # Check if too close to any selected anchor
        too_close = any(
            abs(pos - sel_pos) <= diversity_distance
            for sel_pos in selected_positions
        )
        
        if not too_close:
            selected.append((pos, eid, sim))
            selected_positions.add(pos)
            
            if len(selected) >= max_anchors:
                break
    
    return selected


def _expand_around_anchors(
    anchors: List[Tuple[int, str, float]],
    window: int,
    max_exchanges: int,
    topic_length: int
) -> set:
    """
    Expand window around each anchor, merge overlaps, cap to max.
    
    Returns set of positions to include.
    """
    if not anchors:
        return set()
    
    # Collect all positions within window of each anchor
    positions = set()
    for pos, _, _ in anchors:
        start = max(0, pos - window)
        end = min(topic_length, pos + window + 1)
        for p in range(start, end):
            positions.add(p)
    
    # If over budget, prioritize anchors and their immediate neighbors
    if len(positions) > max_exchanges:
        anchor_positions = {pos for pos, _, _ in anchors}
        
        # Start with anchors
        result = set(anchor_positions)
        
        # Add neighbors in order of distance from anchors
        for dist in range(1, window + 1):
            if len(result) >= max_exchanges:
                break
            for anchor_pos, _, _ in anchors:
                for offset in [-dist, dist]:
                    neighbor = anchor_pos + offset
                    if 0 <= neighbor < topic_length and len(result) < max_exchanges:
                        result.add(neighbor)
        
        return result
    
    return positions


def _is_topic_compressed(conn: sqlite3.Connection, topic_id: int) -> bool:
    """Check if topic has a compression summary."""
    # Check compressions_v2 table or similar
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 1 FROM compressions c
            JOIN topics t ON c.original_branch_head = t.end_node_id
            WHERE t.id = ?
            LIMIT 1
        """, (topic_id,))
        return cursor.fetchone() is not None
    except:
        return False


def _get_topic_summary(conn: sqlite3.Connection, topic_id: int) -> Optional[str]:
    """Get compression summary for a topic."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT n.content FROM nodes n
            JOIN compressions c ON n.id = c.compressed_node_id
            JOIN topics t ON c.original_branch_head = t.end_node_id
            WHERE t.id = ?
            LIMIT 1
        """, (topic_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    except:
        return None


def _get_max_anchors_ongoing(tier: Tier, config: ExpansionConfig) -> int:
    """Get max anchors for ongoing topic by tier."""
    if tier == Tier.A:
        return config.tier_a_max_anchors
    elif tier == Tier.B:
        return config.tier_b_max_anchors
    else:  # Tier.C
        return config.tier_c_max_anchors


def _get_max_anchors_compressed(tier: Tier, config: ExpansionConfig) -> int:
    """Get max anchors for compressed topic by tier."""
    if tier == Tier.A:
        return config.compressed_tier_a_max_anchors
    elif tier == Tier.B:
        return config.compressed_tier_b_max_anchors
    else:  # Tier.C
        return config.compressed_tier_c_max_anchors


def _get_window_size(tier: Tier, is_compressed: bool, config: ExpansionConfig) -> int:
    """Get window size by tier and compression state."""
    if is_compressed:
        if tier == Tier.C:
            return config.compressed_tier_c_window
        return 0  # Compressed A/B don't expand beyond anchors
    else:
        if tier == Tier.A:
            return config.tier_a_window
        elif tier == Tier.B:
            return config.tier_b_window
        else:  # Tier.C
            return config.tier_c_window


def _get_max_exchanges(tier: Tier, is_compressed: bool, config: ExpansionConfig) -> int:
    """Get max exchanges by tier and compression state."""
    if is_compressed:
        if tier == Tier.A:
            return 0  # Summary only
        elif tier == Tier.B:
            return config.compressed_tier_b_max_exchanges
        else:  # Tier.C
            return config.compressed_tier_c_max_exchanges
    else:
        if tier == Tier.A:
            return config.tier_a_max_anchors * 2  # Just anchors
        elif tier == Tier.B:
            return config.tier_b_max_exchanges
        else:  # Tier.C
            return config.tier_c_max_exchanges


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
