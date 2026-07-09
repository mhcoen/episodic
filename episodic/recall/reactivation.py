"""
Implicit topic reactivation probe.

Detects when a user message relates to a previously inactive topic
and returns a decision about whether to reactivate that topic.

Uses two-channel matching:
  Channel A: Semantic similarity (user input ↔ topic centroid)
  Channel B: Alias matching (distinctive topic terms in referential queries)
"""

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from episodic.config import config
from episodic.db_connection import get_connection
from episodic.recall.resume_cues import has_resume_cues
from episodic.recall.topic_aliases import compute_alias_score, get_topic_aliases_batch

logger = logging.getLogger(__name__)

from episodic.recall.reactivation_helpers import (  # noqa: F401  (re-exported)
    K_TOPICS, M_EXCHANGES, S_SUPPORT, S_SUPPORT_ALIAS, DELTA_BAND,
    COOLDOWN_TURNS, DORMANCY_MIN, SIM_THRESHOLD_NORMAL, SIM_THRESHOLD_RESUME_CUE,
    ALIAS_HITS_MIN,
    _get_current_turn_idx, _get_topic_info, _get_dormant_topic_centroids,
    _compute_similarity, _get_embeddings_for_nodes, _get_topic_exchanges,
    _check_support, _get_topic_preview, _get_topic_snippets, _get_topic_summary,
    _get_recent_exchanges, _get_anchor_exchange, _truncate_to_budget,
)


@dataclass
class DisambiguationOption:
    """Option for disambiguation when multiple topics match."""
    topic_name: str
    topic_start_node_id: str
    similarity: float
    support_count: int
    preview: str = ""  # Short preview of topic content
    turns_ago: int = 0  # How many turns since last activity
    snippets: List[str] = field(default_factory=list)  # Evidence snippets


@dataclass
class ReactivationDecision:
    """Result of reactivation probe."""
    action: Literal["CONTINUE", "REACTIVATE", "DISAMBIGUATE"]
    topic_name: Optional[str] = None
    topic_start_node_id: Optional[str] = None
    options: Optional[List[DisambiguationOption]] = None
    debug: Dict[str, Any] = field(default_factory=dict)


def assemble_reactivation_packet(
    topic_start_node_id: str,
    user_embedding: np.ndarray,
    token_budget: int = 150,
    conn: Optional[sqlite3.Connection] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Assemble a context packet for topic reactivation.

    Rules:
    - If topic compressed: summary + 1 anchor exchange
    - If not compressed: last 2 exchanges from topic

    Args:
        topic_start_node_id: Start node ID of the topic to reactivate
        user_embedding: Embedding of current user input (for anchor selection)
        token_budget: Maximum tokens for the packet
        conn: Optional database connection

    Returns:
        Tuple of (packet_text, debug_info)
    """
    debug_info: Dict[str, Any] = {
        'topic_start_node_id': topic_start_node_id,
        'token_budget': token_budget,
    }

    def _assemble(c: sqlite3.Connection) -> Tuple[str, Dict[str, Any]]:
        # Get topic info
        topic_info = _get_topic_info(c, topic_start_node_id)
        if not topic_info:
            debug_info['error'] = 'topic_not_found'
            return "", debug_info

        topic_name = topic_info['name']
        end_node_id = topic_info.get('end_node_id')
        debug_info['topic_name'] = topic_name

        # Check if topic has a compression summary
        summary = _get_topic_summary(c, topic_start_node_id)

        if summary:
            debug_info['has_summary'] = True
            # Use summary + 1 anchor exchange
            anchor = _get_anchor_exchange(c, topic_start_node_id, end_node_id, user_embedding)

            parts = [f"[Previous context from '{topic_name}':]"]
            parts.append(summary)

            if anchor:
                user_content, asst_content = anchor
                parts.append("")
                parts.append(f"User: {user_content}")
                parts.append(f"Assistant: {asst_content}")
                debug_info['anchor_included'] = True

            packet = "\n".join(parts)

        else:
            debug_info['has_summary'] = False
            # Use last 2 exchanges
            exchanges = _get_recent_exchanges(c, topic_start_node_id, end_node_id, limit=2)
            debug_info['exchanges_count'] = len(exchanges)

            if not exchanges:
                debug_info['error'] = 'no_exchanges_found'
                return "", debug_info

            parts = [f"[Previous context from '{topic_name}':]"]
            for user_content, asst_content in exchanges:
                parts.append("")
                parts.append(f"User: {user_content}")
                parts.append(f"Assistant: {asst_content}")

            packet = "\n".join(parts)

        # Truncate to budget
        packet = _truncate_to_budget(packet, token_budget)
        debug_info['packet_length'] = len(packet)

        return packet, debug_info

    if conn is not None:
        return _assemble(conn)

    with get_connection() as c:
        return _assemble(c)


def probe_reactivation(
    user_input: str,
    user_embedding: np.ndarray,
    active_topic_start_node_id: Optional[str],
    cooldown_turns: int,
    now: datetime,
    recent_nodes: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None
) -> ReactivationDecision:
    """
    Probe whether to reactivate a dormant topic based on user input.

    Args:
        user_input: The user's message text
        user_embedding: Pre-computed embedding for user_input
        active_topic_start_node_id: Start node ID of currently active topic (if any)
        cooldown_turns: Number of turns remaining in cooldown period
        now: Current timestamp
        recent_nodes: Recent conversation nodes for context
        conn: Optional database connection

    Returns:
        ReactivationDecision with action and optional topic info
    """
    # Detect resume cues early (affects threshold selection)
    resume_cues_detected = has_resume_cues(user_input)

    debug_info: Dict[str, Any] = {
        'cooldown_turns': cooldown_turns,
        'active_topic': active_topic_start_node_id,
        'resume_cues_detected': resume_cues_detected,
        # Feature logging for calibration
        'candidates': [],
        'best_vs_active_gap': None,
        'support_counts': {},
        'alias_scores': {},
        'gates_passed': [],
        'gates_failed': [],
        'channel_a_pass': False,
        'channel_b_pass': False,
        'confidence': 0.0,
    }

    # Early exit: cooldown active
    if cooldown_turns > 0:
        debug_info['exit_reason'] = 'cooldown_active'
        debug_info['gates_failed'].append('cooldown')
        return ReactivationDecision(action="CONTINUE", debug=debug_info)

    # Early exit: input too short
    if len(user_input.split()) < 4:
        debug_info['exit_reason'] = 'input_too_short'
        debug_info['gates_failed'].append('input_length')
        return ReactivationDecision(action="CONTINUE", debug=debug_info)

    def _probe(c: sqlite3.Connection) -> ReactivationDecision:
        current_turn_idx = _get_current_turn_idx(c)
        debug_info['current_turn_idx'] = current_turn_idx

        # Get the most-recently-active dormant topics with centroids. The
        # dormancy filter and recency bound are applied in SQL so the per-turn
        # cost is O(limit), not O(total topics).
        max_candidates = config.get('reactivation_max_candidates', 50)
        candidates = _get_dormant_topic_centroids(
            c, current_turn_idx, active_topic_start_node_id,
            DORMANCY_MIN, max_candidates,
        )
        if not candidates:
            debug_info['exit_reason'] = 'no_dormant_topics'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        debug_info['total_topics'] = len(candidates)

        # Attach dormancy for downstream ranking/debug (filters already applied
        # in SQL; recompute defensively).
        dormant_topics = []
        for topic in candidates:
            dormancy = current_turn_idx - topic['last_active_turn_idx']
            if topic['start_node_id'] == active_topic_start_node_id:
                continue
            if dormancy < DORMANCY_MIN:
                continue
            topic['dormancy'] = dormancy
            dormant_topics.append(topic)

        if not dormant_topics:
            debug_info['exit_reason'] = 'no_dormant_topics'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        debug_info['dormant_topics'] = len(dormant_topics)

        # Get centroid embeddings for dormant topics
        centroid_node_ids = [t['centroid_medoid_exchange_id'] for t in dormant_topics]
        centroid_embeddings = _get_embeddings_for_nodes(centroid_node_ids)

        if not centroid_embeddings:
            debug_info['exit_reason'] = 'no_centroid_embeddings'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        # Compute similarity to each topic centroid
        topic_similarities = []
        for topic in dormant_topics:
            centroid_id = topic['centroid_medoid_exchange_id']
            if centroid_id in centroid_embeddings:
                sim = _compute_similarity(user_embedding, centroid_embeddings[centroid_id])
                topic_similarities.append((topic, sim))

        if not topic_similarities:
            debug_info['exit_reason'] = 'no_similarities_computed'
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        # Sort by similarity descending
        topic_similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top K topics
        top_k = topic_similarities[:K_TOPICS]
        debug_info['top_k_similarities'] = [(t['name'], s) for t, s in top_k]

        # Build detailed candidates list for feature logging
        for rank, (topic, sim) in enumerate(top_k):
            debug_info['candidates'].append({
                'topic': topic['name'],
                'topic_start_node_id': topic['start_node_id'],
                'sim': sim,
                'rank': rank,
                'dormancy': topic.get('dormancy', 0),
            })

        # Two-channel gate: semantic similarity OR alias matching
        best_topic, best_sim = top_k[0]
        debug_info['best_similarity'] = best_sim
        debug_info['best_topic'] = best_topic['name']

        # Select threshold based on resume cue presence
        sim_threshold = SIM_THRESHOLD_RESUME_CUE if resume_cues_detected else SIM_THRESHOLD_NORMAL
        debug_info['sim_threshold_used'] = sim_threshold

        # Channel A: Semantic similarity check
        channel_a_pass = best_sim >= sim_threshold
        debug_info['channel_a_pass'] = channel_a_pass
        if channel_a_pass:
            debug_info['gates_passed'].append('channel_a_similarity')

        # Channel B: Alias matching (only if resume cues detected)
        channel_b_pass = False
        best_alias_score = 0

        if resume_cues_detected:
            # Compute alias scores for top candidates
            topic_ids = [t['start_node_id'] for t, _ in top_k]
            topic_aliases = get_topic_aliases_batch(topic_ids, conn=c)

            for topic, sim in top_k:
                aliases = topic_aliases.get(topic['start_node_id'], set())
                alias_score = compute_alias_score(user_input, aliases)
                debug_info['alias_scores'][topic['name']] = alias_score
                if topic['start_node_id'] == best_topic['start_node_id']:
                    best_alias_score = alias_score

            # Check if best candidate passes channel B
            if best_alias_score >= ALIAS_HITS_MIN:
                channel_b_pass = True
                debug_info['channel_b_pass'] = True
                debug_info['gates_passed'].append('channel_b_alias')

        # Must pass at least one channel
        if not channel_a_pass and not channel_b_pass:
            debug_info['exit_reason'] = 'neither_channel_passed'
            debug_info['gates_failed'].append('similarity_threshold')
            if resume_cues_detected:
                debug_info['gates_failed'].append('alias_threshold')
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        # Support check for best candidate (and close contenders for ambiguity)
        # First, identify candidates within rank_gap of best
        rank_gap = max(2, min(5, math.ceil(0.1 * K_TOPICS)))
        rank_gap_threshold = best_sim - 0.05  # Within 0.05 of best

        # Candidates to check: best + any within rank_gap
        candidates_to_check = [(best_topic, best_sim)]
        for topic, sim in top_k[1:]:
            if sim >= rank_gap_threshold:
                candidates_to_check.append((topic, sim))

        # Gather exchange embeddings for candidates
        exchange_ids_needed = []
        for topic, _ in candidates_to_check:
            exchanges = _get_topic_exchanges(
                c, topic['start_node_id'], topic.get('end_node_id'), M_EXCHANGES
            )
            topic['_exchange_ids'] = exchanges
            exchange_ids_needed.extend(exchanges)

        exchange_embeddings = _get_embeddings_for_nodes(list(set(exchange_ids_needed)))

        # Check support for best candidate first
        best_exchanges = best_topic.get('_exchange_ids', [])
        best_support_count, best_exchange_sim = _check_support(
            user_embedding, best_exchanges, exchange_embeddings, DELTA_BAND
        )

        debug_info['best_support_count'] = best_support_count
        debug_info['best_exchange_sim'] = best_exchange_sim
        debug_info['support_counts']['best'] = best_support_count

        # Best candidate must pass support threshold
        # Use lower threshold for channel B (alias matching) to support short topics
        required_support = S_SUPPORT_ALIAS if channel_b_pass else S_SUPPORT
        debug_info['required_support'] = required_support

        if best_support_count < required_support:
            debug_info['exit_reason'] = 'best_candidate_insufficient_support'
            debug_info['gates_failed'].append('support')
            return ReactivationDecision(action="CONTINUE", debug=debug_info)

        debug_info['gates_passed'].append('support')

        # Check if there are close contenders with support (for ambiguity)
        close_with_support = []
        for topic, centroid_sim in candidates_to_check[1:]:
            exchanges = topic.get('_exchange_ids', [])
            support_count, exchange_sim = _check_support(
                user_embedding, exchanges, exchange_embeddings, DELTA_BAND
            )
            debug_info['support_counts'][topic['name']] = support_count
            if support_count >= S_SUPPORT:
                close_with_support.append({
                    'topic': topic,
                    'centroid_sim': centroid_sim,
                    'support_count': support_count,
                    'best_exchange_sim': exchange_sim,
                })

        debug_info['close_contenders_with_support'] = len(close_with_support)
        if len(close_with_support) > 0:
            debug_info['support_counts']['second'] = close_with_support[0]['support_count'] if close_with_support else 0

        # Check for ambiguity
        if close_with_support:
            # Multiple topics match - disambiguate
            debug_info['ambiguity_detected'] = True
            debug_info['rank_gap'] = rank_gap

            options = [
                DisambiguationOption(
                    topic_name=best_topic['name'],
                    topic_start_node_id=best_topic['start_node_id'],
                    similarity=best_sim,
                    support_count=best_support_count,
                    preview=_get_topic_preview(c, best_topic['start_node_id']),
                    turns_ago=best_topic.get('dormancy', 0),
                    snippets=_get_topic_snippets(c, best_topic['start_node_id']),
                )
            ]
            for cand in close_with_support:
                topic = cand['topic']
                options.append(DisambiguationOption(
                    topic_name=topic['name'],
                    topic_start_node_id=topic['start_node_id'],
                    similarity=cand['centroid_sim'],
                    support_count=cand['support_count'],
                    preview=_get_topic_preview(c, topic['start_node_id']),
                    turns_ago=topic.get('dormancy', 0),
                    snippets=_get_topic_snippets(c, topic['start_node_id']),
                ))

            return ReactivationDecision(
                action="DISAMBIGUATE",
                options=options,
                debug=debug_info
            )

        # Clear winner - reactivate best candidate
        debug_info['support_count'] = best_support_count
        debug_info['rank_gap_passes'] = True
        debug_info['dormancy_turns'] = best_topic.get('dormancy', 0)
        debug_info['gates_passed'].append('rank_gap')
        debug_info['gates_passed'].append('dormancy')

        # Compute confidence score (higher = more confident)
        # Factors: similarity, alias hits, support count, dormancy, no close contenders
        # Use dynamic threshold for similarity contribution
        sim_contrib = max(0, (best_sim - sim_threshold) / 0.4) * 0.3
        alias_contrib = min(best_alias_score / 3.0, 1.0) * 0.1 if channel_b_pass else 0

        confidence = min(1.0, (
            sim_contrib +  # Similarity contribution
            alias_contrib +  # Alias contribution (if channel B passed)
            min(best_support_count / 4.0, 1.0) * 0.3 +  # Support contribution (up to 0.3)
            (1.0 if not close_with_support else 0.5) * 0.2 +  # Uniqueness contribution
            min(best_topic.get('dormancy', 0) / 10.0, 1.0) * 0.1  # Dormancy contribution
        ))
        debug_info['confidence'] = confidence

        return ReactivationDecision(
            action="REACTIVATE",
            topic_name=best_topic['name'],
            topic_start_node_id=best_topic['start_node_id'],
            debug=debug_info
        )

    # Execute with connection
    if conn is not None:
        return _probe(conn)

    with get_connection() as c:
        return _probe(c)
