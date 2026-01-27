"""
Recall pipeline.

End-to-end recall: query → semantic hits → promotion → ranking → expansion → formatting.

Includes ambiguity detection: if top candidates form multiple competitive clusters,
returns an AMBIGUOUS result for the caller to handle disambiguation.
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from episodic.query.types import ResolvedQuery
from episodic.rag_collections import get_multi_collection_rag, CollectionType

from .promotion import promote_hits_to_topics, PromotionResult
from .ranking import rank_topics, RankingResult, get_top_topics, get_top_statements
from .expansion import expand_topic, Tier, TopicExpansion, ExpansionConfig, DEFAULT_CONFIG
from .budget import map_parser_output_to_budget, RecallBudget, get_budget_description
from .formatting import format_recall_result, FormattedRecall
from .ambiguity import (
    Candidate as AmbiguityCandidate,
    AmbiguityConfig,
    AmbiguityResult,
    ClusterOption,
    ambiguity_detect,
    format_disambiguation_prompt,
)

logger = logging.getLogger(__name__)

# Minimum candidates required to run ambiguity detection
MIN_CANDIDATES_FOR_AMBIGUITY = 10


class RecallResultKind(Enum):
    """Kind of recall result."""
    HITS = "hits"           # Normal results
    AMBIGUOUS = "ambiguous" # Multiple plausible interpretations detected
    EMPTY = "empty"         # No results found


@dataclass
class SemanticHit:
    """A semantic search hit with optional embedding."""
    exchange_id: str
    relevance_score: float
    metadata: Dict
    text: str = ""
    embedding: Optional[np.ndarray] = None


@dataclass
class RecallResult:
    """Complete result of a recall operation."""
    kind: RecallResultKind
    formatted: FormattedRecall
    budget: RecallBudget
    # For HITS/EMPTY results
    promotion: Optional[PromotionResult] = None
    ranking: Optional[RankingResult] = None
    topic_expansions: Optional[List[TopicExpansion]] = None
    # For AMBIGUOUS results
    ambiguity: Optional[AmbiguityResult] = None
    cluster_options: List[ClusterOption] = field(default_factory=list)
    raw_hits: List[SemanticHit] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Get formatted context string for LLM injection."""
        if self.kind == RecallResultKind.AMBIGUOUS:
            return ""  # Don't inject context for ambiguous queries
        return self.formatted.to_context_string(self.budget)

    def is_empty(self) -> bool:
        """Check if no results were found."""
        return self.kind == RecallResultKind.EMPTY or (
            len(self.formatted.conversation_blocks) == 0 and
            len(self.formatted.statement_blocks) == 0
        )

    def is_ambiguous(self) -> bool:
        """Check if query was ambiguous."""
        return self.kind == RecallResultKind.AMBIGUOUS

    def get_disambiguation_prompt(self, query: str) -> str:
        """Get user-facing disambiguation prompt."""
        if not self.ambiguity:
            return ""
        return format_disambiguation_prompt(query, self.ambiguity)


def recall(
    conn: sqlite3.Connection,
    query: ResolvedQuery,
    query_form: Optional[str] = None,
    max_semantic_hits: int = 50,
    expansion_config: ExpansionConfig = DEFAULT_CONFIG,
    skip_ambiguity_check: bool = False,
    selected_cluster: Optional[int] = None,
) -> RecallResult:
    """
    Execute a recall query.

    Args:
        conn: SQLite connection
        query: ResolvedQuery from parser/resolver
        query_form: From DiscussionQuery.query_form (if applicable)
        max_semantic_hits: Maximum hits to fetch from semantic search
        expansion_config: Configuration for expansion tiers
        skip_ambiguity_check: If True, skip ambiguity detection
        selected_cluster: If set, filter to this cluster (after disambiguation)

    Returns:
        RecallResult with formatted blocks and metadata, or AMBIGUOUS result
    """
    # Step 1: Determine budget from parser output
    budget = map_parser_output_to_budget(
        query_form=query_form,
        has_broadness_cue=query.has_broadness_cue,
        speaker=query.speaker,
        mode=query.mode
    )
    logger.debug(f"Recall budget: {get_budget_description(budget)}")

    # Step 2: Get semantic hits (with embeddings for ambiguity detection)
    n_fetch = int(max_semantic_hits * budget.overfetch_multiplier)
    need_embeddings = not skip_ambiguity_check and selected_cluster is None
    hits = _get_semantic_hits(
        query.target,
        n_fetch,
        query.temporal,
        budget.broad_horizon,
        include_embeddings=need_embeddings,
    )

    if not hits:
        return _empty_result(budget)

    logger.debug(f"Semantic search returned {len(hits)} hits")

    # Step 3: Ambiguity detection (unless skipped or cluster already selected)
    if need_embeddings and len(hits) >= MIN_CANDIDATES_FOR_AMBIGUITY:
        ambiguity_result = _check_ambiguity(query.target or "", hits)

        if ambiguity_result and ambiguity_result.ambiguous:
            logger.info(
                f"Ambiguity detected for '{query.target}': "
                f"k={ambiguity_result.chosen_k}, clusters={ambiguity_result.cluster_sizes}"
            )
            return RecallResult(
                kind=RecallResultKind.AMBIGUOUS,
                formatted=FormattedRecall(
                    conversation_blocks=[],
                    statement_blocks=[],
                    total_exchanges=0,
                ),
                budget=budget,
                ambiguity=ambiguity_result,
                cluster_options=ambiguity_result.options,
                raw_hits=hits,
            )

    # Step 4: Filter to selected cluster if specified
    if selected_cluster is not None:
        hits = _filter_to_cluster(query.target or "", hits, selected_cluster)
        if not hits:
            return _empty_result(budget)

    # Step 5: Promote hits to topics
    # Convert SemanticHit to dict for promotion (legacy interface)
    hits_for_promotion = [
        {
            'exchange_id': h.exchange_id,
            'relevance_score': h.relevance_score,
            'metadata': h.metadata,
        }
        for h in hits
    ]
    promotion = promote_hits_to_topics(conn, hits_for_promotion, similarity_key='relevance_score')

    for audit_entry in promotion.audit_entries:
        logger.debug(f"Promotion audit: {audit_entry}")

    # Step 6: Rank topics
    ranking = rank_topics(promotion)

    logger.debug(f"Ranked {len(ranking.ranked_topics)} topics, "
                 f"{len(ranking.unassigned_hits)} unassigned hits")

    # Step 7: Select topics and statements based on budget
    top_topics = get_top_topics(ranking, budget.max_topics)
    top_topic_ids = [t.topic_id for t in top_topics]

    # Get statement candidates (exclude hits from top topics)
    statement_hits = get_top_statements(
        ranking,
        budget.max_statements,
        exclude_topic_ids=top_topic_ids
    )

    # Step 8: Expand selected topics
    topic_expansions = []
    for ranked_topic in top_topics:
        # Determine tier based on budget and evidence strength
        tier = _select_tier(ranked_topic, budget)

        expansion = expand_topic(
            conn,
            ranked_topic,
            tier,
            expansion_config
        )
        topic_expansions.append(expansion)

    # Step 9: Build score map for provenance
    topic_scores = {
        t.topic_id: (t.best_hit, t.hit_count)
        for t in top_topics
    }

    # Step 10: Format result
    formatted = format_recall_result(
        conn,
        topic_expansions,
        statement_hits,
        topic_scores,
        budget
    )

    return RecallResult(
        kind=RecallResultKind.HITS,
        formatted=formatted,
        budget=budget,
        promotion=promotion,
        ranking=ranking,
        topic_expansions=topic_expansions,
    )


def _check_ambiguity(target: str, hits: List[SemanticHit]) -> Optional[AmbiguityResult]:
    """
    Run ambiguity detection on semantic hits.

    Args:
        target: Query target string
        hits: List of SemanticHit with embeddings

    Returns:
        AmbiguityResult if detection ran, None if insufficient embeddings
    """
    # Build candidate list for ambiguity detection
    candidates = []
    expected_dim = None

    for i, hit in enumerate(hits):
        if hit.embedding is None:
            logger.debug(f"Hit {i} ({hit.exchange_id}) has no embedding, skipping")
            continue

        emb = hit.embedding

        # Dimension consistency check
        if expected_dim is None:
            expected_dim = len(emb)
            logger.debug(f"Embedding dimension: {expected_dim}")
        elif len(emb) != expected_dim:
            logger.warning(
                f"Hit {i} ({hit.exchange_id}) has dimension {len(emb)}, "
                f"expected {expected_dim}. Skipping ambiguity detection."
            )
            return None

        # Ensure embedding is L2-normalized
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        candidates.append(AmbiguityCandidate(
            id=i,  # Use index as ID for filtering later
            text=hit.text,
            emb=emb,
            retr_score=hit.relevance_score,
        ))

    if len(candidates) < MIN_CANDIDATES_FOR_AMBIGUITY:
        logger.debug(f"Only {len(candidates)} candidates with embeddings, skipping ambiguity check")
        return None

    config = AmbiguityConfig()
    result = ambiguity_detect(target, candidates, config)

    # Log calibration data for tuning
    if result:
        logger.info(
            f"Ambiguity check: query='{target}', n={result.n_candidates}, "
            f"ambiguous={result.ambiguous}, k={result.chosen_k}, "
            f"sizes={result.cluster_sizes}, rank_gap={result.rank_gap}, "
            f"max_rank_gap={result.max_rank_gap}, "
            f"cohesion={result.cohesion_ratios}, separation={result.separation_ratio:.3f}"
            if result.separation_ratio else
            f"Ambiguity check: query='{target}', n={result.n_candidates}, "
            f"ambiguous={result.ambiguous}, reason='{result.reason}'"
        )

    return result


def _filter_to_cluster(
    target: str,
    hits: List[SemanticHit],
    selected_cluster: int,
) -> List[SemanticHit]:
    """
    Filter hits to those belonging to the selected cluster.

    Args:
        target: Query target string
        hits: Original hits with embeddings
        selected_cluster: Cluster option_id to filter to

    Returns:
        Filtered hits belonging to the selected cluster
    """
    # Re-run ambiguity detection to get cluster assignments
    ambiguity_result = _check_ambiguity(target, hits)
    if not ambiguity_result or not ambiguity_result.ambiguous:
        return hits  # No ambiguity, return all hits

    # Find the selected cluster's member indices
    for opt in ambiguity_result.options:
        if opt.option_id == selected_cluster:
            # Get all candidate indices in this cluster (full membership)
            selected_indices = set(opt.member_indices)
            logger.debug(f"Filtering to cluster {selected_cluster} with {len(selected_indices)} members")

            # Return hits at those indices
            return [hits[i] for i in sorted(selected_indices) if i < len(hits)]

    logger.warning(f"Selected cluster {selected_cluster} not found in ambiguity result")
    return hits


def _get_semantic_hits(
    target: Optional[str],
    n_results: int,
    temporal: Optional[Tuple],
    broad_horizon: bool,
    min_similarity: float = 0.35,  # Filter out garbage hits (lowered from 0.5)
    include_embeddings: bool = False,
) -> List[SemanticHit]:
    """Get semantic hits from Chroma.

    Args:
        target: Query string
        n_results: Maximum results to return
        temporal: Optional (start_utc, end_utc) temporal filter
        broad_horizon: If True, relax temporal constraints
        min_similarity: Minimum similarity threshold
        include_embeddings: If True, request embeddings for ambiguity detection

    Returns:
        List of SemanticHit objects
    """
    if not target or not target.strip():
        return []

    try:
        rag = get_multi_collection_rag()
        collection = rag.get_collection(CollectionType.CONVERSATION)

        # Build include list - always need metadatas, distances, documents
        include_list = ["metadatas", "distances", "documents"]
        if include_embeddings:
            include_list.append("embeddings")

        results = collection.query(
            query_texts=[target],
            n_results=n_results,
            include=include_list,
        )

        if not results or not results['ids'] or not results['ids'][0]:
            return []

        # Format results
        hits = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            distance = results['distances'][0][i] if results['distances'] else 0
            document = results['documents'][0][i] if results.get('documents') else ""

            # Get embedding if requested
            embedding = None
            if include_embeddings and results.get('embeddings') and results['embeddings'][0]:
                emb_data = results['embeddings'][0][i]
                embedding = np.array(emb_data) if emb_data is not None else None

            # Convert distance to similarity (Chroma uses L2 distance)
            # L2 distance of 0 = identical, higher = less similar
            # For normalized embeddings, max distance is ~2 (opposite vectors)
            similarity = max(0.0, 1.0 - (distance / 2.0))

            # Filter by minimum similarity
            if similarity < min_similarity:
                logger.debug(f"Filtered hit {metadata.get('user_id', 'unknown')}: similarity {similarity:.3f} < {min_similarity}")
                continue

            # Apply temporal filter if present
            if temporal:
                ts_str = metadata.get('timestamp')
                if ts_str:
                    in_range = _in_temporal_range(ts_str, temporal, broad_horizon)
                    if not in_range:
                        logger.debug(f"Filtered hit {metadata.get('user_id', 'unknown')}: outside temporal range")
                        continue

            hits.append(SemanticHit(
                exchange_id=metadata.get('user_id', results['ids'][0][i]),
                relevance_score=similarity,
                metadata=metadata,
                text=document,
                embedding=embedding,
            ))

        logger.debug(f"Semantic search: {len(results['ids'][0])} raw -> {len(hits)} after filtering")
        return hits

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def _in_temporal_range(
    ts_str: str,
    temporal: Tuple,
    broad_horizon: bool
) -> bool:
    """Check if timestamp is within temporal range."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    try:
        # Parse timestamp
        ts_clean = ts_str.replace('Z', '+00:00')
        ts = datetime.fromisoformat(ts_clean)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        
        start_utc, end_utc = temporal
        
        # Broad horizon: relax constraints
        if broad_horizon:
            # Only enforce end bound, allow older content
            return ts < end_utc
        
        return start_utc <= ts < end_utc
        
    except (ValueError, TypeError):
        # Can't parse timestamp - include by default
        return True


def _select_tier(ranked_topic, budget: RecallBudget) -> Tier:
    """Select expansion tier based on evidence strength and budget."""
    base_tier = budget.topic_tier
    
    # Escalate to Tier C if strong evidence
    if base_tier == Tier.B and ranked_topic.hit_count >= 3:
        return Tier.C
    
    return base_tier


def _empty_result(budget: RecallBudget) -> RecallResult:
    """Create empty result for no-hit case."""
    return RecallResult(
        kind=RecallResultKind.EMPTY,
        formatted=FormattedRecall(
            conversation_blocks=[],
            statement_blocks=[],
            total_exchanges=0
        ),
        budget=budget,
        promotion=PromotionResult(by_topic={}, topic_info={}, audit_entries=[]),
        ranking=RankingResult(ranked_topics=[], unassigned_hits=[]),
        topic_expansions=[],
    )
