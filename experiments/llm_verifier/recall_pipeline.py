#!/usr/bin/env python3
"""
Recall pipeline with ambiguity detection.

Pipeline:
1. recall_intent(query) → retrieve top-N candidates (fast bi-encoder)
2. ambiguity_detect(candidates) →
   - if ambiguous: return AmbiguousPrompt (no results yet)
   - else: run cross-encoder rerank and return results
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from ambiguity_detect import (
    Candidate,
    AmbiguityConfig,
    AmbiguityResult,
    ambiguity_detect,
    format_disambiguation_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """Successful recall result with ranked candidates."""
    query: str
    results: list[Candidate]
    # Diagnostics
    n_retrieved: int
    n_after_rerank: int
    rerank_threshold: float


@dataclass
class AmbiguousPrompt:
    """Disambiguation needed - return this instead of results."""
    query: str
    prompt: str
    options: list[dict]  # Serializable option data
    # For programmatic handling
    ambiguity_result: AmbiguityResult


@dataclass
class RecallPipelineConfig:
    """Configuration for the recall pipeline."""
    # Retrieval
    top_n: int = 50  # Initial retrieval count
    # Ambiguity detection
    ambiguity_n: int = 30  # Candidates to consider for ambiguity
    ambiguity_k_max: int = 4
    ambiguity_min_cluster_size: int = 3
    ambiguity_rank_gap: int = 3  # Max rank gap for competitiveness
    ambiguity_cohesion_ratio: float = 1.5  # Max cluster diameter / mean dist
    ambiguity_separation_ratio: float = 1.0  # Min inter-cluster dist / mean dist
    # Reranking
    rerank_batch_size: int = 10
    rerank_threshold: float = 0.5


class RecallPipeline:
    """
    Recall pipeline with ambiguity detection.

    Usage:
        pipeline = RecallPipeline(retriever, reranker)
        result = pipeline.recall(query)

        if isinstance(result, AmbiguousPrompt):
            # Show disambiguation UI
            print(result.prompt)
        else:
            # Show results
            for r in result.results:
                print(r.text)
    """

    def __init__(
        self,
        retriever,  # Must have: retrieve(query, top_k) -> list[Candidate]
        reranker=None,  # Optional: rerank(query, candidates) -> list[(Candidate, score)]
        config: Optional[RecallPipelineConfig] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.config = config or RecallPipelineConfig()

    def recall(
        self,
        query: str,
        skip_ambiguity_check: bool = False,
        selected_cluster: Optional[int] = None,
    ) -> Union[RecallResult, AmbiguousPrompt]:
        """
        Execute recall with ambiguity detection.

        Args:
            query: The user query
            skip_ambiguity_check: If True, skip ambiguity detection
            selected_cluster: If set, filter to this cluster (after disambiguation)

        Returns:
            RecallResult if unambiguous or after disambiguation
            AmbiguousPrompt if disambiguation needed
        """
        # Step 1: Retrieve top-N candidates
        logger.info(f"Retrieving top-{self.config.top_n} for query: '{query}'")
        candidates = self.retriever.retrieve(query, top_k=self.config.top_n)
        n_retrieved = len(candidates)

        if not candidates:
            return RecallResult(
                query=query,
                results=[],
                n_retrieved=0,
                n_after_rerank=0,
                rerank_threshold=self.config.rerank_threshold,
            )

        # Step 2: Ambiguity detection (unless skipped or cluster selected)
        if not skip_ambiguity_check and selected_cluster is None:
            ambiguity_config = AmbiguityConfig(
                n=self.config.ambiguity_n,
                k_max=self.config.ambiguity_k_max,
                min_cluster_size=self.config.ambiguity_min_cluster_size,
                rank_gap=self.config.ambiguity_rank_gap,
                cohesion_ratio=self.config.ambiguity_cohesion_ratio,
                separation_ratio=self.config.ambiguity_separation_ratio,
            )

            ambiguity_result = ambiguity_detect(query, candidates, ambiguity_config)

            if ambiguity_result.ambiguous:
                logger.info(
                    f"Ambiguity detected for '{query}': "
                    f"k={ambiguity_result.chosen_k}, "
                    f"clusters={ambiguity_result.cluster_sizes}"
                )

                # Build serializable options
                options = []
                for opt in ambiguity_result.options:
                    options.append({
                        "option_id": opt.option_id,
                        "label": ", ".join(opt.label_terms[:3]) if opt.label_terms else opt.label_snippet,
                        "representative_snippet": opt.representative_snippets[0] if opt.representative_snippets else "",
                        "cluster_size": opt.cluster_size,
                        "best_score": opt.best_score,
                    })

                return AmbiguousPrompt(
                    query=query,
                    prompt=format_disambiguation_prompt(query, ambiguity_result),
                    options=options,
                    ambiguity_result=ambiguity_result,
                )

        # Step 3: Filter to selected cluster if specified
        if selected_cluster is not None:
            # Re-run ambiguity detection to get cluster assignments
            ambiguity_config = AmbiguityConfig(
                n=self.config.ambiguity_n,
                k_max=self.config.ambiguity_k_max,
                min_cluster_size=self.config.ambiguity_min_cluster_size,
                rank_gap=self.config.ambiguity_rank_gap,
                cohesion_ratio=self.config.ambiguity_cohesion_ratio,
                separation_ratio=self.config.ambiguity_separation_ratio,
            )
            ambiguity_result = ambiguity_detect(query, candidates, ambiguity_config)

            if ambiguity_result.ambiguous:
                # Find the selected cluster's representative IDs
                for opt in ambiguity_result.options:
                    if opt.option_id == selected_cluster:
                        selected_ids = set(opt.representative_ids)
                        # For now, use representatives as filter
                        # In production, would track full cluster membership
                        candidates = [c for c in candidates if c.id in selected_ids]
                        break

        # Step 4: Rerank with cross-encoder (if available)
        if self.reranker is not None:
            logger.info(f"Reranking {len(candidates)} candidates")
            reranked = self.reranker.rerank(query, candidates[:self.config.rerank_batch_size])
            # Filter by threshold and sort by score
            results = [
                c for c, score in reranked
                if score >= self.config.rerank_threshold
            ]
        else:
            # No reranker - use retrieval order
            results = candidates[:self.config.rerank_batch_size]

        logger.info(f"Returning {len(results)} results for '{query}'")

        return RecallResult(
            query=query,
            results=results,
            n_retrieved=n_retrieved,
            n_after_rerank=len(results),
            rerank_threshold=self.config.rerank_threshold,
        )


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, candidates: list[Candidate]):
        self.candidates = candidates

    def retrieve(self, query: str, top_k: int) -> list[Candidate]:
        # Sort by retrieval score and return top-k
        sorted_candidates = sorted(self.candidates, key=lambda c: -c.retr_score)
        return sorted_candidates[:top_k]


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self, score_boost: float = 0.0):
        self.score_boost = score_boost

    def rerank(self, query: str, candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
        # Simple mock: use retrieval score + boost
        return [(c, c.retr_score + self.score_boost) for c in candidates]


def demo_pipeline():
    """Demonstrate the pipeline with synthetic data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    np.random.seed(42)

    # Create two well-separated clusters (simulating polysemy)
    centroid_a = np.array([1.0, 0.0, 0.0])
    centroid_b = np.array([0.0, 1.0, 0.0])

    candidates = []

    # Cluster A: programming topic
    for i in range(6):
        noise = np.random.randn(3) * 0.1
        emb = centroid_a + noise
        emb = emb / np.linalg.norm(emb)
        candidates.append(Candidate(
            id=i,
            text=f"Java programming language JVM bytecode example {i}",
            emb=emb,
            retr_score=0.90 - i * 0.01,
        ))

    # Cluster B: coffee topic
    for i in range(6):
        noise = np.random.randn(3) * 0.1
        emb = centroid_b + noise
        emb = emb / np.linalg.norm(emb)
        candidates.append(Candidate(
            id=i + 6,
            text=f"Java coffee beans Indonesian brew roast {i}",
            emb=emb,
            retr_score=0.88 - i * 0.01,
        ))

    # Create pipeline
    retriever = MockRetriever(candidates)
    reranker = MockReranker()
    pipeline = RecallPipeline(retriever, reranker)

    print("=" * 60)
    print("DEMO: Recall Pipeline with Ambiguity Detection")
    print("=" * 60)

    # Query 1: Ambiguous query
    print("\n--- Query: 'java' (ambiguous) ---")
    result = pipeline.recall("java")

    if isinstance(result, AmbiguousPrompt):
        print("Result: AMBIGUOUS")
        print(result.prompt)
    else:
        print(f"Result: {len(result.results)} candidates")

    # Query 2: After user selects cluster 1
    print("\n--- Query: 'java' with selected_cluster=1 ---")
    result = pipeline.recall("java", selected_cluster=1)

    if isinstance(result, AmbiguousPrompt):
        print("Result: AMBIGUOUS")
    else:
        print(f"Result: {len(result.results)} candidates")
        for r in result.results[:3]:
            print(f"  - {r.text[:60]}...")

    # Query 3: Unambiguous (single cluster)
    print("\n--- Query: 'focused' (single cluster) ---")
    # Create single-cluster with clear score dominance
    # All high scores (no competitive sub-clusters)
    single_candidates = []
    for i in range(8):
        noise = np.random.randn(3) * 0.05
        emb = centroid_a + noise
        emb = emb / np.linalg.norm(emb)
        single_candidates.append(Candidate(
            id=i + 100,
            text=f"Python programming syntax code example {i}",
            emb=emb,
            retr_score=0.92 - i * 0.005,  # Tight score range: 0.92 to 0.885
        ))
    # Add some low-scoring noise (scattered, not competitive)
    for i in range(4):
        emb = np.random.randn(3)
        emb = emb / np.linalg.norm(emb)
        single_candidates.append(Candidate(
            id=i + 108,
            text=f"Random unrelated content {i}",
            emb=emb,
            retr_score=0.50 - i * 0.05,  # Low scores: 0.50 to 0.35
        ))

    single_retriever = MockRetriever(single_candidates)
    single_pipeline = RecallPipeline(single_retriever, reranker)
    result = single_pipeline.recall("focused")

    if isinstance(result, AmbiguousPrompt):
        print("Result: AMBIGUOUS")
    else:
        print(f"Result: {len(result.results)} candidates (unambiguous)")


if __name__ == "__main__":
    demo_pipeline()
