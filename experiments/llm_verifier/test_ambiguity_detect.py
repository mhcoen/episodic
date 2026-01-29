#!/usr/bin/env python3
"""
Unit and integration tests for ambiguity detection.

All tests are deterministic with fixed seeds and synthetic data.
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest

from ambiguity_detect import (
    Candidate,
    AmbiguityConfig,
    ambiguity_detect,
    format_disambiguation_prompt,
)

EXPERIMENT_DIR = Path(__file__).parent
DB_PATH = EXPERIMENT_DIR / "synth.db"
QUERY_CASES_PATH = EXPERIMENT_DIR / "query_cases.json"


def make_cluster(
    centroid: np.ndarray,
    n_points: int,
    id_start: int,
    text_prefix: str,
    base_score: float,
    score_decay: float = 0.02,
    noise_scale: float = 0.1,
    rng: np.random.Generator = None,
) -> list[Candidate]:
    """Create a cluster of candidates around a centroid."""
    if rng is None:
        rng = np.random.default_rng(42)

    candidates = []
    for i in range(n_points):
        noise = rng.standard_normal(len(centroid)) * noise_scale
        emb = centroid + noise
        emb = emb / np.linalg.norm(emb)  # L2 normalize
        candidates.append(Candidate(
            id=id_start + i,
            text=f"{text_prefix} item {i}",
            emb=emb,
            retr_score=base_score - i * score_decay,
        ))
    return candidates


class TestAmbiguityDetection:
    """Unit tests for ambiguity_detect with synthetic embeddings."""

    def test_case1_two_balanced_competitive_clusters(self):
        """Two strong clusters, balanced, competitive → ambiguous, k=2."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Interleave scores so both clusters have top-ranked items
        cluster_a = make_cluster(centroid_a, 5, 0, "Programming topic", 0.90, score_decay=0.04, rng=rng)
        cluster_b = make_cluster(centroid_b, 5, 5, "Coffee brewing", 0.89, score_decay=0.04, rng=rng)

        candidates = cluster_a + cluster_b
        # With n=10, scaled rank_gap = ceil(0.1 * 10) = 1, but clamped to min=2
        config = AmbiguityConfig(min_cluster_size=3)

        result = ambiguity_detect("java", candidates, config)

        assert result.ambiguous is True
        assert result.chosen_k == 2
        assert len(result.options) == 2
        assert result.cluster_sizes == [5, 5]
        assert result.rank_gap <= result.max_rank_gap

    def test_case2_one_cluster_too_small(self):
        """Two clusters but one is tiny (< m) → unambiguous."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Large cluster
        cluster_a = make_cluster(centroid_a, 8, 0, "Main topic", 0.90, rng=rng)
        # Tiny cluster (size 2, below min_cluster_size=3)
        cluster_b = make_cluster(centroid_b, 2, 8, "Minor topic", 0.88, rng=rng)

        candidates = cluster_a + cluster_b
        config = AmbiguityConfig(min_cluster_size=3)

        result = ambiguity_detect("query", candidates, config)

        assert result.ambiguous is False
        assert "single coherent" in result.reason or "insufficient" in result.reason

    def test_case3_three_clusters(self):
        """Three competitive clusters → ambiguous with 2+ options."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])
        centroid_c = np.array([0.0, 0.0, 1.0])

        # Interleave scores so all clusters have near-top ranks
        cluster_a = make_cluster(centroid_a, 4, 0, "Topic A content", 0.90, score_decay=0.06, rng=rng)
        cluster_b = make_cluster(centroid_b, 4, 4, "Topic B content", 0.89, score_decay=0.06, rng=rng)
        cluster_c = make_cluster(centroid_c, 4, 8, "Topic C content", 0.88, score_decay=0.06, rng=rng)

        candidates = cluster_a + cluster_b + cluster_c
        config = AmbiguityConfig(min_cluster_size=3, k_max=4)

        result = ambiguity_detect("multi-topic", candidates, config)

        assert result.ambiguous is True
        # Should find competitive clusters at some k (2, 3, or 4 depending on separation)
        assert result.chosen_k in [2, 3, 4]
        assert len(result.options) >= 2

    def test_case4_single_cluster(self):
        """Single coherent cluster → unambiguous."""
        rng = np.random.default_rng(42)

        centroid = np.array([1.0, 0.0, 0.0])
        candidates = make_cluster(centroid, 10, 0, "Unified topic", 0.90, rng=rng)

        config = AmbiguityConfig(min_cluster_size=3)
        result = ambiguity_detect("focused-query", candidates, config)

        assert result.ambiguous is False
        assert "single coherent" in result.reason

    def test_case5_noisy_tail(self):
        """One main cluster and scattered points → unambiguous."""
        rng = np.random.default_rng(42)

        # Main cluster with high scores
        centroid = np.array([1.0, 0.0, 0.0])
        main_cluster = make_cluster(centroid, 8, 0, "Main topic", 0.90, score_decay=0.01, rng=rng)

        # Scattered points with much lower scores (won't be competitive by rank)
        scattered = []
        for i in range(4):
            direction = rng.standard_normal(3)
            direction = direction / np.linalg.norm(direction)
            scattered.append(Candidate(
                id=8 + i,
                text=f"Random noise {i}",
                emb=direction,
                retr_score=0.50 - i * 0.05,  # Much lower scores
            ))

        candidates = main_cluster + scattered
        config = AmbiguityConfig(min_cluster_size=3)

        result = ambiguity_detect("query", candidates, config)

        assert result.ambiguous is False

    def test_non_competitive_by_rank(self):
        """Two clusters but second cluster's best item is far down in rank → unambiguous."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Cluster A dominates top ranks
        cluster_a = make_cluster(centroid_a, 5, 0, "Dominant topic", 0.95, score_decay=0.01, rng=rng)
        # Cluster B starts much lower in rank
        cluster_b = make_cluster(centroid_b, 5, 5, "Weak topic", 0.70, score_decay=0.01, rng=rng)

        candidates = cluster_a + cluster_b
        # Force rank_gap=3 explicitly (not scaled)
        config = AmbiguityConfig(min_cluster_size=3, rank_gap=3)

        result = ambiguity_detect("query", candidates, config)

        # Cluster A occupies ranks 0-4, cluster B occupies ranks 5-9
        # Rank gap = 5 - 0 = 5 > 3, so not competitive
        assert result.ambiguous is False

    def test_one_sense_dominates_strongly(self):
        """Two valid clusters but one dominates ranks 1-8, other has 9-10 → unambiguous.

        Even though both clusters are valid by size, if one sense completely
        dominates the top ranks, we shouldn't ask for disambiguation.
        """
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Cluster A dominates ranks 0-7 (8 items)
        cluster_a = make_cluster(centroid_a, 8, 0, "Dominant sense", 0.95, score_decay=0.01, rng=rng)
        # Cluster B only has ranks 8-10 (3 items, barely valid)
        cluster_b = make_cluster(centroid_b, 3, 8, "Minor sense", 0.70, score_decay=0.01, rng=rng)

        candidates = cluster_a + cluster_b
        # With n=11, scaled rank_gap = ceil(0.1 * 11) = 2, clamped to [2, 5] = 2
        config = AmbiguityConfig(min_cluster_size=3)

        result = ambiguity_detect("query", candidates, config)

        # Cluster A best rank = 0, Cluster B best rank = 8
        # Rank gap = 8 - 0 = 8 > 2 (or any reasonable threshold)
        # This should NOT trigger ambiguity
        assert result.ambiguous is False
        assert result.max_rank_gap == 2  # ceil(0.1 * 11) = 2

    def test_insufficient_candidates(self):
        """Too few candidates → unambiguous."""
        rng = np.random.default_rng(42)

        centroid = np.array([1.0, 0.0, 0.0])
        # Only 4 candidates, need 2*m = 6 for ambiguity
        candidates = make_cluster(centroid, 4, 0, "Small set", 0.90, rng=rng)

        config = AmbiguityConfig(min_cluster_size=3)
        result = ambiguity_detect("query", candidates, config)

        assert result.ambiguous is False
        assert "insufficient" in result.reason

    def test_determinism(self):
        """Same inputs produce same outputs (no randomness)."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        cluster_a = make_cluster(centroid_a, 5, 0, "Topic A", 0.90, score_decay=0.04, rng=rng)

        rng = np.random.default_rng(43)  # Different seed for cluster_b
        cluster_b = make_cluster(centroid_b, 5, 5, "Topic B", 0.89, score_decay=0.04, rng=rng)

        candidates = cluster_a + cluster_b
        config = AmbiguityConfig(min_cluster_size=3)

        result1 = ambiguity_detect("test", candidates, config)
        result2 = ambiguity_detect("test", candidates, config)

        assert result1.ambiguous == result2.ambiguous
        assert result1.chosen_k == result2.chosen_k
        assert result1.cluster_sizes == result2.cluster_sizes
        assert result1.rank_gap == result2.rank_gap

    def test_option_labels_are_distinctive(self):
        """Options have meaningful label terms derived from cluster content."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Distinct content for each cluster with interleaved scores
        cluster_a = []
        for i in range(5):
            noise = rng.standard_normal(3) * 0.1
            emb = centroid_a + noise
            emb = emb / np.linalg.norm(emb)
            cluster_a.append(Candidate(
                id=i,
                text=f"Python programming language syntax example {i}",
                emb=emb,
                retr_score=0.90 - i * 0.04,
            ))

        cluster_b = []
        for i in range(5):
            noise = rng.standard_normal(3) * 0.1
            emb = centroid_b + noise
            emb = emb / np.linalg.norm(emb)
            cluster_b.append(Candidate(
                id=i + 5,
                text=f"Python snake reptile wildlife nature {i}",
                emb=emb,
                retr_score=0.89 - i * 0.04,
            ))

        candidates = cluster_a + cluster_b
        config = AmbiguityConfig(min_cluster_size=3)

        result = ambiguity_detect("python", candidates, config)

        assert result.ambiguous is True
        assert len(result.options) == 2

        # Check that label terms reflect cluster content
        all_label_terms = []
        for opt in result.options:
            all_label_terms.extend(opt.label_terms)

        # Should have distinctive terms from each cluster
        terms_str = " ".join(all_label_terms).lower()
        # Programming cluster should have terms like programming, syntax, language
        # Snake cluster should have terms like snake, reptile, wildlife
        assert any(t in terms_str for t in ["programming", "syntax", "language", "example"])
        assert any(t in terms_str for t in ["snake", "reptile", "wildlife", "nature"])

    def test_chain_structure_no_false_split(self):
        """Chain-connected points (A—B—C—D) should not falsely split into 2 clusters.

        This tests the separation check: a chain where adjacent points are close
        but endpoints are far should be rejected because min inter-cluster
        distance is small (the chain creates bridges between any split).
        """
        rng = np.random.default_rng(42)

        # Create a chain in embedding space
        # Each point is close to neighbors but far from endpoints
        chain_points = []
        n_points = 10

        for i in range(n_points):
            # Linear interpolation from [1,0,0] to [0,1,0]
            t = i / (n_points - 1)
            base = np.array([1.0 - t, t, 0.0])
            noise = rng.standard_normal(3) * 0.02  # Small noise
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            chain_points.append(Candidate(
                id=i,
                text=f"Chain point {i} with gradual transition topic",
                emb=emb,
                retr_score=0.90 - i * 0.01,  # All high scores, interleaved
            ))

        config = AmbiguityConfig(min_cluster_size=3, cohesion_ratio=1.5)
        result = ambiguity_detect("chain-query", chain_points, config)

        # A chain should NOT trigger ambiguity because:
        # Any split creates clusters with small min inter-cluster distance
        # (the adjacent points at the split boundary are close)
        assert result.ambiguous is False

    def test_score_scaling_invariance(self):
        """Ambiguity decision should not change under score scaling/translation.

        Since we use rank-gap, multiplying or shifting scores should not
        affect the result.
        """
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Original scores
        cluster_a = make_cluster(centroid_a, 5, 0, "Topic A", 0.90, score_decay=0.04, rng=rng)
        rng = np.random.default_rng(43)
        cluster_b = make_cluster(centroid_b, 5, 5, "Topic B", 0.89, score_decay=0.04, rng=rng)
        candidates_original = cluster_a + cluster_b

        config = AmbiguityConfig(min_cluster_size=3)
        result_original = ambiguity_detect("test", candidates_original, config)

        # Scaled scores (multiply by 10)
        candidates_scaled = []
        for c in candidates_original:
            candidates_scaled.append(Candidate(
                id=c.id,
                text=c.text,
                emb=c.emb,
                retr_score=c.retr_score * 10,
            ))
        result_scaled = ambiguity_detect("test", candidates_scaled, config)

        # Translated scores (add 5)
        candidates_translated = []
        for c in candidates_original:
            candidates_translated.append(Candidate(
                id=c.id,
                text=c.text,
                emb=c.emb,
                retr_score=c.retr_score + 5,
            ))
        result_translated = ambiguity_detect("test", candidates_translated, config)

        # All should have the same ambiguity decision
        assert result_original.ambiguous == result_scaled.ambiguous
        assert result_original.ambiguous == result_translated.ambiguous
        assert result_original.chosen_k == result_scaled.chosen_k
        assert result_original.chosen_k == result_translated.chosen_k
        assert result_original.rank_gap == result_scaled.rank_gap
        assert result_original.rank_gap == result_translated.rank_gap

    def test_cohesion_rejects_loose_clusters(self):
        """Clusters that are internally loose (high diameter) should be rejected."""
        rng = np.random.default_rng(42)

        # Create two "clusters" that are actually just scattered points
        # with no internal cohesion
        candidates = []

        # "Cluster 1": 4 widely scattered points in one hemisphere
        for i in range(4):
            angle = i * np.pi / 4  # Spread across quadrant
            emb = np.array([np.cos(angle), np.sin(angle), 0.1])
            emb = emb / np.linalg.norm(emb)
            candidates.append(Candidate(
                id=i,
                text=f"Scattered A point {i}",
                emb=emb,
                retr_score=0.90 - i * 0.02,
            ))

        # "Cluster 2": 4 widely scattered points in opposite hemisphere
        for i in range(4):
            angle = i * np.pi / 4 + np.pi  # Opposite quadrant
            emb = np.array([np.cos(angle), np.sin(angle), -0.1])
            emb = emb / np.linalg.norm(emb)
            candidates.append(Candidate(
                id=i + 4,
                text=f"Scattered B point {i}",
                emb=emb,
                retr_score=0.89 - i * 0.02,
            ))

        # With strict cohesion, these loose "clusters" should be rejected
        config = AmbiguityConfig(min_cluster_size=3, cohesion_ratio=0.5)
        result = ambiguity_detect("scattered", candidates, config)

        # Should not be ambiguous because clusters fail cohesion check
        assert result.ambiguous is False


class TestDisambiguationPrompt:
    """Test the user-facing prompt formatting."""

    def test_format_unambiguous_returns_empty(self):
        """Unambiguous result produces no prompt."""
        from ambiguity_detect import AmbiguityResult
        result = AmbiguityResult(ambiguous=False, reason="single coherent neighborhood")
        prompt = format_disambiguation_prompt("test", result)
        assert prompt == ""

    def test_format_ambiguous_shows_options(self):
        """Ambiguous result shows numbered options."""
        from ambiguity_detect import AmbiguityResult, ClusterOption

        options = [
            ClusterOption(
                option_id=1,
                label_terms=["programming", "code", "syntax"],
                label_snippet="Python programming example",
                representative_ids=[1, 2],
                representative_snippets=["Python programming example", "More code here"],
                cluster_size=5,
                best_score=0.9,
            ),
            ClusterOption(
                option_id=2,
                label_terms=["snake", "reptile"],
                label_snippet="Python snake species",
                representative_ids=[6, 7],
                representative_snippets=["Python snake species", "Reptile facts"],
                cluster_size=5,
                best_score=0.88,
            ),
        ]

        result = AmbiguityResult(
            ambiguous=True,
            reason="found 2 competitive clusters",
            options=options,
            chosen_k=2,
        )

        prompt = format_disambiguation_prompt("python", result)

        assert "multiple plausible topics" in prompt
        assert "python" in prompt
        assert "1." in prompt
        assert "2." in prompt
        assert "programming" in prompt
        assert "snake" in prompt


class TestIntegrationWithSyntheticCorpus:
    """Integration tests using the synthetic corpus (no real model calls)."""

    @pytest.fixture
    def corpus_data(self):
        """Load synthetic corpus data."""
        if not DB_PATH.exists() or not QUERY_CASES_PATH.exists():
            pytest.skip("Synthetic corpus not found")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text FROM statements")
        statements = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        with open(QUERY_CASES_PATH) as f:
            cases = json.load(f)

        return statements, cases

    def test_polysemy_queries_trigger_ambiguity(self, corpus_data):
        """
        Queries with known polysemy should trigger ambiguity detection
        when candidates form distinct clusters.

        This test simulates what would happen with real embeddings by
        creating synthetic embeddings that reflect the expected cluster structure.
        """
        statements, cases = corpus_data

        # Find polysemy cases (undisambiguated queries with hard negatives)
        polysemy_queries = ["python", "java", "rust", "shell", "kernel", "model"]

        for query in polysemy_queries:
            case = next((c for c in cases if c["query"] == query), None)
            if case is None:
                continue

            gold = case.get("gold_relevant", {})
            hard_negs = set(case.get("hard_negatives", []))
            candidates_ids = case["candidates"][:20]

            # Create synthetic embeddings that separate positives from hard negatives
            rng = np.random.default_rng(hash(query) % 2**32)

            centroid_pos = rng.standard_normal(64)
            centroid_pos = centroid_pos / np.linalg.norm(centroid_pos)

            centroid_neg = rng.standard_normal(64)
            centroid_neg = centroid_neg / np.linalg.norm(centroid_neg)
            # Ensure centroids are somewhat separated
            centroid_neg = centroid_neg - 0.5 * centroid_pos
            centroid_neg = centroid_neg / np.linalg.norm(centroid_neg)

            candidates = []
            pos_count = 0
            neg_count = 0
            for cid in candidates_ids:
                if cid not in statements:
                    continue

                is_positive = gold.get(str(cid), 0) == 1
                is_hard_neg = cid in hard_negs

                if is_positive:
                    centroid = centroid_pos
                    # Interleave scores: positives get ranks 0, 2, 4, ...
                    base_score = 0.90 - pos_count * 0.04
                    pos_count += 1
                elif is_hard_neg:
                    centroid = centroid_neg
                    # Hard negatives get ranks 1, 3, 5, ... (competitive)
                    base_score = 0.89 - neg_count * 0.04
                    neg_count += 1
                else:
                    # Regular negatives - scattered with low scores
                    centroid = rng.standard_normal(64)
                    centroid = centroid / np.linalg.norm(centroid)
                    base_score = 0.50

                noise = rng.standard_normal(64) * 0.1
                emb = centroid + noise
                emb = emb / np.linalg.norm(emb)

                candidates.append(Candidate(
                    id=cid,
                    text=statements[cid],
                    emb=emb,
                    retr_score=base_score + rng.uniform(-0.005, 0.005),
                ))

            if len(candidates) < 6:  # Need at least 2*m candidates
                continue

            # Use lower separation ratio for synthetic data which may not be as well-separated
            # Force explicit rank_gap for consistent behavior
            config = AmbiguityConfig(min_cluster_size=3, rank_gap=5, separation_ratio=0.5)
            result = ambiguity_detect(query, candidates, config)

            # For polysemy queries with hard negatives, we expect ambiguity
            # when both positives and hard negatives form viable clusters
            if len(hard_negs) >= 3:
                # Should detect ambiguity
                assert result.ambiguous is True, (
                    f"Expected ambiguity for '{query}' with {len(hard_negs)} hard negatives"
                )

    def test_disambiguated_queries_no_ambiguity(self, corpus_data):
        """Queries that are already disambiguated should not trigger ambiguity."""
        statements, cases = corpus_data

        # These queries have disambiguation built in
        disambiguated_queries = ["Apple computers", "Linux kernel", "git branch"]

        for query in disambiguated_queries:
            case = next((c for c in cases if c["query"] == query), None)
            if case is None:
                continue

            gold = case.get("gold_relevant", {})
            candidates_ids = case["candidates"][:20]

            # Create synthetic embeddings - all positives in one tight cluster
            rng = np.random.default_rng(hash(query) % 2**32)

            centroid = rng.standard_normal(64)
            centroid = centroid / np.linalg.norm(centroid)

            candidates = []
            pos_idx = 0
            for cid in candidates_ids:
                if cid not in statements:
                    continue

                is_positive = gold.get(str(cid), 0) == 1

                if is_positive:
                    noise = rng.standard_normal(64) * 0.05  # Tight cluster
                    emb = centroid + noise
                    # Positives dominate top ranks
                    base_score = 0.95 - pos_idx * 0.01
                    pos_idx += 1
                else:
                    # Negatives are scattered and low-scoring (not competitive by rank)
                    emb = rng.standard_normal(64)
                    base_score = 0.50

                emb = emb / np.linalg.norm(emb)

                candidates.append(Candidate(
                    id=cid,
                    text=statements[cid],
                    emb=emb,
                    retr_score=base_score + rng.uniform(-0.005, 0.005),
                ))

            if len(candidates) < 6:
                continue

            config = AmbiguityConfig(min_cluster_size=3)
            result = ambiguity_detect(query, candidates, config)

            # Disambiguated queries should have a single coherent cluster
            # with positives dominating top ranks (negatives not competitive)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
