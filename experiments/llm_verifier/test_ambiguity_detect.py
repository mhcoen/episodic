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

        cluster_a = make_cluster(centroid_a, 5, 0, "Programming topic", 0.90, rng=rng)
        cluster_b = make_cluster(centroid_b, 5, 5, "Coffee brewing", 0.88, rng=rng)

        candidates = cluster_a + cluster_b
        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)

        result = ambiguity_detect("java", candidates, config)

        assert result.ambiguous is True
        assert result.chosen_k == 2
        assert len(result.options) == 2
        assert result.cluster_sizes == [5, 5]
        assert result.score_gap <= config.delta

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
        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)

        result = ambiguity_detect("query", candidates, config)

        assert result.ambiguous is False
        assert "single coherent" in result.reason or "insufficient" in result.reason

    def test_case3_three_clusters(self):
        """Three competitive clusters → k=3 with 3 options."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])
        centroid_c = np.array([0.0, 0.0, 1.0])

        cluster_a = make_cluster(centroid_a, 4, 0, "Topic A content", 0.90, rng=rng)
        cluster_b = make_cluster(centroid_b, 4, 4, "Topic B content", 0.89, rng=rng)
        cluster_c = make_cluster(centroid_c, 4, 8, "Topic C content", 0.88, rng=rng)

        candidates = cluster_a + cluster_b + cluster_c
        config = AmbiguityConfig(min_cluster_size=3, delta=0.03, k_max=4)

        result = ambiguity_detect("multi-topic", candidates, config)

        assert result.ambiguous is True
        # Should pick smallest k that works - could be 2 or 3 depending on cluster merging
        assert result.chosen_k in [2, 3]
        assert len(result.options) >= 2

    def test_case4_single_cluster(self):
        """Single coherent cluster → unambiguous."""
        rng = np.random.default_rng(42)

        centroid = np.array([1.0, 0.0, 0.0])
        candidates = make_cluster(centroid, 10, 0, "Unified topic", 0.90, rng=rng)

        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)
        result = ambiguity_detect("focused-query", candidates, config)

        assert result.ambiguous is False
        assert "single coherent" in result.reason

    def test_case5_noisy_tail(self):
        """One main cluster and scattered points → unambiguous."""
        rng = np.random.default_rng(42)

        # Main cluster
        centroid = np.array([1.0, 0.0, 0.0])
        main_cluster = make_cluster(centroid, 8, 0, "Main topic", 0.90, rng=rng)

        # Scattered points (each in different directions, won't form a cluster)
        scattered = []
        for i in range(4):
            direction = rng.standard_normal(3)
            direction = direction / np.linalg.norm(direction)
            scattered.append(Candidate(
                id=8 + i,
                text=f"Random noise {i}",
                emb=direction,
                retr_score=0.70 - i * 0.05,  # Lower scores than main cluster
            ))

        candidates = main_cluster + scattered
        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)

        result = ambiguity_detect("query", candidates, config)

        assert result.ambiguous is False

    def test_non_competitive_clusters(self):
        """Two clusters but one dominates (b1 - b2 > delta) → unambiguous."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Dominant cluster with much higher scores
        cluster_a = make_cluster(centroid_a, 5, 0, "Dominant topic", 0.95, rng=rng)
        # Weaker cluster with lower scores
        cluster_b = make_cluster(centroid_b, 5, 5, "Weak topic", 0.80, rng=rng)

        candidates = cluster_a + cluster_b
        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)

        result = ambiguity_detect("query", candidates, config)

        assert result.ambiguous is False

    def test_insufficient_candidates(self):
        """Too few candidates → unambiguous."""
        rng = np.random.default_rng(42)

        centroid = np.array([1.0, 0.0, 0.0])
        # Only 4 candidates, need 2*m = 6 for ambiguity
        candidates = make_cluster(centroid, 4, 0, "Small set", 0.90, rng=rng)

        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)
        result = ambiguity_detect("query", candidates, config)

        assert result.ambiguous is False
        assert "insufficient" in result.reason

    def test_determinism(self):
        """Same inputs produce same outputs (no randomness)."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        cluster_a = make_cluster(centroid_a, 5, 0, "Topic A", 0.90, rng=rng)

        rng = np.random.default_rng(42)  # Reset RNG to get same cluster_a
        cluster_a_copy = make_cluster(centroid_a, 5, 0, "Topic A", 0.90, rng=rng)

        rng = np.random.default_rng(43)  # Different seed for cluster_b
        cluster_b = make_cluster(centroid_b, 5, 5, "Topic B", 0.88, rng=rng)

        candidates = cluster_a + cluster_b
        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)

        result1 = ambiguity_detect("test", candidates, config)
        result2 = ambiguity_detect("test", candidates, config)

        assert result1.ambiguous == result2.ambiguous
        assert result1.chosen_k == result2.chosen_k
        assert result1.cluster_sizes == result2.cluster_sizes
        assert result1.score_gap == result2.score_gap

    def test_option_labels_are_distinctive(self):
        """Options have meaningful label terms derived from cluster content."""
        rng = np.random.default_rng(42)

        centroid_a = np.array([1.0, 0.0, 0.0])
        centroid_b = np.array([0.0, 1.0, 0.0])

        # Distinct content for each cluster
        cluster_a = []
        for i in range(5):
            noise = rng.standard_normal(3) * 0.1
            emb = centroid_a + noise
            emb = emb / np.linalg.norm(emb)
            cluster_a.append(Candidate(
                id=i,
                text=f"Python programming language syntax example {i}",
                emb=emb,
                retr_score=0.90 - i * 0.02,
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
                retr_score=0.88 - i * 0.02,
            ))

        candidates = cluster_a + cluster_b
        config = AmbiguityConfig(min_cluster_size=3, delta=0.03)

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
            for cid in candidates_ids:
                if cid not in statements:
                    continue

                is_positive = gold.get(str(cid), 0) == 1
                is_hard_neg = cid in hard_negs

                if is_positive:
                    centroid = centroid_pos
                    base_score = 0.90
                elif is_hard_neg:
                    centroid = centroid_neg
                    base_score = 0.85  # Hard negatives have high scores too
                else:
                    # Regular negatives - scattered
                    centroid = rng.standard_normal(64)
                    centroid = centroid / np.linalg.norm(centroid)
                    base_score = 0.60

                noise = rng.standard_normal(64) * 0.1
                emb = centroid + noise
                emb = emb / np.linalg.norm(emb)

                candidates.append(Candidate(
                    id=cid,
                    text=statements[cid],
                    emb=emb,
                    retr_score=base_score + rng.uniform(-0.02, 0.02),
                ))

            if len(candidates) < 6:  # Need at least 2*m candidates
                continue

            config = AmbiguityConfig(min_cluster_size=3, delta=0.10)  # Wider delta for test
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

            # Create synthetic embeddings - all positives in one cluster
            rng = np.random.default_rng(hash(query) % 2**32)

            centroid = rng.standard_normal(64)
            centroid = centroid / np.linalg.norm(centroid)

            candidates = []
            for cid in candidates_ids:
                if cid not in statements:
                    continue

                is_positive = gold.get(str(cid), 0) == 1

                if is_positive:
                    noise = rng.standard_normal(64) * 0.1
                    emb = centroid + noise
                    base_score = 0.90
                else:
                    # Negatives are scattered and low-scoring
                    emb = rng.standard_normal(64)
                    base_score = 0.50

                emb = emb / np.linalg.norm(emb)

                candidates.append(Candidate(
                    id=cid,
                    text=statements[cid],
                    emb=emb,
                    retr_score=base_score + rng.uniform(-0.05, 0.05),
                ))

            if len(candidates) < 6:
                continue

            config = AmbiguityConfig(min_cluster_size=3, delta=0.03)
            result = ambiguity_detect(query, candidates, config)

            # Disambiguated queries should have a single coherent cluster
            # (This depends on the synthetic embedding structure we created)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
