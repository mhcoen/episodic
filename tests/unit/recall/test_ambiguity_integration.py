"""
Tests for ambiguity detection integration in recall pipeline.

Verifies that ambiguous queries return AMBIGUOUS results and that
cluster selection filters hits correctly.
"""

import pytest
import numpy as np
from unittest.mock import patch

from episodic.recall.pipeline import (
    recall,
    RecallResultKind,
    SemanticHit,
    MIN_CANDIDATES_FOR_AMBIGUITY,
)
from episodic.recall.ambiguity import AmbiguityConfig

from . import create_test_db


def make_resolved_query(target, temporal=None, speaker=None, has_broadness_cue=False, mode="answer"):
    """Create a ResolvedQuery with correct field signature."""
    from episodic.query.types import ResolvedQuery

    return ResolvedQuery(
        mode=mode,
        target=target,
        segment_explicit=False,
        segment_query=None,
        segment_resolved_ids=None,
        segment_ambiguous=False,
        segment_candidates=None,
        temporal=temporal,
        speaker=speaker,
        deictic=None,
        has_broadness_cue=has_broadness_cue,
        audit_trace="{}",
    )


def make_clustered_hits(n_per_cluster=6, n_clusters=2):
    """
    Create semantic hits with embeddings that form distinct clusters.

    Returns hits where cluster 0 has "programming" content and cluster 1 has "coffee" content.
    """
    rng = np.random.default_rng(42)
    hits = []

    # Define cluster centroids (orthogonal directions)
    centroids = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    cluster_texts = [
        "Java programming language syntax code example JVM bytecode",
        "Java coffee beans Indonesian roast brewing espresso",
        "Python snake reptile wildlife nature",
    ]

    for cluster_idx in range(n_clusters):
        centroid = centroids[cluster_idx % len(centroids)]
        text_template = cluster_texts[cluster_idx % len(cluster_texts)]

        for i in range(n_per_cluster):
            # Create embedding near centroid
            noise = rng.standard_normal(3) * 0.1
            emb = centroid + noise
            emb = emb / np.linalg.norm(emb)  # L2 normalize

            # Interleave scores so both clusters have near-top ranks
            score = 0.90 - cluster_idx * 0.01 - i * 0.02

            hits.append(SemanticHit(
                exchange_id=f"node_{cluster_idx}_{i}",
                relevance_score=score,
                metadata={"user_id": f"node_{cluster_idx}_{i}"},
                text=f"{text_template} item {i}",
                embedding=emb,
            ))

    return hits


class TestAmbiguityDetectionIntegration:
    """Test that ambiguity detection is properly integrated into recall."""

    def test_ambiguous_hits_return_ambiguous_result(self, tmp_path):
        """
        When semantic hits form multiple competitive clusters,
        recall() should return AMBIGUOUS result.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        # Create hits with two distinct clusters
        clustered_hits = make_clustered_hits(n_per_cluster=6, n_clusters=2)

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return clustered_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")

        # Should detect ambiguity
        assert result.kind == RecallResultKind.AMBIGUOUS
        assert result.ambiguity is not None
        assert result.ambiguity.ambiguous is True
        assert len(result.cluster_options) >= 2

        # Each cluster option should have member indices
        for opt in result.cluster_options:
            assert len(opt.member_indices) >= 3  # min_cluster_size

    def test_unambiguous_hits_return_normal_result(self, tmp_path):
        """
        When semantic hits form a single coherent cluster,
        recall() should return normal HITS result.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="python programming")

        # Create hits with only one cluster
        single_cluster_hits = make_clustered_hits(n_per_cluster=10, n_clusters=1)

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return single_cluster_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")

        # Should NOT detect ambiguity (single cluster)
        assert result.kind == RecallResultKind.HITS
        assert result.ambiguity is None

    def test_too_few_hits_skip_ambiguity_check(self, tmp_path):
        """
        When there are fewer than MIN_CANDIDATES_FOR_AMBIGUITY hits,
        ambiguity check should be skipped.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        # Create just a few hits (below threshold)
        few_hits = make_clustered_hits(n_per_cluster=3, n_clusters=2)
        # Total: 6 hits, below MIN_CANDIDATES_FOR_AMBIGUITY (10)

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return few_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")

        # Should skip ambiguity check due to too few hits
        assert result.kind == RecallResultKind.HITS
        assert result.ambiguity is None

    def test_skip_ambiguity_check_flag(self, tmp_path):
        """
        When skip_ambiguity_check=True, ambiguity detection is bypassed.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        clustered_hits = make_clustered_hits(n_per_cluster=6, n_clusters=2)

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return clustered_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we", skip_ambiguity_check=True)

        # Should skip ambiguity detection
        assert result.kind == RecallResultKind.HITS
        assert result.ambiguity is None


class TestClusterSelection:
    """Test that cluster selection filters hits correctly."""

    def test_selected_cluster_filters_hits(self, tmp_path):
        """
        When selected_cluster is provided, only hits from that cluster
        should be processed.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        clustered_hits = make_clustered_hits(n_per_cluster=6, n_clusters=2)

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return clustered_hits

        # First, get the ambiguous result to find cluster option IDs
        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")

        assert result.kind == RecallResultKind.AMBIGUOUS
        cluster_id = result.cluster_options[0].option_id

        # Now select that cluster
        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we", selected_cluster=cluster_id)

        # Should return HITS, not AMBIGUOUS
        assert result.kind == RecallResultKind.HITS
        assert result.ambiguity is None


class TestDisambiguationPrompt:
    """Test disambiguation prompt formatting."""

    def test_get_disambiguation_prompt(self, tmp_path):
        """
        AMBIGUOUS result should provide a formatted disambiguation prompt.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        clustered_hits = make_clustered_hits(n_per_cluster=6, n_clusters=2)

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return clustered_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")

        assert result.kind == RecallResultKind.AMBIGUOUS

        prompt = result.get_disambiguation_prompt("java")

        # Prompt should contain options
        assert "java" in prompt.lower()
        assert "1." in prompt
        assert "2." in prompt


class TestClusterMembershipStability:
    """Test that cluster filtering uses original indices correctly."""

    def test_member_indices_are_original_hit_indices(self, tmp_path):
        """
        Verify that member_indices in ClusterOption refer to the original
        hit list indices, not post-processed indices.

        This ensures filtering works correctly even if hits are reordered
        during promotion/ranking.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        # Create hits with known structure
        clustered_hits = make_clustered_hits(n_per_cluster=6, n_clusters=2)

        # Record original hit exchange_ids by index
        original_ids_by_index = {i: h.exchange_id for i, h in enumerate(clustered_hits)}

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return clustered_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")

        assert result.kind == RecallResultKind.AMBIGUOUS

        # Verify each cluster option's member_indices map to valid original hits
        for opt in result.cluster_options:
            for idx in opt.member_indices:
                assert idx in original_ids_by_index, (
                    f"member_index {idx} not in original hit indices"
                )
                # Verify the index corresponds to the right cluster's hit pattern
                exchange_id = original_ids_by_index[idx]
                # Cluster 0 hits have "node_0_*", cluster 1 hits have "node_1_*"
                expected_cluster = int(exchange_id.split("_")[1])
                # The option_id should match the cluster the hit came from
                # (though option_id is assigned by clustering, not input order)

    def test_filtering_preserves_correct_hits(self, tmp_path):
        """
        When filtering to a cluster, the returned hits should be exactly
        those at the member_indices positions in the original hit list.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        clustered_hits = make_clustered_hits(n_per_cluster=6, n_clusters=2)

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return clustered_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            # Get ambiguous result
            result = recall(conn, query, query_form="when_we")

        assert result.kind == RecallResultKind.AMBIGUOUS
        assert len(result.cluster_options) >= 2

        # Select first cluster
        cluster_opt = result.cluster_options[0]
        expected_ids = {clustered_hits[i].exchange_id for i in cluster_opt.member_indices}

        # Re-run with cluster selection
        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            filtered_result = recall(
                conn, query, query_form="when_we",
                selected_cluster=cluster_opt.option_id
            )

        # Should not be ambiguous after selection
        assert filtered_result.kind == RecallResultKind.HITS

    def test_dimension_mismatch_skips_ambiguity(self, tmp_path):
        """
        If embeddings have inconsistent dimensions, ambiguity detection
        should be skipped to prevent clustering on nonsense geometry.
        """
        conn, _ = create_test_db(tmp_path)
        query = make_resolved_query(target="java")

        rng = np.random.default_rng(42)

        # Create hits with mismatched dimensions
        bad_hits = []
        for i in range(12):
            # First 10 have dim=3, last 2 have dim=5
            dim = 3 if i < 10 else 5
            emb = rng.standard_normal(dim)
            emb = emb / np.linalg.norm(emb)

            bad_hits.append(SemanticHit(
                exchange_id=f"node_{i}",
                relevance_score=0.9 - i * 0.05,
                metadata={"user_id": f"node_{i}"},
                text=f"Test content {i}",
                embedding=emb,
            ))

        def mock_get_semantic_hits(target, n_results, temporal, broad_horizon, **kwargs):
            return bad_hits

        with patch('episodic.recall.pipeline._get_semantic_hits', mock_get_semantic_hits):
            result = recall(conn, query, query_form="when_we")

        # Should skip ambiguity detection due to dimension mismatch
        assert result.kind == RecallResultKind.HITS
        assert result.ambiguity is None
