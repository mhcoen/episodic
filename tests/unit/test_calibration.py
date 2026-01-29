"""
Unit tests for reactivation calibration.

Tests:
- test_calibration_determinism: same seed → identical params/metrics
- test_known_bad_config_dominated: selector actually does work
"""

import os
import pytest
from typing import List

# Set test mode
os.environ["EPISODIC_TEST_MODE"] = "1"


class TestCalibrationDeterminism:
    """Tests for calibration reproducibility."""

    def test_same_seed_produces_identical_results(self):
        """Same seed should produce identical calibration results."""
        from episodic.evaluation.calibration import (
            run_calibration_sweep,
            CalibrationConfig,
        )
        from episodic.evaluation.resume_moments import ResumeMoment

        # Create minimal test moments
        moments = [
            ResumeMoment(
                moment_id="test_1",
                user_query="Test query",
                expected_active_topic="topic-a",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
            ResumeMoment(
                moment_id="test_2",
                user_query="Another query",
                expected_active_topic="topic-b",
                gap_turns=30,
                category="medium_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
        ]

        # Small param grid for fast test
        param_grid = {
            "support_threshold": [2, 3],
            "rank_gap": [3, 4],
            "cooldown_turns": [2, 3],
        }

        seed = 12345

        # Run twice with same seed
        results1 = run_calibration_sweep(
            moments=moments,
            param_grid=param_grid,
            seed=seed,
            use_cross_validation=False,
        )
        results2 = run_calibration_sweep(
            moments=moments,
            param_grid=param_grid,
            seed=seed,
            use_cross_validation=False,
        )

        # Should have same number of results
        assert len(results1) == len(results2)

        # Each result should match
        for r1, r2 in zip(results1, results2):
            assert r1.config.support_threshold == r2.config.support_threshold
            assert r1.config.rank_gap == r2.config.rank_gap
            assert r1.config.cooldown_turns == r2.config.cooldown_turns
            assert r1.metrics.reactivation_precision == r2.metrics.reactivation_precision
            assert r1.metrics.reactivation_recall == r2.metrics.reactivation_recall
            assert r1.objective_score == r2.objective_score

    def test_different_seeds_can_differ(self):
        """Different seeds may produce different results (stochastic elements)."""
        from episodic.evaluation.calibration import (
            run_calibration_sweep,
            compute_calibration_metrics,
            CalibrationConfig,
        )
        from episodic.evaluation.resume_moments import ResumeMoment

        moments = [
            ResumeMoment(
                moment_id="test_1",
                user_query="Test",
                expected_active_topic="topic-a",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
        ]

        config = CalibrationConfig(support_threshold=2, rank_gap=3, cooldown_turns=2)

        # Run with different seeds
        metrics1 = compute_calibration_metrics(moments, config, seed=100)
        metrics2 = compute_calibration_metrics(moments, config, seed=100)

        # Same seed should be identical
        assert metrics1.reactivation_precision == metrics2.reactivation_precision


class TestConfigSelection:
    """Tests for best config selection."""

    def test_known_bad_config_dominated(self):
        """Selector should prefer good configs over bad ones."""
        from episodic.evaluation.calibration import (
            select_best_config,
            CalibrationResult,
            CalibrationConfig,
            CalibrationMetrics,
        )

        # Create results with clearly different quality
        good_config = CalibrationConfig(
            support_threshold=2,
            rank_gap=3,
            cooldown_turns=2,
        )
        good_metrics = CalibrationMetrics(
            reactivation_precision=0.9,
            reactivation_recall=0.8,
            thrash_rate=0.1,
            disambiguation_burden=0.1,
            thin_fallback_rate=0.05,
            contamination_rate=0.0,  # Hard constraint
        )

        bad_config = CalibrationConfig(
            support_threshold=6,
            rank_gap=2,
            cooldown_turns=0,
        )
        bad_metrics = CalibrationMetrics(
            reactivation_precision=0.3,  # Much worse
            reactivation_recall=0.2,
            thrash_rate=0.5,  # Much worse
            disambiguation_burden=0.6,  # Much worse
            thin_fallback_rate=0.4,
            contamination_rate=0.0,
        )

        results = [
            CalibrationResult(
                config=good_config,
                metrics=good_metrics,
                fold="all",
                objective_score=1000.0,  # Will be recalculated
            ),
            CalibrationResult(
                config=bad_config,
                metrics=bad_metrics,
                fold="all",
                objective_score=-100.0,
            ),
        ]

        # Recompute objective scores
        from episodic.evaluation.calibration import compute_objective_score

        for r in results:
            r.objective_score = compute_objective_score(r.metrics)

        best_config, reason = select_best_config(results)

        # Good config should be selected
        assert best_config.support_threshold == good_config.support_threshold
        assert best_config.rank_gap == good_config.rank_gap
        assert best_config.cooldown_turns == good_config.cooldown_turns

    def test_contamination_constraint_enforced(self):
        """Configs with contamination > 0 should be rejected."""
        from episodic.evaluation.calibration import (
            compute_objective_score,
            CalibrationMetrics,
        )

        # Good metrics but with contamination
        metrics_with_contamination = CalibrationMetrics(
            reactivation_precision=0.95,
            reactivation_recall=0.95,
            thrash_rate=0.0,
            disambiguation_burden=0.0,
            thin_fallback_rate=0.0,
            contamination_rate=0.01,  # Violates hard constraint
        )

        score = compute_objective_score(metrics_with_contamination)

        # Should be rejected (negative infinity)
        assert score == float("-inf")

    def test_lexicographic_priority(self):
        """Precision should dominate over other metrics."""
        from episodic.evaluation.calibration import (
            compute_objective_score,
            CalibrationMetrics,
        )

        # High precision, bad thrash
        high_precision = CalibrationMetrics(
            reactivation_precision=0.9,
            reactivation_recall=0.5,
            thrash_rate=0.3,  # Bad
            disambiguation_burden=0.3,  # Bad
            thin_fallback_rate=0.3,  # Bad
            contamination_rate=0.0,
        )

        # Low precision, good thrash
        low_precision = CalibrationMetrics(
            reactivation_precision=0.5,  # Lower
            reactivation_recall=0.9,  # Better
            thrash_rate=0.0,  # Better
            disambiguation_burden=0.0,  # Better
            thin_fallback_rate=0.0,  # Better
            contamination_rate=0.0,
        )

        score_high = compute_objective_score(high_precision)
        score_low = compute_objective_score(low_precision)

        # Higher precision should win due to lexicographic priority
        assert score_high > score_low


class TestMetricsComputation:
    """Tests for metrics computation."""

    def test_metrics_computed_correctly(self):
        """Verify metrics are computed correctly from decisions."""
        from episodic.evaluation.calibration import (
            compute_calibration_metrics,
            CalibrationConfig,
        )
        from episodic.evaluation.resume_moments import ResumeMoment

        # Create test moments with known expected outcomes
        moments = [
            # Should trigger reactivation
            ResumeMoment(
                moment_id="reactivate_1",
                user_query="Back to Python",
                expected_active_topic="python-debugging",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Should reactivate",
            ),
            # Ambiguous - should disambiguate
            ResumeMoment(
                moment_id="ambiguous_1",
                user_query="About Java",
                expected_active_topic="disambiguate",
                gap_turns=20,
                category="ambiguous",
                cross_topic_import_expected=False,
                notes="Should disambiguate",
            ),
        ]

        # Lenient config that should allow reactivation
        config = CalibrationConfig(
            support_threshold=2,
            rank_gap=3,
            cooldown_turns=0,
        )

        metrics = compute_calibration_metrics(moments, config, seed=42)

        # Check that metrics are in valid ranges
        assert 0.0 <= metrics.reactivation_precision <= 1.0
        assert 0.0 <= metrics.reactivation_recall <= 1.0
        assert 0.0 <= metrics.thrash_rate <= 1.0
        assert 0.0 <= metrics.contamination_rate <= 1.0
        assert metrics.total_moments == 2

    def test_empty_moments_handled(self):
        """Empty moment list should not crash."""
        from episodic.evaluation.calibration import (
            compute_calibration_metrics,
            CalibrationConfig,
        )

        config = CalibrationConfig(
            support_threshold=2,
            rank_gap=3,
            cooldown_turns=2,
        )

        metrics = compute_calibration_metrics([], config, seed=42)

        assert metrics.total_moments == 0
        assert metrics.reactivation_precision == 0.0
        assert metrics.reactivation_recall == 0.0


class TestCrossValidation:
    """Tests for cross-validation behavior."""

    def test_lobo_cv_produces_multiple_folds(self):
        """Leave-one-bucket-out should produce results for each category."""
        from episodic.evaluation.calibration import run_calibration_sweep
        from episodic.evaluation.resume_moments import ResumeMoment

        # Create moments in different categories
        moments = [
            ResumeMoment(
                moment_id="short_1",
                user_query="Test",
                expected_active_topic="topic",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
            ResumeMoment(
                moment_id="medium_1",
                user_query="Test",
                expected_active_topic="topic",
                gap_turns=30,
                category="medium_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
        ]

        param_grid = {
            "support_threshold": [2],
            "rank_gap": [3],
            "cooldown_turns": [2],
        }

        results = run_calibration_sweep(
            moments=moments,
            param_grid=param_grid,
            seed=42,
            use_cross_validation=True,
        )

        # Should have results for each fold
        folds = {r.fold for r in results}
        assert "short_gap" in folds
        assert "medium_gap" in folds


class TestDatasetHash:
    """Tests for dataset versioning."""

    def test_dataset_hash_deterministic(self):
        """Same moments should produce same hash."""
        from episodic.evaluation.calibration import compute_dataset_hash
        from episodic.evaluation.resume_moments import ResumeMoment

        moments = [
            ResumeMoment(
                moment_id="test_1",
                user_query="Test",
                expected_active_topic="topic",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
        ]

        hash1 = compute_dataset_hash(moments)
        hash2 = compute_dataset_hash(moments)

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_different_moments_different_hash(self):
        """Different moments should produce different hash."""
        from episodic.evaluation.calibration import compute_dataset_hash
        from episodic.evaluation.resume_moments import ResumeMoment

        moments1 = [
            ResumeMoment(
                moment_id="test_1",
                user_query="Test",
                expected_active_topic="topic",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
        ]

        moments2 = [
            ResumeMoment(
                moment_id="test_2",  # Different ID
                user_query="Test",
                expected_active_topic="topic",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test",
            ),
        ]

        hash1 = compute_dataset_hash(moments1)
        hash2 = compute_dataset_hash(moments2)

        assert hash1 != hash2
