"""
Tests for reactivation replay harness functionality.

Tests compute_metrics(), export_features(), get_replay_summary(),
and related functions in reactivation_replay.py.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import pytest

from episodic.evaluation.reactivation_replay import (
    ReplayResult,
    ReplayMetrics,
    compute_metrics,
    export_features,
    get_replay_summary,
    _evaluate_correctness,
    _count_thrash_events,
)


class TestReplayResult:
    """Tests for ReplayResult dataclass."""

    def test_create_replay_result(self):
        """Test creating a ReplayResult."""
        result = ReplayResult(
            turn_id="node_123",
            user_content="Hello there",
            ground_truth="continue",
            probe_decision="CONTINUE",
            probe_topic=None,
            correct=True,
            features={"confidence": 0.9}
        )

        assert result.turn_id == "node_123"
        assert result.ground_truth == "continue"
        assert result.probe_decision == "CONTINUE"
        assert result.correct is True
        assert result.features["confidence"] == 0.9


class TestEvaluateCorrectness:
    """Tests for _evaluate_correctness helper."""

    def test_continue_correct(self):
        """Test correct continue decision."""
        assert _evaluate_correctness("CONTINUE", None, "continue") is True

    def test_continue_incorrect_should_reactivate(self):
        """Test incorrect continue when should reactivate."""
        assert _evaluate_correctness("CONTINUE", None, "reactivate:Python") is False

    def test_reactivate_correct_topic(self):
        """Test correct reactivation to right topic."""
        assert _evaluate_correctness("REACTIVATE", "Python", "reactivate:Python") is True

    def test_reactivate_correct_partial_match(self):
        """Test reactivation with partial topic match."""
        assert _evaluate_correctness("REACTIVATE", "Python Programming", "reactivate:Python") is True
        assert _evaluate_correctness("REACTIVATE", "Python", "reactivate:Python Programming") is True

    def test_reactivate_incorrect_topic(self):
        """Test incorrect reactivation to wrong topic."""
        # When probe topic doesn't match at all
        assert _evaluate_correctness("REACTIVATE", "Coffee", "reactivate:Python") is False

    def test_reactivate_when_should_continue(self):
        """Test reactivation when should continue."""
        assert _evaluate_correctness("REACTIVATE", "Python", "continue") is False

    def test_new_topic_should_continue(self):
        """Test new_topic ground truth expects continue."""
        assert _evaluate_correctness("CONTINUE", None, "new_topic") is True
        assert _evaluate_correctness("REACTIVATE", "Python", "new_topic") is False

    def test_no_ground_truth(self):
        """Test when no ground truth is provided."""
        assert _evaluate_correctness("CONTINUE", None, None) is True
        assert _evaluate_correctness("REACTIVATE", "Python", None) is True

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert _evaluate_correctness("CONTINUE", None, "CONTINUE") is True
        assert _evaluate_correctness("REACTIVATE", "Python", "REACTIVATE:PYTHON") is True


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_perfect_score_all_continues(self):
        """Test perfect accuracy with all correct continues."""
        results = [
            ReplayResult("n1", "Hi", "continue", "CONTINUE", None, True),
            ReplayResult("n2", "Hello", "continue", "CONTINUE", None, True),
            ReplayResult("n3", "Hey", "continue", "CONTINUE", None, True),
        ]

        metrics = compute_metrics(results)

        assert metrics.total == 3
        assert metrics.correct == 3
        assert metrics.accuracy == 1.0
        assert metrics.true_negatives == 3
        assert metrics.false_positives == 0
        assert metrics.true_positives == 0
        assert metrics.false_negatives == 0

    def test_perfect_score_all_reactivations(self):
        """Test perfect accuracy with all correct reactivations."""
        results = [
            ReplayResult("n1", "About Python", "reactivate:Python", "REACTIVATE", "Python", True),
            ReplayResult("n2", "More Python", "reactivate:Python", "REACTIVATE", "Python", True),
        ]

        metrics = compute_metrics(results)

        assert metrics.total == 2
        assert metrics.correct == 2
        assert metrics.accuracy == 1.0
        assert metrics.true_positives == 2
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_mixed_results_with_false_positives(self):
        """Test mixed results including false positives."""
        results = [
            # True positive - correctly reactivated
            ReplayResult("n1", "Python", "reactivate:Python", "REACTIVATE", "Python", True),
            # False positive - reactivated when should continue
            ReplayResult("n2", "Weather", "continue", "REACTIVATE", "Python", False),
            # True negative - correctly continued
            ReplayResult("n3", "Hello", "continue", "CONTINUE", None, True),
        ]

        metrics = compute_metrics(results)

        assert metrics.total == 3
        assert metrics.correct == 2
        assert metrics.accuracy == 2/3
        assert metrics.true_positives == 1
        assert metrics.false_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_negatives == 0
        assert metrics.precision == 0.5  # 1 TP / (1 TP + 1 FP)

    def test_missed_resumes_counted(self):
        """Test that missed resumes (false negatives) are counted."""
        results = [
            # False negative - should have reactivated but continued
            ReplayResult("n1", "Python", "reactivate:Python", "CONTINUE", None, False),
            # Another missed resume
            ReplayResult("n2", "Python", "reactivate:Python", "CONTINUE", None, False),
        ]

        metrics = compute_metrics(results)

        assert metrics.false_negatives == 2
        assert metrics.missed_resumes == 2
        assert metrics.recall == 0.0  # 0 TP / (0 TP + 2 FN)

    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 calculations."""
        results = [
            # 2 true positives
            ReplayResult("n1", "Py1", "reactivate:Python", "REACTIVATE", "Python", True),
            ReplayResult("n2", "Py2", "reactivate:Python", "REACTIVATE", "Python", True),
            # 1 false positive
            ReplayResult("n3", "X", "continue", "REACTIVATE", "Python", False),
            # 1 false negative
            ReplayResult("n4", "Py3", "reactivate:Python", "CONTINUE", None, False),
            # 1 true negative
            ReplayResult("n5", "Hi", "continue", "CONTINUE", None, True),
        ]

        metrics = compute_metrics(results)

        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
        assert abs(metrics.precision - 2/3) < 0.01
        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
        assert abs(metrics.recall - 2/3) < 0.01
        # F1 = 2 * P * R / (P + R) = 2 * (2/3) * (2/3) / (4/3) = 2/3
        assert abs(metrics.f1 - 2/3) < 0.01

    def test_by_topic_breakdown(self):
        """Test per-topic breakdown."""
        results = [
            ReplayResult("n1", "Py", "reactivate:Python", "REACTIVATE", "Python", True),
            ReplayResult("n2", "Py", "reactivate:Python", "REACTIVATE", "Python", True),
            ReplayResult("n3", "Js", "reactivate:JavaScript", "REACTIVATE", "JavaScript", True),
            ReplayResult("n4", "X", "continue", "REACTIVATE", "Python", False),
        ]

        metrics = compute_metrics(results)

        assert "Python" in metrics.by_topic
        assert "JavaScript" in metrics.by_topic
        assert metrics.by_topic["Python"]["tp"] == 2
        assert metrics.by_topic["Python"]["fp"] == 1
        assert metrics.by_topic["JavaScript"]["tp"] == 1

    def test_empty_results(self):
        """Test with empty results list."""
        metrics = compute_metrics([])

        assert metrics.total == 0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0


class TestCountThrashEvents:
    """Tests for _count_thrash_events helper."""

    def test_no_thrash_all_continues(self):
        """Test no thrash events with all continues."""
        results = [
            ReplayResult("n1", "", "", "CONTINUE", None, True),
            ReplayResult("n2", "", "", "CONTINUE", None, True),
            ReplayResult("n3", "", "", "CONTINUE", None, True),
        ]

        count = _count_thrash_events(results, window=3)
        assert count == 0

    def test_no_thrash_same_topic_reactivations(self):
        """Test no thrash when reactivating to same topic."""
        results = [
            ReplayResult("n1", "", "", "REACTIVATE", "Python", True),
            ReplayResult("n2", "", "", "CONTINUE", None, True),
            ReplayResult("n3", "", "", "REACTIVATE", "Python", True),
        ]

        count = _count_thrash_events(results, window=3)
        assert count == 0

    def test_thrash_detected_different_topics(self):
        """Test thrash detected with different topic reactivations."""
        results = [
            ReplayResult("n1", "", "", "REACTIVATE", "Python", True),
            ReplayResult("n2", "", "", "REACTIVATE", "JavaScript", True),
            ReplayResult("n3", "", "", "REACTIVATE", "Python", True),
        ]

        count = _count_thrash_events(results, window=3)
        assert count > 0

    def test_thrash_outside_window(self):
        """Test no thrash when switches are outside window."""
        results = [
            ReplayResult("n1", "", "", "REACTIVATE", "Python", True),
            ReplayResult("n2", "", "", "CONTINUE", None, True),
            ReplayResult("n3", "", "", "CONTINUE", None, True),
            ReplayResult("n4", "", "", "CONTINUE", None, True),
            ReplayResult("n5", "", "", "REACTIVATE", "JavaScript", True),
        ]

        # With window=3, Python and JavaScript reactivations are too far apart
        count = _count_thrash_events(results, window=3)
        # First few turns: only Python in window
        # Later turns: only JavaScript in window
        assert count == 0

    def test_thrash_rate_in_metrics(self):
        """Test thrash rate is computed in metrics."""
        results = [
            ReplayResult("n1", "", "", "REACTIVATE", "Python", True),
            ReplayResult("n2", "", "", "REACTIVATE", "JavaScript", True),
        ]

        metrics = compute_metrics(results)
        assert metrics.thrash_rate >= 0


class TestExportFeatures:
    """Tests for export_features()."""

    def test_export_creates_valid_jsonl(self):
        """Test that export creates valid JSONL file."""
        results = [
            ReplayResult(
                turn_id="n1",
                user_content="Hello",
                ground_truth="continue",
                probe_decision="CONTINUE",
                probe_topic=None,
                correct=True,
                features={"confidence": 0.9, "best_similarity": 0.5}
            ),
            ReplayResult(
                turn_id="n2",
                user_content="Python",
                ground_truth="reactivate:Python",
                probe_decision="REACTIVATE",
                probe_topic="Python",
                correct=True,
                features={"confidence": 0.85, "dormancy_turns": 5}
            ),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name

        try:
            export_features(results, output_path)

            # Read and verify
            with open(output_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 2

            record1 = json.loads(lines[0])
            assert record1['turn_id'] == 'n1'
            assert record1['ground_truth'] == 'continue'
            assert record1['probe_decision'] == 'CONTINUE'
            assert record1['correct'] is True
            assert record1['confidence'] == 0.9

            record2 = json.loads(lines[1])
            assert record2['turn_id'] == 'n2'
            assert record2['probe_topic'] == 'Python'
            assert record2['dormancy_turns'] == 5

        finally:
            os.unlink(output_path)

    def test_export_empty_results(self):
        """Test exporting empty results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name

        try:
            export_features([], output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            assert content == ""
        finally:
            os.unlink(output_path)


class TestGetReplaySummary:
    """Tests for get_replay_summary()."""

    def test_summary_contains_key_metrics(self):
        """Test that summary contains all key metrics."""
        metrics = ReplayMetrics(
            total=100,
            correct=85,
            accuracy=0.85,
            true_positives=20,
            false_positives=5,
            true_negatives=65,
            false_negatives=10,
            precision=0.8,
            recall=0.667,
            f1=0.727,
            thrash_rate=0.02,
            missed_resumes=10,
            by_topic={"Python": {"tp": 15, "fp": 3, "fn": 5}}
        )

        summary = get_replay_summary(metrics)

        assert "100" in summary  # Total turns
        assert "85" in summary   # Correct
        assert "85.0%" in summary or "85%" in summary  # Accuracy
        assert "True Positives" in summary
        assert "False Positives" in summary
        assert "True Negatives" in summary
        assert "False Negatives" in summary
        assert "Precision" in summary
        assert "Recall" in summary
        assert "F1" in summary
        assert "Thrash Rate" in summary
        assert "Missed Resumes" in summary

    def test_summary_includes_by_topic(self):
        """Test that summary includes per-topic breakdown."""
        metrics = ReplayMetrics(
            total=10,
            correct=8,
            accuracy=0.8,
            true_positives=5,
            false_positives=1,
            true_negatives=3,
            false_negatives=1,
            precision=0.833,
            recall=0.833,
            f1=0.833,
            thrash_rate=0.0,
            missed_resumes=1,
            by_topic={
                "Python": {"tp": 3, "fp": 1, "fn": 1},
                "JavaScript": {"tp": 2, "fp": 0, "fn": 0}
            }
        )

        summary = get_replay_summary(metrics)

        assert "By Topic" in summary
        assert "Python" in summary
        assert "JavaScript" in summary

    def test_summary_handles_empty_by_topic(self):
        """Test summary handles empty by_topic dict."""
        metrics = ReplayMetrics(
            total=5,
            correct=5,
            accuracy=1.0,
            true_positives=0,
            false_positives=0,
            true_negatives=5,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            thrash_rate=0.0,
            missed_resumes=0,
            by_topic={}
        )

        summary = get_replay_summary(metrics)

        # Should not crash and should have basic metrics
        assert "Total turns: 5" in summary
        assert "By Topic" not in summary or summary.count("By Topic") == 1
