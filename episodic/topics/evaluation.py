"""
Evaluation harness for topic strategies.

This module provides infrastructure for testing and comparing
topic detection strategies against labeled test cases.

Includes operational metrics beyond raw F1:
- Windowed F1 (W-F1): F1 with tolerance window
- BOR: Boundary Oversegmentation Ratio
- Purity/Coverage: Segment quality metrics
- Major-boundary recall: Heuristic-based major boundary detection
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from episodic.topics.strategy import TopicStrategy, Confidence

# Re-exports for backward compatibility
from episodic.topics.eval_models import (
    BoundaryAlignment, ALIGNMENT_PRESETS,
    to_canonical_boundaries, from_canonical_boundaries,
    normalize_strategy_output, Message, EvalCase, BoundaryResult, EvaluationResult,
    is_likely_major_boundary, MAJOR_BOUNDARY_PATTERNS,
)
from episodic.topics.eval_metrics import (
    OperationalMetrics, compute_windowed_metrics, compute_windowed_metrics_one_to_one,
    compute_exact_f1, compute_windowdiff, compute_segmentation_similarity,
    compute_bor, compute_purity_coverage, boundaries_to_segments,
    compute_operational_metrics, aggregate_operational_metrics,
)

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """
    Harness for evaluating topic strategies against test cases.

    Runs strategies, collects metrics, and enables comparison.
    """

    def __init__(self, test_cases: Optional[List[EvalCase]] = None):
        """
        Initialize the evaluation harness.

        Args:
            test_cases: List of test cases to evaluate against
        """
        self.test_cases = test_cases or []
        self.results: List[EvaluationResult] = []

    def add_test_case(self, test_case: EvalCase) -> None:
        """Add a test case to the harness."""
        self.test_cases.append(test_case)

    def load_test_cases(self, path: str) -> None:
        """Load test cases from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        for case_data in data.get('test_cases', []):
            self.test_cases.append(EvalCase.from_dict(case_data))

        logger.info(f"Loaded {len(self.test_cases)} test cases from {path}")

    def save_test_cases(self, path: str) -> None:
        """Save test cases to a JSON file."""
        data = {
            'test_cases': [tc.to_dict() for tc in self.test_cases],
            'saved_at': datetime.now().isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def evaluate_strategy(
        self,
        strategy: TopicStrategy,
        test_case: EvalCase,
        verbose: bool = False,
        strategy_alignment: BoundaryAlignment = None
    ) -> EvaluationResult:
        """
        Evaluate a strategy on a single test case using canonical boundaries.

        Both gold boundaries (from test case) and predicted boundaries (from
        strategy) are converted to canonical representation before comparison.

        Args:
            strategy: The strategy to evaluate
            test_case: The test case to run
            verbose: Print progress
            strategy_alignment: How the strategy reports boundaries
                               (default: user_starts_topic - boundaries detected
                               on user messages where topic changes)

        Returns:
            EvaluationResult with metrics
        """
        boundary_results = []
        total_start = time.time()

        # Convert expected boundaries to canonical form
        canonical_expected = test_case.get_canonical_boundaries()

        # Default strategy alignment: detects on user messages
        if strategy_alignment is None:
            strategy_alignment = ALIGNMENT_PRESETS['user_starts_topic']

        # Track predicted boundaries (in strategy's native format)
        predicted_indices = []

        # Build message history incrementally and check each point
        message_history = []
        detection_results = {}  # Map message index -> detection info

        for i, message in enumerate(test_case.messages):
            # Strategies typically run detection on user messages
            # but we record results for all positions
            if message.role == 'user' and len(message_history) >= 2:
                # Run detection
                start_time = time.time()
                decision = strategy.get_decision(
                    query=message.content,
                    messages=message_history,
                    current_thread=None
                )
                processing_time = (time.time() - start_time) * 1000

                detection_results[i] = {
                    'detected': decision.topic_changed,
                    'confidence': decision.confidence,
                    'confidence_score': decision.confidence_score,
                    'signals': decision.signals,
                    'processing_time_ms': processing_time
                }

                if decision.topic_changed:
                    predicted_indices.append(i)
            else:
                detection_results[i] = {
                    'detected': False,
                    'confidence': Confidence.UNCERTAIN,
                    'confidence_score': 0.0,
                    'signals': {},
                    'processing_time_ms': 0.0
                }

            # Add to history after detection
            message_history.append(message.to_dict())

        # Convert predictions to canonical form
        canonical_predicted = normalize_strategy_output(
            predicted_indices,
            test_case.messages,
            strategy_alignment
        )

        # Create boundary results for all potential boundary positions
        # Canonical boundaries are in range [1, T-1]
        for i in range(1, len(test_case.messages)):
            expected_boundary = i in canonical_expected
            detected_boundary = i in canonical_predicted

            # Get detection info (might be from this position or adjacent)
            det_info = detection_results.get(i, detection_results.get(i-1, {}))

            result = BoundaryResult(
                message_index=i,
                expected_boundary=expected_boundary,
                detected_boundary=detected_boundary,
                confidence=det_info.get('confidence', Confidence.UNCERTAIN),
                confidence_score=det_info.get('confidence_score', 0.0),
                processing_time_ms=det_info.get('processing_time_ms', 0.0),
                signals=det_info.get('signals', {})
            )
            boundary_results.append(result)

            if verbose:
                status = "\u2713" if result.is_correct else "\u2717"
                exp = "B" if expected_boundary else "-"
                det = "B" if detected_boundary else "-"
                conf = det_info.get('confidence_score', 0.0)
                print(f"  [{i}] {status} expected={exp} detected={det} conf={conf:.2f}")

        total_time = (time.time() - total_start) * 1000

        return EvaluationResult(
            test_case_id=test_case.id,
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            boundary_results=boundary_results,
            total_time_ms=total_time
        )

    def evaluate_all(
        self,
        strategy: TopicStrategy,
        verbose: bool = False
    ) -> List[EvaluationResult]:
        """
        Evaluate a strategy on all test cases.

        Args:
            strategy: The strategy to evaluate
            verbose: Print progress

        Returns:
            List of EvaluationResults
        """
        results = []

        for test_case in self.test_cases:
            if verbose:
                print(f"\nEvaluating: {test_case.name}")

            result = self.evaluate_strategy(strategy, test_case, verbose)
            results.append(result)
            self.results.append(result)

        return results

    def compare_strategies(
        self,
        strategies: List[TopicStrategy],
        verbose: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple strategies on all test cases.

        Args:
            strategies: List of strategies to compare
            verbose: Print progress

        Returns:
            Dict mapping strategy names to aggregate metrics
        """
        comparison = {}

        for strategy in strategies:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Strategy: {strategy.name}")
                print('='*50)

            results = self.evaluate_all(strategy, verbose)

            # Aggregate metrics
            total_tp = sum(r.true_positives for r in results)
            total_fp = sum(r.false_positives for r in results)
            total_tn = sum(r.true_negatives for r in results)
            total_fn = sum(r.false_negatives for r in results)
            total_time = sum(r.total_time_ms for r in results)

            tp_fp = total_tp + total_fp
            tp_fn = total_tp + total_fn
            total = total_tp + total_fp + total_tn + total_fn

            precision = total_tp / tp_fp if tp_fp > 0 else 0.0
            recall = total_tp / tp_fn if tp_fn > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (total_tp + total_tn) / total if total > 0 else 0.0

            comparison[strategy.name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'true_negatives': total_tn,
                'false_negatives': total_fn,
                'total_time_ms': total_time,
                'num_test_cases': len(results),
            }

        return comparison

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all results."""
        if not self.results:
            return {}

        # Group by strategy
        by_strategy = {}
        for result in self.results:
            if result.strategy_name not in by_strategy:
                by_strategy[result.strategy_name] = []
            by_strategy[result.strategy_name].append(result)

        aggregate = {}
        for strategy_name, results in by_strategy.items():
            total_tp = sum(r.true_positives for r in results)
            total_fp = sum(r.false_positives for r in results)
            total_tn = sum(r.true_negatives for r in results)
            total_fn = sum(r.false_negatives for r in results)

            tp_fp = total_tp + total_fp
            tp_fn = total_tp + total_fn
            total = total_tp + total_fp + total_tn + total_fn

            aggregate[strategy_name] = {
                'precision': total_tp / tp_fp if tp_fp > 0 else 0.0,
                'recall': total_tp / tp_fn if tp_fn > 0 else 0.0,
                'accuracy': (total_tp + total_tn) / total if total > 0 else 0.0,
                'num_evaluations': len(results),
            }

        return aggregate

    def save_results(self, path: str) -> None:
        """Save evaluation results to a JSON file."""
        data = {
            'results': [r.to_dict() for r in self.results],
            'aggregate': self.get_aggregate_metrics(),
            'saved_at': datetime.now().isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print a summary of evaluation results."""
        aggregate = self.get_aggregate_metrics()

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        for strategy_name, metrics in aggregate.items():
            print(f"\n{strategy_name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Evaluations: {metrics['num_evaluations']}")


def create_simple_test_case(
    id: str,
    name: str,
    topics: List[Tuple[str, List[str]]],
    description: str = ""
) -> EvalCase:
    """
    Helper to create a simple test case from topic segments.

    Args:
        id: Test case ID
        name: Test case name
        topics: List of (topic_name, [user_messages]) tuples
        description: Description of the test case

    Returns:
        EvalCase with messages and expected boundaries
    """
    messages = []
    boundaries = []
    topic_names = {}

    msg_index = 0
    for topic_idx, (topic_name, user_messages) in enumerate(topics):
        # First message of each topic (except first) is a boundary
        if topic_idx > 0:
            boundaries.append(msg_index)
            topic_names[msg_index] = topic_name

        for user_msg in user_messages:
            # Add user message
            messages.append(Message(role='user', content=user_msg))
            msg_index += 1

            # Add placeholder assistant response
            messages.append(Message(role='assistant', content=f"Response about {topic_name}."))
            msg_index += 1

    return EvalCase(
        id=id,
        name=name,
        description=description or f"Test case with {len(topics)} topics",
        messages=messages,
        expected_boundaries=boundaries,
        expected_topic_names=topic_names,
    )
