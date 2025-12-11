"""
Evaluation harness for topic strategies.

This module provides infrastructure for testing and comparing
topic detection strategies against labeled test cases.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from episodic.topics.strategy import TopicStrategy, TopicDecision, Confidence

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a test conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'node_id': self.node_id,
            **self.metadata
        }


@dataclass
class TestCase:
    """
    A labeled test case for evaluating topic detection.

    Contains a conversation with labeled topic boundaries and
    expected retrieval behavior.
    """
    id: str
    name: str
    description: str
    messages: List[Message]

    # Expected topic boundaries (indices of messages that start new topics)
    expected_boundaries: List[int]

    # Expected topic names at each boundary (optional)
    expected_topic_names: Dict[int, str] = field(default_factory=dict)

    # For retrieval testing: query and expected thread to retrieve
    retrieval_tests: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "synthetic"  # synthetic, real, imported

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'messages': [m.to_dict() for m in self.messages],
            'expected_boundaries': self.expected_boundaries,
            'expected_topic_names': self.expected_topic_names,
            'retrieval_tests': self.retrieval_tests,
            'tags': self.tags,
            'difficulty': self.difficulty,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        messages = [
            Message(
                role=m['role'],
                content=m['content'],
                node_id=m.get('node_id'),
                metadata={k: v for k, v in m.items() if k not in ['role', 'content', 'node_id']}
            )
            for m in data['messages']
        ]
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            messages=messages,
            expected_boundaries=data['expected_boundaries'],
            expected_topic_names=data.get('expected_topic_names', {}),
            retrieval_tests=data.get('retrieval_tests', []),
            tags=data.get('tags', []),
            difficulty=data.get('difficulty', 'medium'),
            source=data.get('source', 'synthetic'),
        )


@dataclass
class BoundaryResult:
    """Result of boundary detection for a single message."""
    message_index: int
    expected_boundary: bool
    detected_boundary: bool
    confidence: Confidence
    confidence_score: float
    processing_time_ms: float
    signals: Dict[str, float]

    @property
    def is_correct(self) -> bool:
        return self.expected_boundary == self.detected_boundary

    @property
    def is_true_positive(self) -> bool:
        return self.expected_boundary and self.detected_boundary

    @property
    def is_false_positive(self) -> bool:
        return not self.expected_boundary and self.detected_boundary

    @property
    def is_true_negative(self) -> bool:
        return not self.expected_boundary and not self.detected_boundary

    @property
    def is_false_negative(self) -> bool:
        return self.expected_boundary and not self.detected_boundary


@dataclass
class EvaluationResult:
    """Result of evaluating a strategy on a test case."""
    test_case_id: str
    strategy_name: str
    strategy_version: str
    boundary_results: List[BoundaryResult]
    total_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Computed metrics
    @property
    def true_positives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_true_positive)

    @property
    def false_positives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_false_positive)

    @property
    def true_negatives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_true_negative)

    @property
    def false_negatives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_false_negative)

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)"""
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)"""
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / total"""
        total = len(self.boundary_results)
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_case_id': self.test_case_id,
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'total_time_ms': self.total_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'metrics': {
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'true_negatives': self.true_negatives,
                'false_negatives': self.false_negatives,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'accuracy': self.accuracy,
            },
            'boundary_results': [
                {
                    'message_index': r.message_index,
                    'expected': r.expected_boundary,
                    'detected': r.detected_boundary,
                    'correct': r.is_correct,
                    'confidence': r.confidence.value,
                    'confidence_score': r.confidence_score,
                    'processing_time_ms': r.processing_time_ms,
                }
                for r in self.boundary_results
            ]
        }


class EvaluationHarness:
    """
    Harness for evaluating topic strategies against test cases.

    Runs strategies, collects metrics, and enables comparison.
    """

    def __init__(self, test_cases: Optional[List[TestCase]] = None):
        """
        Initialize the evaluation harness.

        Args:
            test_cases: List of test cases to evaluate against
        """
        self.test_cases = test_cases or []
        self.results: List[EvaluationResult] = []

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the harness."""
        self.test_cases.append(test_case)

    def load_test_cases(self, path: str) -> None:
        """Load test cases from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        for case_data in data.get('test_cases', []):
            self.test_cases.append(TestCase.from_dict(case_data))

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
        test_case: TestCase,
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a strategy on a single test case.

        Args:
            strategy: The strategy to evaluate
            test_case: The test case to run
            verbose: Print progress

        Returns:
            EvaluationResult with metrics
        """
        boundary_results = []
        total_start = time.time()

        # Convert expected boundaries to a set for O(1) lookup
        expected_boundary_set = set(test_case.expected_boundaries)

        # Build message history incrementally and check each point
        message_history = []

        for i, message in enumerate(test_case.messages):
            # Skip assistant messages for boundary detection
            # (boundaries are detected on user messages)
            if message.role != 'user':
                message_history.append(message.to_dict())
                continue

            # Determine if this is an expected boundary
            expected_boundary = i in expected_boundary_set

            # Run detection
            start_time = time.time()

            if len(message_history) < 2:
                # Not enough history for detection
                detected_boundary = False
                confidence = Confidence.UNCERTAIN
                confidence_score = 0.0
                signals = {}
            else:
                decision = strategy.get_decision(
                    query=message.content,
                    messages=message_history,
                    current_thread=None
                )
                detected_boundary = decision.topic_changed
                confidence = decision.confidence
                confidence_score = decision.confidence_score
                signals = decision.signals

            processing_time = (time.time() - start_time) * 1000

            # Record result
            result = BoundaryResult(
                message_index=i,
                expected_boundary=expected_boundary,
                detected_boundary=detected_boundary,
                confidence=confidence,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                signals=signals
            )
            boundary_results.append(result)

            if verbose:
                status = "✓" if result.is_correct else "✗"
                exp = "B" if expected_boundary else "-"
                det = "B" if detected_boundary else "-"
                print(f"  [{i}] {status} expected={exp} detected={det} conf={confidence_score:.2f}")

            # Add to history after detection
            message_history.append(message.to_dict())

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
) -> TestCase:
    """
    Helper to create a simple test case from topic segments.

    Args:
        id: Test case ID
        name: Test case name
        topics: List of (topic_name, [user_messages]) tuples
        description: Description of the test case

    Returns:
        TestCase with messages and expected boundaries
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

    return TestCase(
        id=id,
        name=name,
        description=description or f"Test case with {len(topics)} topics",
        messages=messages,
        expected_boundaries=boundaries,
        expected_topic_names=topic_names,
    )
