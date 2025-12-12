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
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import logging

from episodic.topics.strategy import TopicStrategy, TopicDecision, Confidence

logger = logging.getLogger(__name__)


# ============================================================================
# OPERATIONAL METRICS
# ============================================================================

@dataclass
class OperationalMetrics:
    """
    Metrics that measure operational usefulness, not just benchmark accuracy.

    These metrics are more aligned with how topic detection impacts
    the actual system behavior (compression, retrieval, navigation).
    """
    # Standard metrics
    f1: float
    precision: float
    recall: float

    # Windowed metrics (with tolerance)
    windowed_f1_w1: float  # W=1 tolerance
    windowed_f1_w2: float  # W=2 tolerance
    windowed_precision_w1: float
    windowed_recall_w1: float

    # Standard segmentation metrics
    windowdiff: float  # Pevzner & Hearst (2002) - lower is better
    segmentation_similarity: float  # Higher is better

    # Boundary ratio
    bor: float  # Boundary Oversegmentation Ratio

    # Segment quality
    purity: float
    coverage: float

    # Major boundary metrics (heuristic)
    major_boundary_recall: float
    major_boundary_precision: float
    num_major_boundaries: int

    # Counts
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    num_gold_boundaries: int
    num_predicted_boundaries: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_windowed_metrics(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int,
    window: int = 1
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 with tolerance window.

    A predicted boundary at t is considered correct if there's a gold
    boundary in [t-window, t+window].

    Args:
        gold_boundaries: Set of gold boundary positions
        predicted_boundaries: Set of predicted boundary positions
        num_messages: Total number of messages
        window: Tolerance window size

    Returns:
        (precision, recall, f1) with tolerance
    """
    if not gold_boundaries and not predicted_boundaries:
        return 1.0, 1.0, 1.0

    # For each predicted boundary, check if it matches any gold within window
    matched_predictions = set()
    matched_golds = set()

    for pred in predicted_boundaries:
        for gold in gold_boundaries:
            if abs(pred - gold) <= window:
                matched_predictions.add(pred)
                matched_golds.add(gold)
                break

    # Precision: what fraction of predictions are near a gold boundary
    precision = len(matched_predictions) / len(predicted_boundaries) if predicted_boundaries else 0.0

    # Recall: what fraction of gold boundaries have a nearby prediction
    recall = len(matched_golds) / len(gold_boundaries) if gold_boundaries else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_windowdiff(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int,
    window_size: Optional[int] = None
) -> float:
    """
    Compute WindowDiff metric (Pevzner & Hearst, 2002).

    WindowDiff measures the proportion of windows where the number of
    boundaries in the reference differs from the hypothesis.

    Lower is better (0 = perfect, 1 = worst).

    Args:
        gold_boundaries: Set of gold boundary positions
        predicted_boundaries: Set of predicted boundary positions
        num_messages: Total number of messages
        window_size: Window size (default: half average segment length)

    Returns:
        WindowDiff score (0-1, lower is better)
    """
    if num_messages <= 1:
        return 0.0

    # Default window size: half average segment length
    if window_size is None:
        if gold_boundaries:
            avg_segment = num_messages / (len(gold_boundaries) + 1)
            window_size = max(2, int(avg_segment / 2))
        else:
            window_size = max(2, num_messages // 4)

    # Count boundaries in each window
    errors = 0
    total_windows = 0

    for i in range(num_messages - window_size):
        # Count boundaries in window [i, i+window_size)
        gold_count = sum(1 for b in gold_boundaries if i < b <= i + window_size)
        pred_count = sum(1 for b in predicted_boundaries if i < b <= i + window_size)

        if gold_count != pred_count:
            errors += 1
        total_windows += 1

    if total_windows == 0:
        return 0.0

    return errors / total_windows


def compute_segmentation_similarity(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int
) -> float:
    """
    Compute Segmentation Similarity (Fournier, 2012).

    Based on edit distance between boundary sequences.
    Higher is better (1 = perfect).

    Args:
        gold_boundaries: Set of gold boundary positions
        predicted_boundaries: Set of predicted boundary positions
        num_messages: Total number of messages

    Returns:
        Similarity score (0-1, higher is better)
    """
    if num_messages <= 1:
        return 1.0

    # Convert to segment length sequences
    def boundaries_to_lengths(boundaries: Set[int], n: int) -> List[int]:
        sorted_bounds = sorted(boundaries)
        lengths = []
        start = 0
        for b in sorted_bounds:
            if b > start:
                lengths.append(b - start)
            start = b
        if start < n:
            lengths.append(n - start)
        return lengths if lengths else [n]

    gold_lengths = boundaries_to_lengths(gold_boundaries, num_messages)
    pred_lengths = boundaries_to_lengths(predicted_boundaries, num_messages)

    # Compute boundary edit distance (simplified version)
    # Full version uses transposition costs, but this is a good approximation
    max_edits = max(len(gold_boundaries), len(predicted_boundaries), 1)

    # Count mismatches using windowed matching
    matched = 0
    for g in gold_boundaries:
        for p in predicted_boundaries:
            if abs(g - p) <= 1:  # Within 1 position
                matched += 1
                break

    if not gold_boundaries and not predicted_boundaries:
        return 1.0

    max_possible = max(len(gold_boundaries), len(predicted_boundaries))
    return matched / max_possible if max_possible > 0 else 1.0


def compute_bor(
    num_gold_boundaries: int,
    num_predicted_boundaries: int
) -> float:
    """
    Compute Boundary Oversegmentation Ratio.

    BOR = num_predicted / num_gold

    Interpretation:
    - BOR = 1.0: Same number of boundaries
    - BOR > 1.0: Oversegmentation (too many boundaries)
    - BOR < 1.0: Undersegmentation (too few boundaries)

    Healthy range: 0.8 - 1.4
    """
    if num_gold_boundaries == 0:
        return float('inf') if num_predicted_boundaries > 0 else 1.0
    return num_predicted_boundaries / num_gold_boundaries


def compute_purity_coverage(
    gold_segments: List[Set[int]],
    predicted_segments: List[Set[int]]
) -> Tuple[float, float]:
    """
    Compute segment purity and coverage.

    Purity: For each predicted segment, what fraction belongs to a single gold segment?
    Coverage: For each gold segment, what fraction is in a single predicted segment?

    Args:
        gold_segments: List of sets, each containing message indices in a gold segment
        predicted_segments: List of sets, each containing message indices in a predicted segment

    Returns:
        (purity, coverage) scores
    """
    if not gold_segments or not predicted_segments:
        return 0.0, 0.0

    # Purity: for each predicted segment, max overlap with any gold segment / segment size
    purities = []
    for pred_seg in predicted_segments:
        if not pred_seg:
            continue
        max_overlap = max(
            len(pred_seg & gold_seg) for gold_seg in gold_segments
        )
        purities.append(max_overlap / len(pred_seg))

    # Coverage: for each gold segment, max overlap with any predicted segment / segment size
    coverages = []
    for gold_seg in gold_segments:
        if not gold_seg:
            continue
        max_overlap = max(
            len(gold_seg & pred_seg) for pred_seg in predicted_segments
        )
        coverages.append(max_overlap / len(gold_seg))

    purity = sum(purities) / len(purities) if purities else 0.0
    coverage = sum(coverages) / len(coverages) if coverages else 0.0

    return purity, coverage


def boundaries_to_segments(
    boundaries: Set[int],
    num_messages: int
) -> List[Set[int]]:
    """
    Convert boundary positions to segment sets.

    Args:
        boundaries: Set of boundary positions (where new segments start)
        num_messages: Total number of messages

    Returns:
        List of sets, each containing message indices in a segment
    """
    if num_messages == 0:
        return []

    sorted_boundaries = sorted(boundaries)
    segments = []

    # First segment: [0, first_boundary)
    start = 0
    for boundary in sorted_boundaries:
        if boundary > start:
            segments.append(set(range(start, boundary)))
        start = boundary

    # Last segment: [last_boundary, num_messages)
    if start < num_messages:
        segments.append(set(range(start, num_messages)))

    return segments


# Major boundary detection heuristics
MAJOR_BOUNDARY_PATTERNS = [
    r'\b(by the way|btw)\b',
    r'\b(anyway|anyhow)\b',
    r'\b(on (a |an )?other (note|topic|subject))\b',
    r'\b(changing (topic|subject|gears))\b',
    r'\b(new question|different question)\b',
    r'\b(moving on|let\'?s move on)\b',
    r'\b(back to|getting back to)\b',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MAJOR_BOUNDARY_PATTERNS]


def is_likely_major_boundary(
    message_content: str,
    prev_messages: List[Dict[str, Any]],
    semantic_distance: Optional[float] = None
) -> bool:
    """
    Heuristically determine if a message is likely a major topic boundary.

    Major boundaries are high-cost transitions that should not be missed.

    Heuristics:
    1. Explicit transition markers ("by the way", "new question", etc.)
    2. High semantic distance from recent context (top 20th percentile)
    3. Return to previously discussed entity/keyword

    Args:
        message_content: The message text
        prev_messages: Previous messages for context
        semantic_distance: Optional pre-computed semantic distance (0-1)

    Returns:
        True if this is likely a major boundary
    """
    content_lower = message_content.lower()

    # Check explicit markers
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(content_lower):
            return True

    # Check semantic distance (if provided)
    if semantic_distance is not None and semantic_distance > 0.6:
        return True

    # Check for question after long assistant response (often indicates new topic)
    if prev_messages and len(prev_messages) >= 2:
        last_msg = prev_messages[-1]
        if last_msg.get('role') == 'assistant':
            last_content = last_msg.get('content', '')
            if len(last_content) > 500 and content_lower.rstrip().endswith('?'):
                return True

    return False


def compute_operational_metrics(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int,
    messages: Optional[List[Dict[str, Any]]] = None,
    semantic_distances: Optional[Dict[int, float]] = None
) -> OperationalMetrics:
    """
    Compute all operational metrics for a single dialogue.

    Args:
        gold_boundaries: Set of gold boundary positions
        predicted_boundaries: Set of predicted boundary positions
        num_messages: Total number of messages
        messages: Optional message list for major boundary heuristics
        semantic_distances: Optional dict of position -> semantic distance

    Returns:
        OperationalMetrics with all computed values
    """
    # Convert to sets if lists were passed
    if isinstance(gold_boundaries, list):
        gold_boundaries = set(gold_boundaries)
    if isinstance(predicted_boundaries, list):
        predicted_boundaries = set(predicted_boundaries)

    # Standard metrics
    tp = len(gold_boundaries & predicted_boundaries)
    fp = len(predicted_boundaries - gold_boundaries)
    fn = len(gold_boundaries - predicted_boundaries)
    tn = num_messages - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Windowed metrics
    w1_prec, w1_rec, w1_f1 = compute_windowed_metrics(
        gold_boundaries, predicted_boundaries, num_messages, window=1
    )
    _, _, w2_f1 = compute_windowed_metrics(
        gold_boundaries, predicted_boundaries, num_messages, window=2
    )

    # BOR
    bor = compute_bor(len(gold_boundaries), len(predicted_boundaries))

    # WindowDiff and Segmentation Similarity
    windowdiff = compute_windowdiff(gold_boundaries, predicted_boundaries, num_messages)
    seg_sim = compute_segmentation_similarity(gold_boundaries, predicted_boundaries, num_messages)

    # Purity and coverage
    gold_segments = boundaries_to_segments(gold_boundaries, num_messages)
    pred_segments = boundaries_to_segments(predicted_boundaries, num_messages)
    purity, coverage = compute_purity_coverage(gold_segments, pred_segments)

    # Major boundary metrics
    major_gold = set()
    if messages:
        for pos in gold_boundaries:
            if pos < len(messages):
                msg = messages[pos]
                content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
                prev = messages[:pos] if pos > 0 else []
                sem_dist = semantic_distances.get(pos) if semantic_distances else None
                if is_likely_major_boundary(content, prev, sem_dist):
                    major_gold.add(pos)

    major_tp = len(major_gold & predicted_boundaries)
    major_recall = major_tp / len(major_gold) if major_gold else 1.0
    major_precision = major_tp / len(predicted_boundaries) if predicted_boundaries else 0.0

    return OperationalMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        windowed_f1_w1=w1_f1,
        windowed_f1_w2=w2_f1,
        windowed_precision_w1=w1_prec,
        windowed_recall_w1=w1_rec,
        windowdiff=windowdiff,
        segmentation_similarity=seg_sim,
        bor=bor,
        purity=purity,
        coverage=coverage,
        major_boundary_recall=major_recall,
        major_boundary_precision=major_precision,
        num_major_boundaries=len(major_gold),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        num_gold_boundaries=len(gold_boundaries),
        num_predicted_boundaries=len(predicted_boundaries)
    )


def aggregate_operational_metrics(
    metrics_list: List[OperationalMetrics]
) -> OperationalMetrics:
    """
    Aggregate operational metrics across multiple dialogues.

    Uses micro-averaging (sum counts, then compute ratios).
    """
    if not metrics_list:
        return OperationalMetrics(
            f1=0, precision=0, recall=0,
            windowed_f1_w1=0, windowed_f1_w2=0,
            windowed_precision_w1=0, windowed_recall_w1=0,
            windowdiff=0, segmentation_similarity=0,
            bor=0, purity=0, coverage=0,
            major_boundary_recall=0, major_boundary_precision=0,
            num_major_boundaries=0,
            true_positives=0, false_positives=0,
            true_negatives=0, false_negatives=0,
            num_gold_boundaries=0, num_predicted_boundaries=0
        )

    # Sum counts
    total_tp = sum(m.true_positives for m in metrics_list)
    total_fp = sum(m.false_positives for m in metrics_list)
    total_tn = sum(m.true_negatives for m in metrics_list)
    total_fn = sum(m.false_negatives for m in metrics_list)
    total_gold = sum(m.num_gold_boundaries for m in metrics_list)
    total_pred = sum(m.num_predicted_boundaries for m in metrics_list)
    total_major = sum(m.num_major_boundaries for m in metrics_list)

    # Recompute ratios
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    bor = total_pred / total_gold if total_gold > 0 else 1.0

    # Average the windowed and segment metrics (macro-average)
    avg_w1_f1 = sum(m.windowed_f1_w1 for m in metrics_list) / len(metrics_list)
    avg_w2_f1 = sum(m.windowed_f1_w2 for m in metrics_list) / len(metrics_list)
    avg_w1_prec = sum(m.windowed_precision_w1 for m in metrics_list) / len(metrics_list)
    avg_w1_rec = sum(m.windowed_recall_w1 for m in metrics_list) / len(metrics_list)
    avg_windowdiff = sum(m.windowdiff for m in metrics_list) / len(metrics_list)
    avg_seg_sim = sum(m.segmentation_similarity for m in metrics_list) / len(metrics_list)
    avg_purity = sum(m.purity for m in metrics_list) / len(metrics_list)
    avg_coverage = sum(m.coverage for m in metrics_list) / len(metrics_list)
    avg_major_rec = sum(m.major_boundary_recall for m in metrics_list) / len(metrics_list)
    avg_major_prec = sum(m.major_boundary_precision for m in metrics_list) / len(metrics_list)

    return OperationalMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        windowed_f1_w1=avg_w1_f1,
        windowed_f1_w2=avg_w2_f1,
        windowed_precision_w1=avg_w1_prec,
        windowed_recall_w1=avg_w1_rec,
        windowdiff=avg_windowdiff,
        segmentation_similarity=avg_seg_sim,
        bor=bor,
        purity=avg_purity,
        coverage=avg_coverage,
        major_boundary_recall=avg_major_rec,
        major_boundary_precision=avg_major_prec,
        num_major_boundaries=total_major,
        true_positives=total_tp,
        false_positives=total_fp,
        true_negatives=total_tn,
        false_negatives=total_fn,
        num_gold_boundaries=total_gold,
        num_predicted_boundaries=total_pred
    )


# ============================================================================
# TEST CASE AND EVALUATION CLASSES
# ============================================================================


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
