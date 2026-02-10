"""
Boundary calibration for domain-specific threshold tuning.

Separates salience scoring from binary decision-making, allowing
thresholds to be tuned per-domain without retraining the model.

Key insight: The neural model produces well-calibrated salience scores,
but the default threshold encodes SuperDialseg's annotation style.
Different datasets may need different thresholds for optimal BOR.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple


@dataclass
class CalibrationResult:
    """Result of calibration optimization."""
    optimal_threshold: float
    achieved_bor: float
    achieved_f1: float
    achieved_precision: float
    achieved_recall: float
    threshold_history: List[Tuple[float, float, float]]  # (threshold, bor, f1)


# Predefined granularity levels for multi-scale segmentation
GRANULARITY_LEVELS = {
    'fine': 0.3,      # Many boundaries, micro-shifts
    'medium': 0.5,    # Default, balanced
    'coarse': 0.7,    # Few boundaries, major themes only
}


class BoundaryCalibrator:
    """
    Converts salience scores to binary decisions with domain-specific calibration.

    The neural model outputs continuous salience scores s ∈ [0,1].
    This class tunes the threshold τ to achieve a target BOR (Boundary
    Oversegmentation Ratio) or target segment length.

    Calibration strategies:
    1. BOR-targeting (supervised): Tune τ until BOR ≈ target (usually 1.0)
    2. Length-targeting (unsupervised): Tune τ until mean segment length ≈ target
    3. F1-maximizing: Find τ that maximizes F1 on validation data
    """

    def __init__(
        self,
        threshold: float = 0.5,
        target_bor: float = 1.0,
        granularity: str = 'medium'
    ):
        """
        Initialize calibrator.

        Args:
            threshold: Initial decision threshold
            target_bor: Target BOR for calibration (1.0 = match gold granularity)
            granularity: Preset granularity level ('fine', 'medium', 'coarse')
        """
        self.threshold = threshold
        self.target_bor = target_bor

        # Apply preset if specified
        if granularity in GRANULARITY_LEVELS:
            self.threshold = GRANULARITY_LEVELS[granularity]

    def apply(self, score: float) -> bool:
        """Apply calibrated threshold to get binary decision."""
        return score > self.threshold

    def apply_batch(self, scores: List[float]) -> List[bool]:
        """Apply threshold to batch of scores."""
        return [score > self.threshold for score in scores]

    def predict_boundaries(
        self,
        scores: List[float],
        indices: Optional[List[int]] = None
    ) -> Set[int]:
        """
        Convert scores to boundary indices.

        Args:
            scores: List of salience scores
            indices: Optional list of message indices corresponding to scores.
                     If None, assumes indices are 0..len(scores)-1.

        Returns:
            Set of boundary indices
        """
        if indices is None:
            indices = list(range(len(scores)))

        return {idx for idx, score in zip(indices, scores) if score > self.threshold}

    def calibrate_for_bor(
        self,
        scores: List[float],
        gold_boundaries: Set[int],
        num_messages: int,
        target_bor: float = 1.0,
        tolerance: float = 0.05,
        max_iterations: int = 50
    ) -> CalibrationResult:
        """
        Calibrate threshold to achieve target BOR using binary search.

        Args:
            scores: List of (index, score) pairs from model predictions
            gold_boundaries: Set of gold boundary indices
            num_messages: Total number of messages
            target_bor: Target BOR (default 1.0 = match gold count)
            tolerance: How close to target BOR is acceptable
            max_iterations: Maximum search iterations

        Returns:
            CalibrationResult with optimal threshold and metrics
        """
        if len(gold_boundaries) == 0:
            return CalibrationResult(
                optimal_threshold=self.threshold,
                achieved_bor=0.0,
                achieved_f1=0.0,
                achieved_precision=0.0,
                achieved_recall=0.0,
                threshold_history=[]
            )

        # Binary search for optimal threshold
        lo, hi = 0.0, 1.0
        history = []

        for _ in range(max_iterations):
            mid = (lo + hi) / 2
            self.threshold = mid

            # Count predictions at this threshold
            pred_count = sum(1 for s in scores if s > mid)
            bor = pred_count / len(gold_boundaries) if len(gold_boundaries) > 0 else 0.0

            # Compute F1 for tracking
            pred_boundaries = {i for i, s in enumerate(scores) if s > mid}
            tp = len(pred_boundaries & gold_boundaries)
            fp = len(pred_boundaries - gold_boundaries)
            fn = len(gold_boundaries - pred_boundaries)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            history.append((mid, bor, f1))

            # Check if within tolerance
            if abs(bor - target_bor) <= tolerance:
                break

            # Adjust search range
            if bor > target_bor:
                lo = mid  # Too many predictions, raise threshold
            else:
                hi = mid  # Too few predictions, lower threshold

        # Final metrics at optimal threshold
        pred_boundaries = {i for i, s in enumerate(scores) if s > self.threshold}
        tp = len(pred_boundaries & gold_boundaries)
        fp = len(pred_boundaries - gold_boundaries)
        fn = len(gold_boundaries - pred_boundaries)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        final_bor = len(pred_boundaries) / len(gold_boundaries) if len(gold_boundaries) > 0 else 0

        return CalibrationResult(
            optimal_threshold=self.threshold,
            achieved_bor=final_bor,
            achieved_f1=f1,
            achieved_precision=precision,
            achieved_recall=recall,
            threshold_history=history
        )

    def calibrate_for_f1(
        self,
        scores: List[float],
        gold_boundaries: Set[int],
        search_range: Tuple[float, float] = (0.1, 0.9),
        num_points: int = 50
    ) -> CalibrationResult:
        """
        Find threshold that maximizes F1 via grid search.

        Args:
            scores: List of salience scores (indexed 0..len-1)
            gold_boundaries: Set of gold boundary indices
            search_range: (min, max) threshold range to search
            num_points: Number of thresholds to evaluate

        Returns:
            CalibrationResult with F1-optimal threshold
        """
        if len(gold_boundaries) == 0:
            return CalibrationResult(
                optimal_threshold=self.threshold,
                achieved_bor=0.0,
                achieved_f1=0.0,
                achieved_precision=0.0,
                achieved_recall=0.0,
                threshold_history=[]
            )

        best_threshold = self.threshold
        best_f1 = 0.0
        history = []

        for i in range(num_points):
            thresh = search_range[0] + (search_range[1] - search_range[0]) * i / (num_points - 1)

            pred_boundaries = {i for i, s in enumerate(scores) if s > thresh}
            tp = len(pred_boundaries & gold_boundaries)
            fp = len(pred_boundaries - gold_boundaries)
            fn = len(gold_boundaries - pred_boundaries)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            bor = len(pred_boundaries) / len(gold_boundaries) if len(gold_boundaries) > 0 else 0

            history.append((thresh, bor, f1))

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        self.threshold = best_threshold

        # Final metrics
        pred_boundaries = {i for i, s in enumerate(scores) if s > self.threshold}
        tp = len(pred_boundaries & gold_boundaries)
        fp = len(pred_boundaries - gold_boundaries)
        fn = len(gold_boundaries - pred_boundaries)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        final_bor = len(pred_boundaries) / len(gold_boundaries) if len(gold_boundaries) > 0 else 0

        return CalibrationResult(
            optimal_threshold=self.threshold,
            achieved_bor=final_bor,
            achieved_f1=best_f1,
            achieved_precision=precision,
            achieved_recall=recall,
            threshold_history=history
        )

    def calibrate_unsupervised(
        self,
        scores: List[float],
        target_segment_length: float,
        num_messages: int,
        tolerance: float = 1.0
    ) -> float:
        """
        Calibrate without labels using target segment length.

        Args:
            scores: List of salience scores
            target_segment_length: Desired average messages per segment
            num_messages: Total number of messages
            tolerance: How close to target length is acceptable

        Returns:
            Calibrated threshold
        """
        # Target number of boundaries to achieve desired segment length
        target_boundaries = max(1, int(num_messages / target_segment_length) - 1)

        # Sort scores descending
        sorted_scores = sorted(scores, reverse=True)

        # Find threshold that yields approximately target_boundaries
        if target_boundaries >= len(sorted_scores):
            self.threshold = 0.0
        else:
            # Set threshold just below the target_boundaries-th highest score
            self.threshold = sorted_scores[target_boundaries] - 0.001

        return self.threshold

    def segment_hierarchical(
        self,
        scores: List[float]
    ) -> Dict[str, Set[int]]:
        """
        Return boundaries at multiple granularity levels.

        Enables multi-scale topic analysis without reprocessing.

        Args:
            scores: List of salience scores

        Returns:
            Dict mapping granularity level to boundary sets
        """
        return {
            level: {i for i, s in enumerate(scores) if s > thresh}
            for level, thresh in GRANULARITY_LEVELS.items()
        }

    def to_dict(self) -> Dict:
        """Serialize calibration state."""
        return {
            'threshold': self.threshold,
            'target_bor': self.target_bor,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BoundaryCalibrator':
        """Deserialize calibration state."""
        return cls(
            threshold=data.get('threshold', 0.5),
            target_bor=data.get('target_bor', 1.0),
        )


def sweep_thresholds(
    scores: List[float],
    gold_boundaries: Set[int],
    thresholds: List[float] = None
) -> List[Dict]:
    """
    Evaluate multiple thresholds and return metrics for each.

    Useful for threshold sensitivity analysis.

    Args:
        scores: List of salience scores
        gold_boundaries: Gold boundary indices
        thresholds: List of thresholds to test (default: 0.1 to 0.9)

    Returns:
        List of dicts with threshold and metrics
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for thresh in thresholds:
        pred = {i for i, s in enumerate(scores) if s > thresh}

        tp = len(pred & gold_boundaries)
        fp = len(pred - gold_boundaries)
        fn = len(gold_boundaries - pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        bor = len(pred) / len(gold_boundaries) if len(gold_boundaries) > 0 else 0

        results.append({
            'threshold': thresh,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'bor': bor,
            'predicted_count': len(pred),
            'gold_count': len(gold_boundaries),
        })

    return results
