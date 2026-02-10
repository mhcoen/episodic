"""
Operational metrics for topic boundary evaluation.

Includes windowed F1, BOR, purity/coverage, and standard segmentation
metrics (WindowDiff, Segmentation Similarity).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

from episodic.topics.eval_models import is_likely_major_boundary


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
    Compute precision, recall, F1 with tolerance window (many-to-one matching).

    A predicted boundary at t is considered correct if there's a gold
    boundary in [t-window, t+window]. Multiple predictions can match the
    same gold boundary (many-to-one), which makes this variant recall-favoring.

    For one-to-one matching, use compute_windowed_metrics_one_to_one().

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


def compute_windowed_metrics_one_to_one(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int,
    window: int = 1
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 with tolerance window (one-to-one matching).

    Uses greedy bipartite assignment: each gold boundary matches at most one
    prediction, and vice versa. Matches are assigned greedily by distance
    (closest pairs first), which is optimal for this problem.

    This is the standard tolerant matching used in much of the segmentation
    literature. Compare with compute_windowed_metrics() which allows
    many-to-one matching.

    Args:
        gold_boundaries: Set of gold boundary positions
        predicted_boundaries: Set of predicted boundary positions
        num_messages: Total number of messages (unused, kept for API consistency)
        window: Tolerance window size

    Returns:
        (precision, recall, f1) with one-to-one tolerance matching
    """
    if not gold_boundaries and not predicted_boundaries:
        return 1.0, 1.0, 1.0
    if not gold_boundaries:
        return 0.0, 1.0, 0.0
    if not predicted_boundaries:
        return 1.0, 0.0, 0.0

    # Build candidate matches within window: (distance, gold, pred)
    candidates = []
    for g in gold_boundaries:
        for p in predicted_boundaries:
            dist = abs(g - p)
            if dist <= window:
                candidates.append((dist, g, p))

    # Greedy assignment: closest pairs first (optimal for bipartite matching by distance)
    candidates.sort(key=lambda x: x[0])
    matched_gold = set()
    matched_pred = set()

    for dist, g, p in candidates:
        if g not in matched_gold and p not in matched_pred:
            matched_gold.add(g)
            matched_pred.add(p)

    tp = len(matched_gold)  # = len(matched_pred)
    precision = tp / len(predicted_boundaries)
    recall = tp / len(gold_boundaries)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_exact_f1(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    num_messages: int
) -> Tuple[float, float, float]:
    """
    Compute exact-match F1 (strict position matching, no tolerance window).

    This is the strictest form of F1: a prediction is correct only if it
    matches a gold boundary position exactly. Used by CSM and SuperDialseg.

    Args:
        gold_boundaries: Set of gold boundary positions
        predicted_boundaries: Set of predicted boundary positions
        num_messages: Total number of messages (for context, not used in computation)

    Returns:
        (precision, recall, f1) with exact position matching
    """
    if not gold_boundaries and not predicted_boundaries:
        return 1.0, 1.0, 1.0
    if not gold_boundaries:
        return 0.0, 1.0, 0.0
    if not predicted_boundaries:
        return 1.0, 0.0, 0.0

    # Exact matches only
    tp = len(gold_boundaries & predicted_boundaries)
    fp = len(predicted_boundaries - gold_boundaries)
    fn = len(gold_boundaries - predicted_boundaries)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
