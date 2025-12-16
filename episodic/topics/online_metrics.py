"""
Online evaluation metrics for topic detection.

Computes metrics specific to online (turn-by-turn) evaluation:
- Boundary delay: How late are predicted boundaries vs gold?
- SUSPECT timing: Time to enter SUSPECT, time to COMMIT
- Churn rate: SUSPECT entries that ABORT vs COMMIT

These complement the standard W-F1, BOR, precision, recall metrics
from evaluation.py.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class OnlineMetrics:
    """
    Metrics for online (turn-by-turn) topic detection evaluation.

    Combines standard metrics (W-F1, BOR) with online-specific metrics
    (delay, timing, churn) to fully characterize strategy behavior.
    """
    # Standard metrics (from evaluation.py, included for completeness)
    w_f1: float                     # Windowed F1 with tolerance
    bor: float                      # Boundary Oversegmentation Ratio
    precision: float
    recall: float

    # Boundary delay (measured only for matched boundary pairs)
    # Uses greedy one-to-one matching within tolerance window
    # delay = predicted_idx - gold_idx (positive = late, negative = early)
    delay_mean: float               # Mean delay for matched boundaries
    delay_std: float                # Std dev of delay
    delay_histogram: Dict[int, int] # delay value -> count
    delay_coverage: float           # matched_gold / total_gold (prevents misleading delay_mean)

    # SUSPECT timing metrics
    time_to_suspect_mean: float     # Avg turns from gold boundary to SUSPECT entry
    time_to_suspect_by_cause: Dict[str, float]  # "neural" | "drift" -> avg time
    time_to_commit_mean: float      # Avg turns in SUSPECT before COMMIT
    suspect_duration_histogram: Dict[int, int]  # duration -> count

    # Churn metrics
    suspect_abort_rate: float       # ABORT count / (ABORT + COMMIT count)
    abort_count: int
    commit_count: int
    commits_by_cause: Dict[str, int]  # "neural" | "drift" -> count

    # Metadata
    num_dialogues: int = 0
    total_turns: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_boundary_delay(
    gold_boundaries: Set[int],
    predicted_boundaries: Set[int],
    tolerance: int = 2
) -> Tuple[float, float, Dict[int, int], float]:
    """
    Compute boundary delay metrics using greedy one-to-one matching.

    For each gold boundary, finds the nearest unmatched predicted boundary
    within tolerance window. Computes delay as predicted_idx - gold_idx.

    Args:
        gold_boundaries: Set of gold boundary indices
        predicted_boundaries: Set of predicted boundary indices
        tolerance: Maximum distance for matching (default: 2)

    Returns:
        (delay_mean, delay_std, delay_histogram, delay_coverage)
        - delay_mean: Mean delay for matched boundaries
        - delay_std: Standard deviation of delays
        - delay_histogram: delay value -> count
        - delay_coverage: matched_gold / total_gold
    """
    if not gold_boundaries:
        return 0.0, 0.0, {}, 1.0

    # Greedy one-to-one matching: for each gold, find nearest unmatched pred
    sorted_gold = sorted(gold_boundaries)
    available_pred = set(predicted_boundaries)
    delays = []
    matched_gold = 0

    for gold in sorted_gold:
        # Find nearest predicted within tolerance
        best_pred = None
        best_dist = float('inf')

        for pred in available_pred:
            dist = abs(pred - gold)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_pred = pred

        if best_pred is not None:
            delay = best_pred - gold  # positive = late, negative = early
            delays.append(delay)
            available_pred.remove(best_pred)
            matched_gold += 1

    # Compute statistics
    delay_coverage = matched_gold / len(gold_boundaries) if gold_boundaries else 1.0

    if not delays:
        return 0.0, 0.0, {}, delay_coverage

    delay_mean = statistics.mean(delays)
    delay_std = statistics.stdev(delays) if len(delays) > 1 else 0.0

    # Build histogram
    delay_histogram: Dict[int, int] = {}
    for d in delays:
        delay_histogram[d] = delay_histogram.get(d, 0) + 1

    return delay_mean, delay_std, delay_histogram, delay_coverage


def compute_suspect_timing(
    commit_events: List[Tuple[int, int, str]],  # (suspect_entry_turn, commit_turn, cause)
    abort_events: List[Tuple[int, int]],  # (suspect_entry_turn, abort_turn)
    gold_boundaries: Set[int]
) -> Tuple[float, Dict[str, float], float, Dict[int, int]]:
    """
    Compute SUSPECT state timing metrics.

    Args:
        commit_events: List of (suspect_entry_turn, commit_turn, cause) tuples
        abort_events: List of (suspect_entry_turn, abort_turn) tuples
        gold_boundaries: Set of gold boundary indices

    Returns:
        (time_to_suspect_mean, time_to_suspect_by_cause, time_to_commit_mean, suspect_duration_histogram)
    """
    if not commit_events and not abort_events:
        return 0.0, {}, 0.0, {}

    # Time to SUSPECT from nearest preceding gold boundary
    time_to_suspect: List[float] = []
    time_to_suspect_neural: List[float] = []
    time_to_suspect_drift: List[float] = []

    sorted_gold = sorted(gold_boundaries) if gold_boundaries else []

    for suspect_entry, _, cause in commit_events:
        # Find nearest preceding gold boundary
        if sorted_gold:
            preceding = [g for g in sorted_gold if g <= suspect_entry]
            if preceding:
                time_to_suspect.append(suspect_entry - max(preceding))
                if cause == "neural":
                    time_to_suspect_neural.append(suspect_entry - max(preceding))
                elif cause == "drift":
                    time_to_suspect_drift.append(suspect_entry - max(preceding))

    time_to_suspect_mean = statistics.mean(time_to_suspect) if time_to_suspect else 0.0
    time_to_suspect_by_cause = {}
    if time_to_suspect_neural:
        time_to_suspect_by_cause["neural"] = statistics.mean(time_to_suspect_neural)
    if time_to_suspect_drift:
        time_to_suspect_by_cause["drift"] = statistics.mean(time_to_suspect_drift)

    # Time in SUSPECT before COMMIT
    commit_durations = [commit - entry for entry, commit, _ in commit_events]
    time_to_commit_mean = statistics.mean(commit_durations) if commit_durations else 0.0

    # Duration histogram (all SUSPECT episodes, both COMMIT and ABORT)
    all_durations = commit_durations + [abort - entry for entry, abort in abort_events]
    duration_histogram: Dict[int, int] = {}
    for d in all_durations:
        duration_histogram[d] = duration_histogram.get(d, 0) + 1

    return time_to_suspect_mean, time_to_suspect_by_cause, time_to_commit_mean, duration_histogram


def compute_churn_metrics(
    commit_events: List[Tuple[int, int, str]],  # (suspect_entry_turn, commit_turn, cause)
    abort_events: List[Tuple[int, int]]  # (suspect_entry_turn, abort_turn)
) -> Tuple[float, int, int, Dict[str, int]]:
    """
    Compute churn (ABORT vs COMMIT) metrics.

    Args:
        commit_events: List of (suspect_entry_turn, commit_turn, cause) tuples
        abort_events: List of (suspect_entry_turn, abort_turn) tuples

    Returns:
        (suspect_abort_rate, abort_count, commit_count, commits_by_cause)
    """
    commit_count = len(commit_events)
    abort_count = len(abort_events)
    total = commit_count + abort_count

    suspect_abort_rate = abort_count / total if total > 0 else 0.0

    commits_by_cause: Dict[str, int] = {}
    for _, _, cause in commit_events:
        commits_by_cause[cause] = commits_by_cause.get(cause, 0) + 1

    return suspect_abort_rate, abort_count, commit_count, commits_by_cause


def compute_online_metrics(
    dialogues: List['DialogueTrace'],  # Forward reference
    tolerance: int = 2
) -> OnlineMetrics:
    """
    Compute online metrics from a list of dialogue traces.

    Args:
        dialogues: List of DialogueTrace objects with gold and predicted boundaries
        tolerance: Tolerance window for boundary matching (default: 2)

    Returns:
        OnlineMetrics with all computed values
    """
    # Import here to avoid circular dependency
    from episodic.topics.evaluation import (
        compute_windowed_metrics,
        compute_bor,
    )

    if not dialogues:
        return OnlineMetrics(
            w_f1=0.0, bor=0.0, precision=0.0, recall=0.0,
            delay_mean=0.0, delay_std=0.0, delay_histogram={}, delay_coverage=0.0,
            time_to_suspect_mean=0.0, time_to_suspect_by_cause={},
            time_to_commit_mean=0.0, suspect_duration_histogram={},
            suspect_abort_rate=0.0, abort_count=0, commit_count=0, commits_by_cause={},
            num_dialogues=0, total_turns=0
        )

    # Aggregate boundaries across dialogues
    all_gold: Set[int] = set()
    all_pred: Set[int] = set()
    all_commit_events: List[Tuple[int, int, str]] = []
    all_abort_events: List[Tuple[int, int]] = []
    offset = 0
    total_turns = 0

    for dial in dialogues:
        # Offset boundaries by dialogue start position
        for g in dial.gold_boundaries:
            all_gold.add(g + offset)
        for p in dial.predicted_boundaries:
            all_pred.add(p + offset)

        # Collect events
        all_commit_events.extend(dial.commit_events)
        all_abort_events.extend(dial.abort_events)

        offset += len(dial.turns)
        total_turns += len(dial.turns)

    # Standard metrics
    precision, recall, w_f1 = compute_windowed_metrics(
        all_gold, all_pred, total_turns, window=tolerance
    )
    bor = compute_bor(len(all_gold), len(all_pred))

    # Delay metrics
    delay_mean, delay_std, delay_hist, delay_cov = compute_boundary_delay(
        all_gold, all_pred, tolerance
    )

    # SUSPECT timing
    time_to_suspect, time_by_cause, time_to_commit, dur_hist = compute_suspect_timing(
        all_commit_events, all_abort_events, all_gold
    )

    # Churn metrics
    abort_rate, abort_cnt, commit_cnt, commits_cause = compute_churn_metrics(
        all_commit_events, all_abort_events
    )

    return OnlineMetrics(
        w_f1=w_f1,
        bor=bor,
        precision=precision,
        recall=recall,
        delay_mean=delay_mean,
        delay_std=delay_std,
        delay_histogram=delay_hist,
        delay_coverage=delay_cov,
        time_to_suspect_mean=time_to_suspect,
        time_to_suspect_by_cause=time_by_cause,
        time_to_commit_mean=time_to_commit,
        suspect_duration_histogram=dur_hist,
        suspect_abort_rate=abort_rate,
        abort_count=abort_cnt,
        commit_count=commit_cnt,
        commits_by_cause=commits_cause,
        num_dialogues=len(dialogues),
        total_turns=total_turns,
    )


def aggregate_online_metrics(metrics_list: List[OnlineMetrics]) -> OnlineMetrics:
    """
    Aggregate OnlineMetrics across multiple evaluations (e.g., datasets).

    Uses micro-averaging for counts and macro-averaging for rates/means.

    Args:
        metrics_list: List of OnlineMetrics to aggregate

    Returns:
        Aggregated OnlineMetrics
    """
    if not metrics_list:
        return OnlineMetrics(
            w_f1=0.0, bor=0.0, precision=0.0, recall=0.0,
            delay_mean=0.0, delay_std=0.0, delay_histogram={}, delay_coverage=0.0,
            time_to_suspect_mean=0.0, time_to_suspect_by_cause={},
            time_to_commit_mean=0.0, suspect_duration_histogram={},
            suspect_abort_rate=0.0, abort_count=0, commit_count=0, commits_by_cause={},
            num_dialogues=0, total_turns=0
        )

    n = len(metrics_list)

    # Macro-average rates and means
    avg_w_f1 = sum(m.w_f1 for m in metrics_list) / n
    avg_bor = sum(m.bor for m in metrics_list) / n
    avg_precision = sum(m.precision for m in metrics_list) / n
    avg_recall = sum(m.recall for m in metrics_list) / n
    avg_delay_mean = sum(m.delay_mean for m in metrics_list) / n
    avg_delay_std = sum(m.delay_std for m in metrics_list) / n
    avg_delay_coverage = sum(m.delay_coverage for m in metrics_list) / n
    avg_time_to_suspect = sum(m.time_to_suspect_mean for m in metrics_list) / n
    avg_time_to_commit = sum(m.time_to_commit_mean for m in metrics_list) / n
    avg_abort_rate = sum(m.suspect_abort_rate for m in metrics_list) / n

    # Micro-aggregate counts
    total_abort = sum(m.abort_count for m in metrics_list)
    total_commit = sum(m.commit_count for m in metrics_list)
    total_dialogues = sum(m.num_dialogues for m in metrics_list)
    total_turns = sum(m.total_turns for m in metrics_list)

    # Merge histograms
    merged_delay_hist: Dict[int, int] = {}
    for m in metrics_list:
        for k, v in m.delay_histogram.items():
            merged_delay_hist[k] = merged_delay_hist.get(k, 0) + v

    merged_dur_hist: Dict[int, int] = {}
    for m in metrics_list:
        for k, v in m.suspect_duration_histogram.items():
            merged_dur_hist[k] = merged_dur_hist.get(k, 0) + v

    # Merge commits_by_cause
    merged_commits_cause: Dict[str, int] = {}
    for m in metrics_list:
        for k, v in m.commits_by_cause.items():
            merged_commits_cause[k] = merged_commits_cause.get(k, 0) + v

    # Merge time_to_suspect_by_cause (weighted average by commit count)
    merged_time_by_cause: Dict[str, float] = {}
    cause_counts: Dict[str, int] = {}
    for m in metrics_list:
        for cause, time in m.time_to_suspect_by_cause.items():
            count = m.commits_by_cause.get(cause, 1)
            merged_time_by_cause[cause] = merged_time_by_cause.get(cause, 0.0) + time * count
            cause_counts[cause] = cause_counts.get(cause, 0) + count
    for cause in merged_time_by_cause:
        if cause_counts.get(cause, 0) > 0:
            merged_time_by_cause[cause] /= cause_counts[cause]

    return OnlineMetrics(
        w_f1=avg_w_f1,
        bor=avg_bor,
        precision=avg_precision,
        recall=avg_recall,
        delay_mean=avg_delay_mean,
        delay_std=avg_delay_std,
        delay_histogram=merged_delay_hist,
        delay_coverage=avg_delay_coverage,
        time_to_suspect_mean=avg_time_to_suspect,
        time_to_suspect_by_cause=merged_time_by_cause,
        time_to_commit_mean=avg_time_to_commit,
        suspect_duration_histogram=merged_dur_hist,
        suspect_abort_rate=avg_abort_rate,
        abort_count=total_abort,
        commit_count=total_commit,
        commits_by_cause=merged_commits_cause,
        num_dialogues=total_dialogues,
        total_turns=total_turns,
    )
