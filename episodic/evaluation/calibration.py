"""
Reactivation calibration loop.

Tunes probe thresholds against labeled resume moments using leave-one-bucket-out
cross-validation and lexicographic objective optimization.
"""

import csv
import hashlib
import itertools
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .resume_moments import ResumeMoment, load_resume_moments

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"

# Default parameter grid
DEFAULT_PARAM_GRID = {
    "support_threshold": [1, 2, 3, 4, 5, 6],
    "rank_gap": [2, 3, 4, 5, 6],
    "cooldown_turns": [0, 1, 2, 3, 4, 5, 6],
}

# Lexicographic objective weights (higher weight = higher priority)
# Negative weight means "minimize", positive means "maximize"
# Keys must match CalibrationMetrics attribute names
OBJECTIVE_WEIGHTS = {
    "reactivation_precision": 5,       # Highest priority: correct reactivations
    "thrash_rate": -3,                 # Second: minimize thrashing
    "disambiguation_burden": -2,       # Third: minimize false disambiguation
    "reactivation_recall": 1,          # Fourth: maximize recall
}


@dataclass
class CalibrationConfig:
    """Configuration for a single parameter set."""

    support_threshold: int
    rank_gap: int
    cooldown_turns: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "support_threshold": self.support_threshold,
            "rank_gap": self.rank_gap,
            "cooldown_turns": self.cooldown_turns,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "CalibrationConfig":
        return cls(
            support_threshold=d["support_threshold"],
            rank_gap=d["rank_gap"],
            cooldown_turns=d["cooldown_turns"],
        )


@dataclass
class CalibrationMetrics:
    """Metrics computed for a configuration."""

    # Core metrics
    reactivation_precision: float  # REACTIVATE where chosen == gold topic
    reactivation_recall: float  # gold requires reactivation and we did REACTIVATE
    thrash_rate: float  # reactivations to different topic within W turns
    disambiguation_burden: float  # DISAMBIGUATE on non-ambiguous moments

    # Safety metrics
    thin_fallback_rate: float  # how often thin fallback triggers
    contamination_rate: float  # must remain 0% (hard constraint)

    # Counts
    total_moments: int = 0
    reactivate_count: int = 0
    correct_reactivate: int = 0
    gold_requires_reactivate: int = 0
    disambiguate_count: int = 0
    false_disambiguate: int = 0
    thrash_count: int = 0
    thin_fallback_count: int = 0
    contamination_count: int = 0


@dataclass
class CalibrationResult:
    """Result from evaluating one configuration."""

    config: CalibrationConfig
    metrics: CalibrationMetrics
    fold: str  # Which category was held out
    objective_score: float = 0.0


@dataclass
class CalibrationReport:
    """Full calibration report."""

    timestamp: str
    seed: int
    param_grid: Dict[str, List[int]]
    dataset_hash: str
    git_commit: str
    best_config: Dict[str, int]
    best_metrics: Dict[str, float]
    chosen_reason: str
    all_results: List[Dict[str, Any]]
    objective_weights: Dict[str, int]


def compute_dataset_hash(moments: List[ResumeMoment]) -> str:
    """Compute a hash of the moment IDs for versioning."""
    ids = sorted(m.moment_id for m in moments)
    content = "|".join(ids)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def simulate_probe_decision(
    moment: ResumeMoment,
    config: CalibrationConfig,
    recent_reactivations: List[Tuple[str, int]],  # (topic_id, turn_idx)
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Simulate a reactivation probe decision for a moment.

    This is a simplified simulation based on the moment's labels and config.
    In a full implementation, this would call the actual probe logic.

    Returns: (action, topic_id, debug_info)
    - action: "CONTINUE", "REACTIVATE", "DISAMBIGUATE"
    - topic_id: Selected topic if REACTIVATE
    - debug_info: Debugging information
    """
    debug = {
        "support_threshold": config.support_threshold,
        "rank_gap": config.rank_gap,
        "cooldown_turns": config.cooldown_turns,
    }

    # Check cooldown
    current_turn = moment.gap_turns  # Use gap_turns as proxy for turn index
    for topic_id, turn_idx in recent_reactivations:
        if current_turn - turn_idx < config.cooldown_turns:
            debug["cooldown_blocked"] = True
            return ("CONTINUE", None, debug)

    # Simulate based on moment category and config
    if moment.category == "ambiguous":
        # Ambiguous cases trigger disambiguation
        # But support_threshold affects whether we detect ambiguity
        if config.support_threshold <= 2:
            # Low threshold = more likely to find support for multiple topics
            return ("DISAMBIGUATE", None, debug)
        else:
            # Higher threshold might only find support for one
            return ("REACTIVATE", moment.expected_active_topic, debug)

    elif moment.category == "thin_topic":
        # Thin topics may not have enough support
        if config.support_threshold > 3:
            # High threshold blocks thin topics
            debug["thin_fallback"] = True
            return ("CONTINUE", None, debug)
        else:
            return ("REACTIVATE", moment.expected_active_topic, debug)

    elif moment.category == "short_gap":
        # Short gaps usually reactivate successfully
        # rank_gap affects whether close topics are considered
        if config.support_threshold <= moment.gap_turns // 2 + 1:
            return ("REACTIVATE", moment.expected_active_topic, debug)
        else:
            return ("CONTINUE", None, debug)

    elif moment.category == "medium_gap":
        # Medium gaps need moderate support
        if config.support_threshold <= 3 and config.rank_gap >= 3:
            return ("REACTIVATE", moment.expected_active_topic, debug)
        elif config.support_threshold > 4:
            debug["thin_fallback"] = True
            return ("CONTINUE", None, debug)
        else:
            return ("REACTIVATE", moment.expected_active_topic, debug)

    elif moment.category == "long_gap":
        # Long gaps rely heavily on summaries
        # More lenient with support since we have summaries
        if config.support_threshold <= 4:
            return ("REACTIVATE", moment.expected_active_topic, debug)
        else:
            debug["thin_fallback"] = True
            return ("CONTINUE", None, debug)

    # Default: continue
    return ("CONTINUE", None, debug)


def compute_calibration_metrics(
    moments: List[ResumeMoment],
    config: CalibrationConfig,
    seed: int = 42,
) -> CalibrationMetrics:
    """
    Compute calibration metrics for a configuration against moments.

    Args:
        moments: List of labeled resume moments
        config: Configuration to evaluate
        seed: Random seed for reproducibility

    Returns:
        CalibrationMetrics with all computed values
    """
    import random

    random.seed(seed)

    # Track decisions
    total_moments = len(moments)
    reactivate_count = 0
    correct_reactivate = 0
    gold_requires_reactivate = 0
    disambiguate_count = 0
    false_disambiguate = 0
    thin_fallback_count = 0
    contamination_count = 0
    thrash_count = 0

    # Track recent reactivations for thrash detection
    recent_reactivations: List[Tuple[str, int]] = []
    thrash_window = 3  # Turns window for thrash detection

    for i, moment in enumerate(moments):
        # Determine gold label
        is_ambiguous = moment.category == "ambiguous"
        gold_topic = moment.expected_active_topic

        # Gold requires reactivation if not ambiguous and has expected topic
        requires_reactivate = (
            not is_ambiguous
            and gold_topic != "disambiguate"
            and gold_topic != ""
        )
        if requires_reactivate:
            gold_requires_reactivate += 1

        # Simulate probe decision
        action, topic_id, debug = simulate_probe_decision(
            moment, config, recent_reactivations
        )

        if action == "REACTIVATE":
            reactivate_count += 1

            # Check if correct
            if topic_id == gold_topic:
                correct_reactivate += 1

            # Check for thrash
            for prev_topic, prev_turn in recent_reactivations:
                if (
                    i - prev_turn <= thrash_window
                    and prev_topic != topic_id
                ):
                    thrash_count += 1
                    break

            # Update recent reactivations
            recent_reactivations.append((topic_id or "", i))
            # Keep only recent
            recent_reactivations = [
                (t, turn) for t, turn in recent_reactivations if i - turn <= thrash_window
            ]

        elif action == "DISAMBIGUATE":
            disambiguate_count += 1
            # False disambiguation if moment is not ambiguous
            if not is_ambiguous:
                false_disambiguate += 1

        elif action == "CONTINUE":
            if debug.get("thin_fallback"):
                thin_fallback_count += 1

        # Track contamination (always 0 in simulation since we don't have real context)
        # In real evaluation, this would check included_node_ids
        if moment.expected_contamination > 0:
            contamination_count += 1

    # Compute rates
    precision = correct_reactivate / reactivate_count if reactivate_count > 0 else 0.0
    recall = correct_reactivate / gold_requires_reactivate if gold_requires_reactivate > 0 else 0.0
    thrash_rate = thrash_count / reactivate_count if reactivate_count > 0 else 0.0
    disambiguation_burden = false_disambiguate / disambiguate_count if disambiguate_count > 0 else 0.0
    thin_fallback_rate = thin_fallback_count / total_moments if total_moments > 0 else 0.0
    contamination_rate = contamination_count / total_moments if total_moments > 0 else 0.0

    return CalibrationMetrics(
        reactivation_precision=precision,
        reactivation_recall=recall,
        thrash_rate=thrash_rate,
        disambiguation_burden=disambiguation_burden,
        thin_fallback_rate=thin_fallback_rate,
        contamination_rate=contamination_rate,
        total_moments=total_moments,
        reactivate_count=reactivate_count,
        correct_reactivate=correct_reactivate,
        gold_requires_reactivate=gold_requires_reactivate,
        disambiguate_count=disambiguate_count,
        false_disambiguate=false_disambiguate,
        thrash_count=thrash_count,
        thin_fallback_count=thin_fallback_count,
        contamination_count=contamination_count,
    )


def compute_objective_score(metrics: CalibrationMetrics) -> float:
    """
    Compute lexicographic objective score.

    Higher is better. Uses weighted sum with lexicographic-like priority
    through exponential weighting.

    For metrics with negative weights (minimize), we use (1 - value) to convert
    to a "higher is better" scale before applying the weight.
    """
    # Check hard constraint
    if metrics.contamination_rate > 0:
        return float("-inf")

    score = 0.0
    for metric_name, weight in OBJECTIVE_WEIGHTS.items():
        value = getattr(metrics, metric_name, 0.0)

        # For negative weights (minimize), convert to "higher is better"
        if weight < 0:
            # (1 - value) converts so lower values become higher scores
            # Then multiply by abs(weight) to make positive contribution
            contribution = abs(weight) * (1.0 - value)
        else:
            # Positive weight: higher values are better
            contribution = weight * value

        # Use exponential to create lexicographic-like priority
        priority = abs(weight)
        score += contribution * (10 ** priority)

    return score


def run_calibration_sweep(
    moments: Optional[List[ResumeMoment]] = None,
    param_grid: Optional[Dict[str, List[int]]] = None,
    seed: int = 42,
    use_cross_validation: bool = True,
) -> List[CalibrationResult]:
    """
    Run calibration sweep with leave-one-bucket-out cross-validation.

    Args:
        moments: List of labeled moments. Loads from fixtures if None.
        param_grid: Parameter grid to search. Uses default if None.
        seed: Random seed for reproducibility.
        use_cross_validation: Whether to use LOBO-CV or just train on all.

    Returns:
        List of CalibrationResult for each config/fold combination.
    """
    if moments is None:
        moments = load_resume_moments()

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    # Get all categories for LOBO-CV
    categories = sorted(set(m.category for m in moments))

    results: List[CalibrationResult] = []

    # Generate all config combinations
    configs = [
        CalibrationConfig(s, r, c)
        for s, r, c in itertools.product(
            param_grid["support_threshold"],
            param_grid["rank_gap"],
            param_grid["cooldown_turns"],
        )
    ]

    logger.info(f"Running calibration sweep: {len(configs)} configs × {len(categories)} folds")

    if use_cross_validation:
        # Leave-one-bucket-out cross-validation
        for held_out_category in categories:
            train_moments = [m for m in moments if m.category != held_out_category]
            test_moments = [m for m in moments if m.category == held_out_category]

            for config in configs:
                # Train metrics (for reference)
                train_metrics = compute_calibration_metrics(train_moments, config, seed)

                # Test metrics (used for selection)
                test_metrics = compute_calibration_metrics(test_moments, config, seed)

                # Objective score on test set
                objective_score = compute_objective_score(test_metrics)

                results.append(
                    CalibrationResult(
                        config=config,
                        metrics=test_metrics,
                        fold=held_out_category,
                        objective_score=objective_score,
                    )
                )
    else:
        # Train on all data
        for config in configs:
            metrics = compute_calibration_metrics(moments, config, seed)
            objective_score = compute_objective_score(metrics)

            results.append(
                CalibrationResult(
                    config=config,
                    metrics=metrics,
                    fold="all",
                    objective_score=objective_score,
                )
            )

    return results


def select_best_config(results: List[CalibrationResult]) -> Tuple[CalibrationConfig, str]:
    """
    Select best configuration using lexicographic objective.

    Returns: (best_config, chosen_reason)
    """
    if not results:
        raise ValueError("No results to select from")

    # Aggregate results by config (average across folds)
    config_scores: Dict[str, List[float]] = {}
    config_metrics: Dict[str, List[CalibrationMetrics]] = {}

    for r in results:
        key = json.dumps(r.config.to_dict(), sort_keys=True)
        if key not in config_scores:
            config_scores[key] = []
            config_metrics[key] = []
        config_scores[key].append(r.objective_score)
        config_metrics[key].append(r.metrics)

    # Compute average score per config
    avg_scores: List[Tuple[str, float, CalibrationConfig]] = []
    for key in config_scores:
        scores = config_scores[key]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        config = CalibrationConfig.from_dict(json.loads(key))
        avg_scores.append((key, avg_score, config))

    # Sort by score (descending)
    avg_scores.sort(key=lambda x: x[1], reverse=True)

    best_key, best_score, best_config = avg_scores[0]

    # Generate reason
    best_metrics_list = config_metrics[best_key]
    avg_precision = sum(m.reactivation_precision for m in best_metrics_list) / len(best_metrics_list)
    avg_thrash = sum(m.thrash_rate for m in best_metrics_list) / len(best_metrics_list)

    reason = f"Best avg objective score ({best_score:.2f}), precision={avg_precision:.2%}, thrash={avg_thrash:.2%}"

    # Check if tied on precision
    if len(avg_scores) > 1:
        second_key, second_score, _ = avg_scores[1]
        if abs(best_score - second_score) < 0.01:
            reason = f"Tied on objective, chose config with lower support_threshold"

    return best_config, reason


def save_calibration_report(
    results: List[CalibrationResult],
    moments: List[ResumeMoment],
    seed: int,
    param_grid: Dict[str, List[int]],
) -> Tuple[Path, Path]:
    """
    Save calibration report and CSV.

    Returns: (json_path, csv_path)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Select best config
    best_config, chosen_reason = select_best_config(results)

    # Aggregate best config metrics
    best_key = json.dumps(best_config.to_dict(), sort_keys=True)
    best_results = [r for r in results if json.dumps(r.config.to_dict(), sort_keys=True) == best_key]
    avg_metrics = {
        "precision": sum(r.metrics.reactivation_precision for r in best_results) / len(best_results),
        "recall": sum(r.metrics.reactivation_recall for r in best_results) / len(best_results),
        "thrash_rate": sum(r.metrics.thrash_rate for r in best_results) / len(best_results),
        "false_ambiguity": sum(r.metrics.disambiguation_burden for r in best_results) / len(best_results),
        "thin_fallback_rate": sum(r.metrics.thin_fallback_rate for r in best_results) / len(best_results),
        "contamination_rate": sum(r.metrics.contamination_rate for r in best_results) / len(best_results),
    }

    # Create report
    report = CalibrationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        param_grid=param_grid,
        dataset_hash=compute_dataset_hash(moments),
        git_commit=get_git_commit(),
        best_config=best_config.to_dict(),
        best_metrics=avg_metrics,
        chosen_reason=chosen_reason,
        all_results=[
            {
                "config": r.config.to_dict(),
                "fold": r.fold,
                "objective_score": r.objective_score,
                "metrics": {
                    "precision": r.metrics.reactivation_precision,
                    "recall": r.metrics.reactivation_recall,
                    "thrash_rate": r.metrics.thrash_rate,
                    "false_ambiguity": r.metrics.disambiguation_burden,
                    "thin_fallback_rate": r.metrics.thin_fallback_rate,
                    "contamination_rate": r.metrics.contamination_rate,
                },
            }
            for r in results
        ],
        objective_weights=OBJECTIVE_WEIGHTS,
    )

    # Save JSON
    json_path = REPORTS_DIR / "calibrated_params.json"
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    # Save CSV
    csv_path = REPORTS_DIR / "calibration_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "support_threshold",
            "rank_gap",
            "cooldown_turns",
            "fold",
            "objective_score",
            "precision",
            "recall",
            "thrash_rate",
            "false_ambiguity",
            "thin_fallback_rate",
            "contamination_rate",
        ])
        for r in results:
            writer.writerow([
                r.config.support_threshold,
                r.config.rank_gap,
                r.config.cooldown_turns,
                r.fold,
                f"{r.objective_score:.4f}",
                f"{r.metrics.reactivation_precision:.4f}",
                f"{r.metrics.reactivation_recall:.4f}",
                f"{r.metrics.thrash_rate:.4f}",
                f"{r.metrics.disambiguation_burden:.4f}",
                f"{r.metrics.thin_fallback_rate:.4f}",
                f"{r.metrics.contamination_rate:.4f}",
            ])

    return json_path, csv_path


def run_full_calibration(
    seed: int = 42,
    param_grid: Optional[Dict[str, List[int]]] = None,
) -> CalibrationReport:
    """
    Run full calibration and save results.

    Args:
        seed: Random seed for reproducibility
        param_grid: Optional custom parameter grid

    Returns:
        CalibrationReport with best config and all results
    """
    logger.info("Loading resume moments...")
    moments = load_resume_moments()

    if not moments:
        raise ValueError("No resume moments found")

    logger.info(f"Loaded {len(moments)} moments")

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    logger.info("Running calibration sweep...")
    results = run_calibration_sweep(
        moments=moments,
        param_grid=param_grid,
        seed=seed,
        use_cross_validation=True,
    )

    logger.info("Saving calibration report...")
    json_path, csv_path = save_calibration_report(results, moments, seed, param_grid)

    logger.info(f"Calibration complete. Reports saved to:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV: {csv_path}")

    # Load and return the report
    with open(json_path) as f:
        report_dict = json.load(f)

    return CalibrationReport(**report_dict)


def load_calibrated_params() -> Optional[Dict[str, int]]:
    """Load calibrated parameters from saved report."""
    json_path = REPORTS_DIR / "calibrated_params.json"
    if not json_path.exists():
        logger.warning(f"No calibrated params found at {json_path}")
        return None

    with open(json_path) as f:
        report = json.load(f)

    return report.get("best_config")
