"""Calibration dataclasses (config, metrics, result, report).

Split out of calibration.py; re-exported there.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


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

