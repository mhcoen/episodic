"""
Topic detection diagnostics for observability.

Computes metrics that help diagnose strategy behavior without
requiring ground truth labels. These are logged for analysis,
not used for automatic routing.

Metrics:
1. Controller stress - how hard commitment is working to hit target rate
2. Salience peak concentration - how confident/decisive the detector is
3. Time-gap-salience correlation - whether dynamics predict boundaries
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticSnapshot:
    """A snapshot of diagnostic metrics at a point in time."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Controller stress metrics
    controller_stress: float = 0.0  # 0-1, higher = more stressed
    min_evidence_saturation: float = 0.0  # How close to bounds (0=middle, 1=at bound)
    adjustment_count: int = 0  # Total adjustments made
    rate_volatility: float = 0.0  # Std dev of recent rates

    # Salience concentration metrics
    salience_peak_concentration: float = 0.0  # Top-k mass (0-1, higher = more decisive)
    salience_bimodality: float = 0.0  # How bimodal the distribution is

    # Time-gap correlation metrics
    time_gap_salience_correlation: float = 0.0  # Correlation coefficient (-1 to 1)
    time_gap_variance: float = 0.0  # Variance in inter-message gaps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'controller_stress': self.controller_stress,
            'min_evidence_saturation': self.min_evidence_saturation,
            'adjustment_count': self.adjustment_count,
            'rate_volatility': self.rate_volatility,
            'salience_peak_concentration': self.salience_peak_concentration,
            'salience_bimodality': self.salience_bimodality,
            'time_gap_salience_correlation': self.time_gap_salience_correlation,
            'time_gap_variance': self.time_gap_variance,
        }


class DiagnosticsCollector:
    """
    Collects and computes diagnostic metrics over time.

    Usage:
        collector = DiagnosticsCollector()

        # After each decision
        collector.record_salience(salience_score)
        collector.record_time_gap(gap_seconds)
        collector.update_controller_state(signals)

        # Get current diagnostics
        snapshot = collector.get_snapshot()
    """

    def __init__(self, window_size: int = 50):
        """
        Initialize diagnostics collector.

        Args:
            window_size: Number of recent observations to keep
        """
        self.window_size = window_size

        # Rolling windows
        self._saliences: List[float] = []
        self._time_gaps: List[float] = []
        self._rates: List[float] = []

        # Controller state (from commitment strategy)
        self._adjustment_count: int = 0
        self._min_evidence: float = 0.7
        self._min_evidence_bounds: tuple = (0.3, 1.5)

    def record_salience(self, salience: float) -> None:
        """Record a salience score."""
        self._saliences.append(salience)
        if len(self._saliences) > self.window_size:
            self._saliences.pop(0)

    def record_time_gap(self, gap_seconds: float) -> None:
        """Record inter-message time gap."""
        self._time_gaps.append(gap_seconds)
        if len(self._time_gaps) > self.window_size:
            self._time_gaps.pop(0)

    def record_rate(self, rate: float) -> None:
        """Record committed boundary rate."""
        self._rates.append(rate)
        if len(self._rates) > self.window_size:
            self._rates.pop(0)

    def update_controller_state(
        self,
        signals: Dict[str, Any],
        min_evidence_bounds: tuple = (0.3, 1.5)
    ) -> None:
        """
        Update controller state from commitment strategy signals.

        Args:
            signals: The signals dict from TopicDecision
            min_evidence_bounds: The configured bounds for min_evidence
        """
        self._min_evidence_bounds = min_evidence_bounds

        # Adaptive commitment signals
        if 'current_min_evidence' in signals:
            self._min_evidence = signals['current_min_evidence']

        if 'adjustment_count' in signals:
            self._adjustment_count = signals['adjustment_count']

        if 'current_rate' in signals:
            self.record_rate(signals['current_rate'])

        # Basic commitment signals
        if 'accumulated_evidence' in signals:
            # Track accumulated evidence as a rate proxy
            self.record_rate(signals['accumulated_evidence'])

        # Salience from neural strategy
        if 'boundary_probability' in signals:
            self.record_salience(signals['boundary_probability'])
        elif 'confidence_score' in signals:
            self.record_salience(signals['confidence_score'])

    def compute_controller_stress(self) -> float:
        """
        Compute controller stress (0-1).

        High stress indicates the controller is struggling to hit target rate.
        """
        if not self._rates:
            return 0.0

        # Stress based on rate volatility and saturation
        saturation = self.compute_min_evidence_saturation()
        volatility = self.compute_rate_volatility()

        # Combine: high saturation OR high volatility = stress
        return min(1.0, saturation * 0.6 + volatility * 0.4)

    def compute_min_evidence_saturation(self) -> float:
        """
        Compute how close min_evidence is to bounds (0-1).

        0 = middle of range, 1 = at a bound
        """
        low, high = self._min_evidence_bounds
        mid = (low + high) / 2
        range_half = (high - low) / 2

        if range_half == 0:
            return 1.0

        distance_from_mid = abs(self._min_evidence - mid)
        return min(1.0, distance_from_mid / range_half)

    def compute_rate_volatility(self) -> float:
        """Compute rate volatility (normalized std dev)."""
        if len(self._rates) < 3:
            return 0.0

        import numpy as np
        std = np.std(self._rates)
        mean = np.mean(self._rates)

        if mean == 0:
            return std  # Just return raw std if mean is 0

        # Coefficient of variation, capped at 1
        return min(1.0, std / mean) if mean > 0 else 0.0

    def compute_salience_peak_concentration(self, top_k: int = 5) -> float:
        """
        Compute what fraction of total salience is in top-k peaks.

        Higher = more decisive detector (clear peaks vs flat distribution)
        """
        if len(self._saliences) < top_k:
            return 0.0

        total = sum(self._saliences)
        if total == 0:
            return 0.0

        sorted_saliences = sorted(self._saliences, reverse=True)
        top_k_sum = sum(sorted_saliences[:top_k])

        return top_k_sum / total

    def compute_salience_bimodality(self) -> float:
        """
        Estimate bimodality of salience distribution.

        Bimodal = clear separation between "boundary" and "non-boundary" scores.
        Uses Sarle's bimodality coefficient approximation.
        """
        if len(self._saliences) < 10:
            return 0.0

        import numpy as np
        data = np.array(self._saliences)

        n = len(data)
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        # Skewness
        skew = np.mean(((data - mean) / std) ** 3)

        # Kurtosis (excess)
        kurt = np.mean(((data - mean) / std) ** 4) - 3

        # Sarle's bimodality coefficient
        # BC = (skew^2 + 1) / (kurt + 3 * (n-1)^2 / ((n-2)(n-3)))
        # Simplified: higher when distribution is bimodal
        bc = (skew ** 2 + 1) / (kurt + 3)

        # Normalize to 0-1 range (BC > 0.555 suggests bimodality)
        return min(1.0, max(0.0, (bc - 0.3) / 0.5))

    def compute_time_gap_salience_correlation(self) -> float:
        """
        Compute correlation between time gaps and saliences.

        High positive correlation suggests time gaps predict boundaries.
        """
        # Need aligned time gaps and saliences
        n = min(len(self._time_gaps), len(self._saliences))
        if n < 5:
            return 0.0

        import numpy as np
        gaps = np.array(self._time_gaps[-n:])
        saliences = np.array(self._saliences[-n:])

        # Use log of gaps (time gaps are often log-normal)
        log_gaps = np.log1p(gaps)

        # Pearson correlation
        if np.std(log_gaps) == 0 or np.std(saliences) == 0:
            return 0.0

        corr = np.corrcoef(log_gaps, saliences)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def compute_time_gap_variance(self) -> float:
        """Compute variance in time gaps (log scale)."""
        if len(self._time_gaps) < 3:
            return 0.0

        import numpy as np
        log_gaps = np.log1p(self._time_gaps)
        return float(np.var(log_gaps))

    def get_snapshot(self) -> DiagnosticSnapshot:
        """Get current diagnostic snapshot."""
        return DiagnosticSnapshot(
            controller_stress=self.compute_controller_stress(),
            min_evidence_saturation=self.compute_min_evidence_saturation(),
            adjustment_count=self._adjustment_count,
            rate_volatility=self.compute_rate_volatility(),
            salience_peak_concentration=self.compute_salience_peak_concentration(),
            salience_bimodality=self.compute_salience_bimodality(),
            time_gap_salience_correlation=self.compute_time_gap_salience_correlation(),
            time_gap_variance=self.compute_time_gap_variance(),
        )

    def reset(self) -> None:
        """Reset all collected data."""
        self._saliences = []
        self._time_gaps = []
        self._rates = []
        self._adjustment_count = 0
        self._min_evidence = 0.7


# Singleton instance
_diagnostics_collector: Optional[DiagnosticsCollector] = None


def get_diagnostics_collector() -> DiagnosticsCollector:
    """Get the singleton diagnostics collector."""
    global _diagnostics_collector
    if _diagnostics_collector is None:
        _diagnostics_collector = DiagnosticsCollector()
    return _diagnostics_collector


def record_decision_diagnostics(
    decision_signals: Dict[str, Any],
    time_gap_seconds: float = 0.0
) -> DiagnosticSnapshot:
    """
    Record diagnostics from a decision and return current snapshot.

    Args:
        decision_signals: The signals dict from TopicDecision
        time_gap_seconds: Time since last message (if available)

    Returns:
        Current diagnostic snapshot
    """
    collector = get_diagnostics_collector()

    # Update from signals
    collector.update_controller_state(decision_signals)

    # Record time gap if provided
    if time_gap_seconds > 0:
        collector.record_time_gap(time_gap_seconds)

    return collector.get_snapshot()
