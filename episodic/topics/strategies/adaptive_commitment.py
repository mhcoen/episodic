"""
Adaptive commitment strategy for topic detection.

Self-adjusts min_evidence based on observed segmentation rate.
Wraps CommitmentPolicyStrategy with rate-based adaptation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from collections import deque

from episodic.topics.strategy import (
    TopicStrategy,
    TopicDecision,
    Thread,
    ThreadLink,
    RetrievedContext,
)
from episodic.topics.strategies.commitment_policy import CommitmentPolicy
from episodic.topics.strategies.commitment_strategy import CommitmentPolicyStrategy

logger = logging.getLogger(__name__)


@dataclass
class AdaptivePolicy:
    """
    Configuration for adaptive commitment.

    Rate Semantics:
        target_rate is defined as "committed boundaries per canonical message position".
        Canonical positions are between-message indices [1, T-1] for T messages.
        This is speaker-agnostic: a dialogue with 10 messages has 9 potential
        boundary positions, so target_rate=0.1 means ~1 boundary per dialogue.

        To convert from typical annotation density:
        - If dataset has avg 3 boundaries per 30-message dialogue: rate = 3/30 = 0.1
        - BOR relates to this: BOR = observed_rate / gold_rate

    Single-Knob Control:
        Only min_evidence is adapted. min_gap stays fixed.
        This avoids coupled dynamics and oscillation.
    """

    # Target boundaries per canonical message position
    # E.g., target_rate=0.1 means 1 boundary per 10 messages
    # Typical values: 0.08-0.15 for task-oriented, 0.05-0.10 for open-domain
    target_rate: float = 0.1

    # Window size for measuring current rate (in messages)
    rate_window: int = 50

    # How aggressively to adjust min_evidence (0-1, higher = faster adaptation)
    # Recommended: 0.1-0.2 for stability
    adaptation_rate: float = 0.15

    # Bounds for min_evidence adjustment (single knob)
    min_evidence_bounds: tuple = (0.3, 1.5)

    # Fixed min_gap (not adapted, for stability)
    fixed_min_gap: int = 2

    # Tolerance band around target rate (no adjustment within this band)
    # E.g., 0.25 means ±25% of target rate is acceptable
    tolerance: float = 0.25

    # Cold-start warmup: number of messages to observe before adapting
    # During warmup, uses initial min_evidence; after warmup, begins adaptation
    warmup_messages: int = 10

    # If True, initialize min_evidence from warmup salience distribution
    warmup_calibrate: bool = True


class AdaptiveCommitmentStrategy(TopicStrategy):
    """
    Self-adjusting commitment strategy that adapts to observed segmentation rate.

    Wraps CommitmentPolicyStrategy (with frozen reference state machine) and
    dynamically adjusts min_evidence based on observed boundary rate.

    Design Principles:
        1. Single-knob control: Only min_evidence is adapted, min_gap stays fixed.
           This avoids coupled dynamics and oscillation.
        2. Cold-start warmup: Observes base salience for N messages before adapting.
           This removes startup transient in short dialogues.
        3. Canonical rate: Rate computed on between-message positions, speaker-agnostic.

    Behavior:
        - If oversegmenting (rate > target): Increase min_evidence (tighten)
        - If undersegmenting (rate < target): Decrease min_evidence (loosen)
        - Within tolerance band: No adjustment

    Use Cases:
        - Static policy (CommitmentPolicyStrategy) is better for offline batch evaluation
        - Adaptive policy is better for streaming/live usage (steady-state self-calibration)

    Example usage:
        base = NeuralStrategy({'granularity': 'fine'})
        adaptive = AdaptiveCommitmentStrategy(base, AdaptivePolicy(target_rate=0.1))
    """

    def __init__(
        self,
        base_strategy: TopicStrategy,
        adaptive_policy: Optional[AdaptivePolicy] = None,
        initial_policy: Optional[CommitmentPolicy] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize adaptive commitment wrapper.

        Args:
            base_strategy: The underlying detection strategy
            adaptive_policy: Adaptive behavior configuration
            initial_policy: Starting commitment policy (min_evidence will be adapted)
            params: Optional override params:
                - target_rate: Override adaptive_policy.target_rate
                - adaptation_rate: Override adaptive_policy.adaptation_rate
        """
        self.adaptive = adaptive_policy or AdaptivePolicy()

        # Build commitment policy - min_gap is fixed, only min_evidence adapts
        self.policy = initial_policy or CommitmentPolicy(
            min_gap=self.adaptive.fixed_min_gap,
            suspect_threshold=0.5,
            abort_threshold=0.3,
            abort_streak=3,
            evidence_decay=0.85,
            min_evidence=0.7,
        )
        # Ensure min_gap matches fixed value
        self.policy.min_gap = self.adaptive.fixed_min_gap

        # Wrap base strategy with commitment policy (frozen reference state machine)
        self._commitment_strategy = CommitmentPolicyStrategy(base_strategy, self.policy)

        # Tracking for rate calculation
        self._message_count = 0
        self._boundary_count = 0
        self._recent_boundaries = deque(maxlen=self.adaptive.rate_window)

        # Warmup tracking
        self._warmup_saliences: List[float] = []
        self._warmup_complete = False

        # Volatility metrics
        self._adjustment_count = 0
        self._rate_history: List[float] = []

        # Apply param overrides
        params = params or {}
        if 'target_rate' in params:
            self.adaptive.target_rate = params['target_rate']
        if 'adaptation_rate' in params:
            self.adaptive.adaptation_rate = params['adaptation_rate']

    @property
    def name(self) -> str:
        return f"Adaptive({self._commitment_strategy.base_strategy.name})"

    @property
    def version(self) -> str:
        return f"2.0.0+{self._commitment_strategy.base_strategy.version}"

    def reset(self):
        """Reset state for new conversation."""
        self._commitment_strategy.reset()
        self._message_count = 0
        self._boundary_count = 0
        self._recent_boundaries.clear()
        self._warmup_saliences = []
        self._warmup_complete = False
        self._adjustment_count = 0
        self._rate_history = []
        # Reset min_evidence to initial value
        self.policy.min_evidence = 0.7

    def _current_rate(self) -> float:
        """
        Calculate current boundary rate from recent history.

        Rate is boundaries per canonical message position (speaker-agnostic).
        """
        if len(self._recent_boundaries) < 10:
            # Not enough data yet, return target to avoid adjustment
            return self.adaptive.target_rate

        boundaries_in_window = sum(self._recent_boundaries)
        return boundaries_in_window / len(self._recent_boundaries)

    def _rate_volatility(self) -> float:
        """Calculate standard deviation of rate over recent history."""
        if len(self._rate_history) < 5:
            return 0.0

        import statistics
        return statistics.stdev(self._rate_history[-20:])

    def _complete_warmup(self):
        """
        Complete warmup phase and optionally calibrate min_evidence.

        If warmup_calibrate is True, sets min_evidence so that approximately
        target_rate fraction of warmup saliences would commit.
        """
        self._warmup_complete = True

        if not self.adaptive.warmup_calibrate or not self._warmup_saliences:
            return

        # Sort saliences and find threshold that gives target_rate
        sorted_saliences = sorted(self._warmup_saliences, reverse=True)
        target_count = max(1, int(len(sorted_saliences) * self.adaptive.target_rate))

        if target_count < len(sorted_saliences):
            # Set min_evidence to the salience at the target percentile
            threshold = sorted_saliences[target_count - 1]
            # Clamp to bounds
            self.policy.min_evidence = max(
                self.adaptive.min_evidence_bounds[0],
                min(self.adaptive.min_evidence_bounds[1], threshold)
            )
            logger.debug(
                f"Adaptive warmup: calibrated min_evidence={self.policy.min_evidence:.2f} "
                f"from {len(sorted_saliences)} samples"
            )

    def _adapt_policy(self):
        """
        Adjust min_evidence based on observed rate vs target.

        Single-knob control: only min_evidence changes, min_gap stays fixed.
        """
        # Don't adapt during warmup
        if not self._warmup_complete:
            return

        current_rate = self._current_rate()
        target = self.adaptive.target_rate

        # Track rate history for volatility calculation
        self._rate_history.append(current_rate)

        # Calculate rate ratio
        if target > 0:
            ratio = current_rate / target
        else:
            ratio = 1.0

        # Check if within tolerance band
        lower_bound = 1.0 - self.adaptive.tolerance
        upper_bound = 1.0 + self.adaptive.tolerance

        if lower_bound <= ratio <= upper_bound:
            # Within tolerance, no adjustment needed
            return

        # Calculate adjustment factor
        alpha = self.adaptive.adaptation_rate
        self._adjustment_count += 1

        if ratio > upper_bound:
            # Oversegmenting: increase min_evidence (tighten)
            adjustment = 1.0 + alpha * (ratio - 1.0)

            new_evidence = min(
                self.adaptive.min_evidence_bounds[1],
                self.policy.min_evidence * adjustment
            )
            self.policy.min_evidence = max(
                self.adaptive.min_evidence_bounds[0], new_evidence
            )

            logger.debug(
                f"Adaptive: tightening (rate={current_rate:.3f} > target={target:.3f}), "
                f"min_evidence={self.policy.min_evidence:.2f}"
            )

        elif ratio < lower_bound:
            # Undersegmenting: decrease min_evidence (loosen)
            adjustment = 1.0 - alpha * (1.0 - ratio)

            new_evidence = max(
                self.adaptive.min_evidence_bounds[0],
                self.policy.min_evidence * adjustment
            )
            self.policy.min_evidence = min(
                self.adaptive.min_evidence_bounds[1], new_evidence
            )

            logger.debug(
                f"Adaptive: loosening (rate={current_rate:.3f} < target={target:.3f}), "
                f"min_evidence={self.policy.min_evidence:.2f}"
            )

    def segment_conversation(self, messages: List[Dict[str, Any]]) -> List[Thread]:
        """Delegate to commitment strategy."""
        return self._commitment_strategy.segment_conversation(messages)

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Delegate to commitment strategy."""
        return self._commitment_strategy.detect_thread_link(
            query, recent_context, past_threads
        )

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Delegate to commitment strategy."""
        return self._commitment_strategy.retrieve_context(
            query, thread_links, threads, max_tokens
        )

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make topic decision with adaptive commitment.

        Delegates to CommitmentPolicyStrategy for the actual decision logic
        (frozen reference state machine), then tracks rate and adapts min_evidence.
        """
        self._message_count += 1

        # Warmup phase: collect base saliences for calibration
        if not self._warmup_complete:
            # Get raw base decision for salience tracking
            base_decision = self._commitment_strategy.base_strategy.get_decision(
                query, messages, current_thread
            )
            self._warmup_saliences.append(base_decision.confidence_score)
            if self._message_count >= self.adaptive.warmup_messages:
                self._complete_warmup()

        # Get decision from commitment strategy (frozen reference state machine)
        decision = self._commitment_strategy.get_decision(query, messages, current_thread)

        # Track for rate calculation
        self._recent_boundaries.append(1 if decision.topic_changed else 0)
        if decision.topic_changed:
            self._boundary_count += 1

        # Adapt policy periodically (every 10 messages, after warmup)
        if self._message_count % 10 == 0 and self._warmup_complete:
            self._adapt_policy()

        # Add adaptive metrics to signals
        signals = dict(decision.signals)
        signals.update({
            'current_min_evidence': self.policy.min_evidence,
            'current_rate': self._current_rate(),
            'target_rate': self.adaptive.target_rate,
            'rate_volatility': self._rate_volatility(),
            'adjustment_count': self._adjustment_count,
            'warmup_complete': self._warmup_complete,
        })

        # Update reasoning with adaptive info
        reasoning = decision.reasoning
        if decision.topic_changed:
            reasoning += f" Rate={self._current_rate():.3f} (target={self.adaptive.target_rate:.3f})"

        return TopicDecision(
            topic_changed=decision.topic_changed,
            new_thread=decision.new_thread,
            thread_links=decision.thread_links,
            retrieved_context=decision.retrieved_context,
            confidence=decision.confidence,
            confidence_score=decision.confidence_score,
            reasoning=reasoning,
            signals=signals,
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=decision.processing_time_ms,
        )
