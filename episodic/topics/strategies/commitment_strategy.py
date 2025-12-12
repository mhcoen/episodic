"""
Commitment policy wrapper for topic detection strategies.

Adds hysteresis to prevent oversegmentation by requiring:
1. Minimum gap between committed boundaries
2. Evidence accumulation across multiple turns
3. Higher threshold for commitment than detection

This separates salience detection (what the underlying strategy does)
from commitment (deciding when to materialize a boundary).

Also provides AdaptiveCommitmentStrategy that self-adjusts based on
observed segmentation rate.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import deque

from episodic.topics.strategy import (
    TopicStrategy,
    TopicDecision,
    Thread,
    ThreadLink,
    RetrievedContext,
    Confidence,
)

logger = logging.getLogger(__name__)


@dataclass
class CommitmentPolicy:
    """Configuration for boundary commitment decisions."""

    # Minimum turns between committed boundaries
    min_gap: int = 3

    # Number of turns to accumulate evidence before committing
    evidence_window: int = 2

    # Higher threshold for commitment than detection
    # If None, uses underlying strategy's threshold
    commitment_threshold: Optional[float] = None

    # Decay factor for evidence over turns (0-1)
    # 1.0 = no decay, 0.5 = halve each turn
    evidence_decay: float = 0.8

    # Minimum accumulated evidence to commit
    min_evidence: float = 1.2


@dataclass
class CommitmentState:
    """Tracks state for commitment decisions."""

    # Message index of last committed boundary (None if none committed)
    last_boundary_idx: Optional[int] = None

    # Evidence buffer: list of (message_idx, confidence) for recent detections
    evidence_buffer: List[tuple] = field(default_factory=list)

    # Current message index
    current_idx: int = 0


class CommitmentPolicyStrategy(TopicStrategy):
    """
    Wrapper strategy that adds commitment policy to any base strategy.

    The base strategy detects potential boundaries (salience).
    This wrapper decides when to commit to them based on:
    - Gap from previous boundary
    - Accumulated evidence
    - Higher commitment threshold

    This prevents oversegmentation while preserving detection sensitivity.

    Example usage:
        base = NeuralStrategy({'granularity': 'fine'})
        wrapped = CommitmentPolicyStrategy(base, CommitmentPolicy(min_gap=3))

        # wrapped.get_decision() now applies commitment policy
    """

    def __init__(
        self,
        base_strategy: TopicStrategy,
        policy: Optional[CommitmentPolicy] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize commitment wrapper.

        Args:
            base_strategy: The underlying detection strategy
            policy: Commitment policy configuration
            params: Optional override params:
                - min_gap: Override policy.min_gap
                - evidence_window: Override policy.evidence_window
                - commitment_threshold: Override policy.commitment_threshold
        """
        self.base_strategy = base_strategy
        self.policy = policy or CommitmentPolicy()
        self._state = CommitmentState()

        # Apply param overrides
        params = params or {}
        if 'min_gap' in params:
            self.policy.min_gap = params['min_gap']
        if 'evidence_window' in params:
            self.policy.evidence_window = params['evidence_window']
        if 'commitment_threshold' in params:
            self.policy.commitment_threshold = params['commitment_threshold']

    @property
    def name(self) -> str:
        return f"Committed({self.base_strategy.name})"

    @property
    def version(self) -> str:
        return f"1.0.0+{self.base_strategy.version}"

    def reset(self):
        """Reset commitment state (e.g., for new conversation)."""
        self._state = CommitmentState()

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """
        Segment conversation with commitment policy applied.

        Runs base strategy then filters boundaries based on policy.
        """
        # Get base segmentation
        base_threads = self.base_strategy.segment_conversation(messages)

        if len(base_threads) <= 1:
            return base_threads

        # Filter boundaries based on commitment policy
        committed_threads = []
        last_boundary = -self.policy.min_gap  # Allow first boundary

        for i, thread in enumerate(base_threads):
            # First thread always included
            if i == 0:
                committed_threads.append(thread)
                continue

            # Check gap from last committed boundary
            thread_start = self._get_thread_start_idx(thread, messages)
            gap = thread_start - last_boundary

            if gap >= self.policy.min_gap:
                committed_threads.append(thread)
                last_boundary = thread_start
            else:
                # Merge with previous thread
                if committed_threads:
                    prev = committed_threads[-1]
                    prev.messages.extend(thread.messages)

        return committed_threads

    def _get_thread_start_idx(
        self,
        thread: Thread,
        all_messages: List[Dict[str, Any]]
    ) -> int:
        """Get the message index where a thread starts."""
        if not thread.messages:
            return 0

        first_msg = thread.messages[0]
        node_id = first_msg.get('node_id')

        for i, msg in enumerate(all_messages):
            if msg.get('node_id') == node_id:
                return i

        return 0

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Delegate to base strategy."""
        return self.base_strategy.detect_thread_link(
            query, recent_context, past_threads
        )

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Delegate to base strategy."""
        return self.base_strategy.retrieve_context(
            query, thread_links, threads, max_tokens
        )

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make topic decision with commitment policy applied.

        Process:
        1. Get detection from base strategy (salience)
        2. If detected, add to evidence buffer
        3. Decay old evidence
        4. Check commitment criteria:
           - Sufficient gap from last boundary
           - Sufficient accumulated evidence
           - Confidence above commitment threshold
        5. Commit boundary only if all criteria met
        """
        # Get base detection
        base_decision = self.base_strategy.get_decision(query, messages, current_thread)

        # Update state
        self._state.current_idx = len(messages)

        # Add to evidence buffer if boundary detected
        if base_decision.topic_changed or base_decision.confidence_score > 0.3:
            self._state.evidence_buffer.append(
                (self._state.current_idx, base_decision.confidence_score)
            )

        # Decay and prune evidence buffer
        self._decay_evidence()

        # Calculate accumulated evidence
        accumulated_evidence = self._calculate_evidence()

        # Check commitment criteria
        should_commit, commit_reason = self._should_commit(
            base_decision, accumulated_evidence
        )

        # Build signals combining base + commitment
        signals = dict(base_decision.signals)
        signals.update({
            'accumulated_evidence': accumulated_evidence,
            'evidence_buffer_size': len(self._state.evidence_buffer),
            'turns_since_boundary': self._turns_since_boundary(),
            'min_gap': self.policy.min_gap,
            'commitment_threshold': self.policy.commitment_threshold,
            'base_detected': base_decision.topic_changed,
            'committed': should_commit,
        })

        # Build reasoning
        if should_commit:
            reasoning = (
                f"Committed boundary: {commit_reason}. "
                f"Evidence={accumulated_evidence:.2f}, "
                f"Gap={self._turns_since_boundary()}"
            )
        else:
            reasoning = (
                f"Detection not committed: {commit_reason}. "
                f"Base: {base_decision.reasoning}"
            )

        # Update last boundary if committing
        if should_commit:
            self._state.last_boundary_idx = self._state.current_idx
            self._state.evidence_buffer = []  # Clear after commit

        return TopicDecision(
            topic_changed=should_commit,
            new_thread=base_decision.new_thread if should_commit else None,
            thread_links=base_decision.thread_links,
            retrieved_context=base_decision.retrieved_context,
            confidence=base_decision.confidence if should_commit else Confidence.LOW,
            confidence_score=accumulated_evidence if should_commit else base_decision.confidence_score,
            reasoning=reasoning,
            signals=signals,
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=base_decision.processing_time_ms,
        )

    def _decay_evidence(self):
        """Apply decay to evidence buffer and remove old entries."""
        if not self._state.evidence_buffer:
            return

        new_buffer = []
        for idx, conf in self._state.evidence_buffer:
            # Calculate age in turns
            age = self._state.current_idx - idx

            # Remove if older than evidence window
            if age > self.policy.evidence_window:
                continue

            # Apply decay based on age
            decayed_conf = conf * (self.policy.evidence_decay ** age)

            # Keep if still meaningful
            if decayed_conf > 0.1:
                new_buffer.append((idx, decayed_conf))

        self._state.evidence_buffer = new_buffer

    def _calculate_evidence(self) -> float:
        """Calculate total accumulated evidence."""
        return sum(conf for _, conf in self._state.evidence_buffer)

    def _turns_since_boundary(self) -> int:
        """Get number of turns since last committed boundary."""
        if self._state.last_boundary_idx is None:
            return self._state.current_idx  # Treat start as boundary
        return self._state.current_idx - self._state.last_boundary_idx

    def _should_commit(
        self,
        base_decision: TopicDecision,
        accumulated_evidence: float
    ) -> tuple:
        """
        Determine if we should commit to a boundary.

        Returns:
            (should_commit, reason)
        """
        # Check 1: Gap from last boundary
        gap = self._turns_since_boundary()
        if gap < self.policy.min_gap:
            return False, f"too soon (gap={gap} < {self.policy.min_gap})"

        # Check 2: Base strategy must have detected something
        if not base_decision.topic_changed and base_decision.confidence_score < 0.3:
            return False, "no detection from base strategy"

        # Check 3: Commitment threshold (if set)
        commit_threshold = self.policy.commitment_threshold
        if commit_threshold is not None:
            if base_decision.confidence_score < commit_threshold:
                return False, f"below threshold ({base_decision.confidence_score:.2f} < {commit_threshold})"

        # Check 4: Sufficient accumulated evidence
        if accumulated_evidence < self.policy.min_evidence:
            return False, f"insufficient evidence ({accumulated_evidence:.2f} < {self.policy.min_evidence})"

        return True, f"criteria met (evidence={accumulated_evidence:.2f}, gap={gap})"


@dataclass
class AdaptivePolicy:
    """Configuration for adaptive commitment."""

    # Target boundaries per N messages (ideal segmentation rate)
    # E.g., target_rate=0.1 means 1 boundary per 10 messages
    target_rate: float = 0.1

    # Window size for measuring current rate
    rate_window: int = 50

    # How aggressively to adjust (0-1, higher = faster adaptation)
    adaptation_rate: float = 0.3

    # Bounds for min_gap adjustment
    min_gap_bounds: tuple = (1, 6)

    # Bounds for min_evidence adjustment
    min_evidence_bounds: tuple = (0.3, 1.5)

    # Tolerance band around target rate (no adjustment within this band)
    # E.g., 0.2 means ±20% of target rate is acceptable
    tolerance: float = 0.2


class AdaptiveCommitmentStrategy(TopicStrategy):
    """
    Self-adjusting commitment strategy that adapts to observed segmentation rate.

    Monitors the ratio of boundaries to messages and adjusts commitment
    parameters to keep the rate near a target:

    - If oversegmenting (rate > target): Tighten commitment (increase min_gap, min_evidence)
    - If undersegmenting (rate < target): Loosen commitment (decrease min_gap, min_evidence)
    - Within tolerance band: No adjustment

    This allows the strategy to automatically calibrate for different
    conversation styles and domains without manual tuning.

    Example usage:
        base = NeuralStrategy({'granularity': 'fine'})
        adaptive = AdaptiveCommitmentStrategy(base, target_rate=0.1)

        # Strategy will self-adjust to produce ~1 boundary per 10 messages
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
            initial_policy: Starting commitment policy (will be adjusted)
            params: Optional override params:
                - target_rate: Override adaptive_policy.target_rate
                - adaptation_rate: Override adaptive_policy.adaptation_rate
        """
        self.base_strategy = base_strategy
        self.adaptive = adaptive_policy or AdaptivePolicy()

        # Start with medium commitment policy
        self.policy = initial_policy or CommitmentPolicy(
            min_gap=2,
            evidence_window=2,
            min_evidence=0.7,
            evidence_decay=0.85,
        )

        self._state = CommitmentState()

        # Tracking for rate calculation
        self._message_count = 0
        self._boundary_count = 0
        self._recent_boundaries = deque(maxlen=self.adaptive.rate_window)

        # Apply param overrides
        params = params or {}
        if 'target_rate' in params:
            self.adaptive.target_rate = params['target_rate']
        if 'adaptation_rate' in params:
            self.adaptive.adaptation_rate = params['adaptation_rate']

    @property
    def name(self) -> str:
        return f"Adaptive({self.base_strategy.name})"

    @property
    def version(self) -> str:
        return f"1.0.0+{self.base_strategy.version}"

    def reset(self):
        """Reset state for new conversation."""
        self._state = CommitmentState()
        self._message_count = 0
        self._boundary_count = 0
        self._recent_boundaries.clear()

    def _current_rate(self) -> float:
        """Calculate current boundary rate from recent history."""
        if len(self._recent_boundaries) < 10:
            # Not enough data yet, return target to avoid adjustment
            return self.adaptive.target_rate

        boundaries_in_window = sum(self._recent_boundaries)
        return boundaries_in_window / len(self._recent_boundaries)

    def _adapt_policy(self):
        """Adjust policy based on observed rate vs target."""
        current_rate = self._current_rate()
        target = self.adaptive.target_rate

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

        if ratio > upper_bound:
            # Oversegmenting: tighten commitment
            adjustment = 1.0 + alpha * (ratio - 1.0)

            # Increase min_gap
            new_gap = min(
                self.adaptive.min_gap_bounds[1],
                int(self.policy.min_gap * adjustment)
            )
            self.policy.min_gap = max(self.adaptive.min_gap_bounds[0], new_gap)

            # Increase min_evidence
            new_evidence = min(
                self.adaptive.min_evidence_bounds[1],
                self.policy.min_evidence * adjustment
            )
            self.policy.min_evidence = max(
                self.adaptive.min_evidence_bounds[0], new_evidence
            )

            logger.debug(
                f"Adaptive: tightening (rate={current_rate:.3f} > target={target:.3f}), "
                f"min_gap={self.policy.min_gap}, min_evidence={self.policy.min_evidence:.2f}"
            )

        elif ratio < lower_bound:
            # Undersegmenting: loosen commitment
            adjustment = 1.0 - alpha * (1.0 - ratio)

            # Decrease min_gap
            new_gap = max(
                self.adaptive.min_gap_bounds[0],
                int(self.policy.min_gap * adjustment)
            )
            self.policy.min_gap = min(self.adaptive.min_gap_bounds[1], new_gap)

            # Decrease min_evidence
            new_evidence = max(
                self.adaptive.min_evidence_bounds[0],
                self.policy.min_evidence * adjustment
            )
            self.policy.min_evidence = min(
                self.adaptive.min_evidence_bounds[1], new_evidence
            )

            logger.debug(
                f"Adaptive: loosening (rate={current_rate:.3f} < target={target:.3f}), "
                f"min_gap={self.policy.min_gap}, min_evidence={self.policy.min_evidence:.2f}"
            )

    def segment_conversation(self, messages: List[Dict[str, Any]]) -> List[Thread]:
        """Delegate to base strategy (adaptation happens in get_decision)."""
        return self.base_strategy.segment_conversation(messages)

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Delegate to base strategy."""
        return self.base_strategy.detect_thread_link(
            query, recent_context, past_threads
        )

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Delegate to base strategy."""
        return self.base_strategy.retrieve_context(
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

        Process:
        1. Get detection from base strategy
        2. Apply current commitment policy
        3. Track boundary/message counts
        4. Periodically adapt policy based on observed rate
        """
        # Get base detection
        base_decision = self.base_strategy.get_decision(query, messages, current_thread)

        # Update state
        self._state.current_idx = len(messages)
        self._message_count += 1

        # Add to evidence buffer if boundary detected
        if base_decision.topic_changed or base_decision.confidence_score > 0.3:
            self._state.evidence_buffer.append(
                (self._state.current_idx, base_decision.confidence_score)
            )

        # Decay evidence
        self._decay_evidence()

        # Calculate accumulated evidence
        accumulated_evidence = self._calculate_evidence()

        # Check commitment criteria
        should_commit, commit_reason = self._should_commit(
            base_decision, accumulated_evidence
        )

        # Track for rate calculation
        self._recent_boundaries.append(1 if should_commit else 0)
        if should_commit:
            self._boundary_count += 1

        # Adapt policy periodically (every 10 messages)
        if self._message_count % 10 == 0:
            self._adapt_policy()

        # Build signals
        signals = dict(base_decision.signals)
        signals.update({
            'accumulated_evidence': accumulated_evidence,
            'evidence_buffer_size': len(self._state.evidence_buffer),
            'turns_since_boundary': self._turns_since_boundary(),
            'current_min_gap': self.policy.min_gap,
            'current_min_evidence': self.policy.min_evidence,
            'current_rate': self._current_rate(),
            'target_rate': self.adaptive.target_rate,
            'base_detected': base_decision.topic_changed,
            'committed': should_commit,
        })

        # Build reasoning
        if should_commit:
            reasoning = (
                f"Committed boundary: {commit_reason}. "
                f"Rate={self._current_rate():.3f} (target={self.adaptive.target_rate:.3f})"
            )
        else:
            reasoning = (
                f"Detection not committed: {commit_reason}. "
                f"Base: {base_decision.reasoning}"
            )

        # Update last boundary if committing
        if should_commit:
            self._state.last_boundary_idx = self._state.current_idx
            self._state.evidence_buffer = []

        return TopicDecision(
            topic_changed=should_commit,
            new_thread=base_decision.new_thread if should_commit else None,
            thread_links=base_decision.thread_links,
            retrieved_context=base_decision.retrieved_context,
            confidence=base_decision.confidence if should_commit else Confidence.LOW,
            confidence_score=accumulated_evidence if should_commit else base_decision.confidence_score,
            reasoning=reasoning,
            signals=signals,
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=base_decision.processing_time_ms,
        )

    def _decay_evidence(self):
        """Apply decay to evidence buffer."""
        if not self._state.evidence_buffer:
            return

        new_buffer = []
        for idx, conf in self._state.evidence_buffer:
            age = self._state.current_idx - idx
            if age > self.policy.evidence_window:
                continue
            decayed_conf = conf * (self.policy.evidence_decay ** age)
            if decayed_conf > 0.1:
                new_buffer.append((idx, decayed_conf))

        self._state.evidence_buffer = new_buffer

    def _calculate_evidence(self) -> float:
        """Calculate total accumulated evidence."""
        return sum(conf for _, conf in self._state.evidence_buffer)

    def _turns_since_boundary(self) -> int:
        """Get number of turns since last committed boundary."""
        if self._state.last_boundary_idx is None:
            return self._state.current_idx
        return self._state.current_idx - self._state.last_boundary_idx

    def _should_commit(
        self,
        base_decision: TopicDecision,
        accumulated_evidence: float
    ) -> tuple:
        """Determine if we should commit to a boundary."""
        gap = self._turns_since_boundary()
        if gap < self.policy.min_gap:
            return False, f"too soon (gap={gap} < {self.policy.min_gap})"

        if not base_decision.topic_changed and base_decision.confidence_score < 0.3:
            return False, "no detection from base strategy"

        if self.policy.commitment_threshold is not None:
            if base_decision.confidence_score < self.policy.commitment_threshold:
                return False, f"below threshold ({base_decision.confidence_score:.2f})"

        if accumulated_evidence < self.policy.min_evidence:
            return False, f"insufficient evidence ({accumulated_evidence:.2f} < {self.policy.min_evidence:.2f})"

        return True, f"criteria met (evidence={accumulated_evidence:.2f}, gap={gap})"
