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

    # Threshold to enter SUSPECT state (first spike detection)
    suspect_threshold: float = 0.5

    # Threshold below which we consider returning to original topic (for ABORT)
    abort_threshold: float = 0.3

    # How many consecutive low-confidence turns before aborting SUSPECT state
    abort_streak: int = 3

    # Decay factor for evidence over turns (0-1)
    # 1.0 = no decay, 0.5 = halve each turn
    evidence_decay: float = 0.8

    # Minimum accumulated evidence to commit
    min_evidence: float = 1.2


# State machine states
class CommitState:
    STABLE = "STABLE"    # Normal operation, sliding window
    SUSPECT = "SUSPECT"  # Potential topic change detected, frozen reference


@dataclass
class CommitmentState:
    """Tracks state for commitment decisions."""

    # Current state machine state
    state: str = CommitState.STABLE

    # Message index of last committed boundary (None if none committed)
    last_boundary_idx: Optional[int] = None

    # Current message index
    current_idx: int = 0

    # === SUSPECT state fields ===
    # Frozen "before" context captured when entering SUSPECT
    frozen_before: Optional[List[Dict[str, Any]]] = None

    # Frozen "straddle" message - the last message before the suspected topic change
    # This preserves the training format: after = [straddle_msg, query]
    frozen_straddle_msg: Optional[Dict[str, Any]] = None

    # Node ID where suspicion began (for boundary emit on commit)
    suspect_start_node_id: Optional[str] = None

    # Accumulated evidence while in SUSPECT state
    accumulated_evidence: float = 0.0

    # Count of consecutive low-confidence turns (for ABORT logic)
    low_confidence_streak: int = 0


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
                - suspect_threshold: Override policy.suspect_threshold
                - abort_threshold: Override policy.abort_threshold
                - abort_streak: Override policy.abort_streak
                - min_evidence: Override policy.min_evidence
        """
        self.base_strategy = base_strategy
        self.policy = policy or CommitmentPolicy()
        self._state = CommitmentState()

        # Apply param overrides
        params = params or {}
        if 'min_gap' in params:
            self.policy.min_gap = params['min_gap']
        if 'suspect_threshold' in params:
            self.policy.suspect_threshold = params['suspect_threshold']
        if 'abort_threshold' in params:
            self.policy.abort_threshold = params['abort_threshold']
        if 'abort_streak' in params:
            self.policy.abort_streak = params['abort_streak']
        if 'min_evidence' in params:
            self.policy.min_evidence = params['min_evidence']

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
        Make topic decision with commitment policy using frozen reference state machine.

        State Machine:
        - STABLE: Normal sliding window comparison. If confidence >= suspect_threshold,
          transition to SUSPECT and freeze the "before" context.
        - SUSPECT: Compare new messages against frozen pre-change context. Accumulate
          evidence. COMMIT if evidence >= min_evidence. ABORT if confidence stays
          below abort_threshold for abort_streak consecutive turns.

        This ensures "are we still far from the old topic?" rather than "did we
        change topics again?" - the frozen reference gives stable score semantics
        during evidence accumulation.
        """
        import time
        start_time = time.time()

        # Update state tracking
        self._state.current_idx = len(messages) - 1 if messages else 0

        # Get the node_id for the current message
        current_node_id = None
        if messages and self._state.current_idx < len(messages):
            current_node_id = messages[self._state.current_idx].get('node_id')

        # === STATE MACHINE ===

        if self._state.state == CommitState.STABLE:
            # Normal operation: sliding window comparison
            base_decision = self.base_strategy.get_decision(query, messages, current_thread)
            confidence = base_decision.confidence_score

            # Check if we should enter SUSPECT state
            gap = self._turns_since_boundary()
            if confidence >= self.policy.suspect_threshold and gap >= self.policy.min_gap:
                # Enter SUSPECT: freeze the "before" context
                self._enter_suspect(messages, current_node_id, confidence)

                # Check if we can commit immediately (single high signal)
                if self._state.accumulated_evidence >= self.policy.min_evidence:
                    return self._commit_boundary(base_decision, start_time)

                # Not enough evidence yet, stay in SUSPECT
                return self._build_suspect_decision(base_decision, start_time, "entered SUSPECT")

            # No detection, stay STABLE
            return self._build_stable_decision(base_decision, start_time)

        else:  # SUSPECT state
            # Compare against frozen reference context with frozen straddle message
            # This preserves the training format: after = [straddle_msg, query]
            base_decision = self.base_strategy.get_decision(
                query, messages, current_thread,
                frozen_before_context=self._state.frozen_before,
                frozen_straddle_msg=self._state.frozen_straddle_msg
            )
            confidence = base_decision.confidence_score

            # Accumulate evidence with decay
            self._state.accumulated_evidence = (
                self._state.accumulated_evidence * self.policy.evidence_decay + confidence
            )

            # Check ABORT condition: confidence below threshold?
            if confidence < self.policy.abort_threshold:
                self._state.low_confidence_streak += 1
                if self._state.low_confidence_streak >= self.policy.abort_streak:
                    # ABORT: return to original topic, false alarm
                    self._abort_suspect()
                    return self._build_stable_decision(base_decision, start_time, "ABORT: returned to topic")
            else:
                self._state.low_confidence_streak = 0

            # Check COMMIT condition: enough evidence?
            if self._state.accumulated_evidence >= self.policy.min_evidence:
                return self._commit_boundary(base_decision, start_time)

            # Stay in SUSPECT, continue accumulating
            return self._build_suspect_decision(base_decision, start_time, "accumulating evidence")

    def _enter_suspect(
        self,
        messages: List[Dict[str, Any]],
        node_id: Optional[str],
        initial_confidence: float
    ):
        """Enter SUSPECT state: freeze the before-context and straddle message."""
        self._state.state = CommitState.SUSPECT
        self._state.suspect_start_node_id = node_id
        self._state.accumulated_evidence = initial_confidence
        self._state.low_confidence_streak = 0

        # Capture the "before" context - the 4 messages before the current one
        # This is what NeuralStrategy would use as its before_messages
        if len(messages) >= 5:
            start_idx = len(messages) - 5
            # Try to start with a user message
            if messages[start_idx].get('role') == 'assistant' and start_idx > 0:
                start_idx -= 1
            end_idx = min(start_idx + 4, len(messages) - 1)
            self._state.frozen_before = messages[start_idx:end_idx]
        else:
            self._state.frozen_before = messages[:-1] if len(messages) > 1 else []

        # Capture the "straddle" message - the last message before the suspected change
        # This preserves the training format: after = [straddle_msg, query]
        # The straddle message is the last message in the current context (messages[-1])
        if messages:
            self._state.frozen_straddle_msg = messages[-1]
        else:
            self._state.frozen_straddle_msg = None

        logger.debug(
            f"Entered SUSPECT: frozen {len(self._state.frozen_before)} messages + straddle, "
            f"initial evidence={initial_confidence:.3f}"
        )

    def _abort_suspect(self):
        """Abort SUSPECT state: false alarm, return to STABLE."""
        logger.debug(
            f"ABORT: low confidence for {self._state.low_confidence_streak} turns, "
            f"evidence was {self._state.accumulated_evidence:.3f}"
        )
        self._state.state = CommitState.STABLE
        self._state.frozen_before = None
        self._state.frozen_straddle_msg = None
        self._state.suspect_start_node_id = None
        self._state.accumulated_evidence = 0.0
        self._state.low_confidence_streak = 0

    def _commit_boundary(self, base_decision: TopicDecision, start_time: float) -> TopicDecision:
        """Commit to the boundary detected when entering SUSPECT."""
        import time
        boundary_node_id = self._state.suspect_start_node_id
        evidence = self._state.accumulated_evidence

        # Update tracking
        self._state.last_boundary_idx = self._state.current_idx

        # Reset to STABLE
        self._state.state = CommitState.STABLE
        self._state.frozen_before = None
        self._state.frozen_straddle_msg = None
        self._state.suspect_start_node_id = None
        self._state.accumulated_evidence = 0.0
        self._state.low_confidence_streak = 0

        reasoning = f"COMMIT: evidence={evidence:.2f} >= {self.policy.min_evidence}"
        if boundary_node_id:
            reasoning += f", boundary at {boundary_node_id[:8]}..."

        signals = dict(base_decision.signals)
        signals.update({
            'state': 'COMMIT',
            'accumulated_evidence': evidence,
            'min_evidence': self.policy.min_evidence,
            'boundary_node_id': boundary_node_id,
            'committed': True,
        })

        logger.debug(reasoning)

        return TopicDecision(
            topic_changed=True,
            new_thread=base_decision.new_thread,
            thread_links=base_decision.thread_links,
            retrieved_context=base_decision.retrieved_context,
            confidence=Confidence.HIGH,
            confidence_score=evidence,
            reasoning=reasoning,
            signals=signals,
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _build_stable_decision(
        self,
        base_decision: TopicDecision,
        start_time: float,
        extra_reason: str = ""
    ) -> TopicDecision:
        """Build decision for STABLE state (no commit)."""
        import time
        reasoning = f"STABLE: {base_decision.reasoning}"
        if extra_reason:
            reasoning = f"{extra_reason}. {reasoning}"

        signals = dict(base_decision.signals)
        signals.update({
            'state': 'STABLE',
            'turns_since_boundary': self._turns_since_boundary(),
            'suspect_threshold': self.policy.suspect_threshold,
            'committed': False,
        })

        return TopicDecision(
            topic_changed=False,
            new_thread=None,
            thread_links=base_decision.thread_links,
            retrieved_context=base_decision.retrieved_context,
            confidence=Confidence.LOW,
            confidence_score=base_decision.confidence_score,
            reasoning=reasoning,
            signals=signals,
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _build_suspect_decision(
        self,
        base_decision: TopicDecision,
        start_time: float,
        reason: str
    ) -> TopicDecision:
        """Build decision for SUSPECT state (accumulating evidence)."""
        import time
        reasoning = (
            f"SUSPECT: {reason}, evidence={self._state.accumulated_evidence:.2f}, "
            f"need={self.policy.min_evidence}"
        )

        signals = dict(base_decision.signals)
        signals.update({
            'state': 'SUSPECT',
            'accumulated_evidence': self._state.accumulated_evidence,
            'min_evidence': self.policy.min_evidence,
            'low_confidence_streak': self._state.low_confidence_streak,
            'abort_streak': self.policy.abort_streak,
            'committed': False,
        })

        return TopicDecision(
            topic_changed=False,
            new_thread=None,
            thread_links=base_decision.thread_links,
            retrieved_context=base_decision.retrieved_context,
            confidence=Confidence.LOW,
            confidence_score=base_decision.confidence_score,
            reasoning=reasoning,
            signals=signals,
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _turns_since_boundary(self) -> int:
        """Get number of turns since last committed boundary."""
        if self._state.last_boundary_idx is None:
            return self._state.current_idx  # Treat start as boundary
        return self._state.current_idx - self._state.last_boundary_idx



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
