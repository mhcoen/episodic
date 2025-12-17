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
from episodic.debug_system import debug_print

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

    # Drift threshold for fast-path SUSPECT entry (hybrid trigger)
    # High embedding drift can trigger SUSPECT even if neural confidence is low
    # Set to None to disable drift fast-path
    drift_suspect_threshold: Optional[float] = 0.95

    # === Persistence requirement (two-sided test) ===
    # After evidence threshold is met, require K more turns with confidence
    # above return_threshold before committing. This catches transient digressions
    # where the user returns to the original topic.
    #
    # Commit requires BOTH:
    #   1. accumulated_evidence >= min_evidence
    #   2. last commit_persistence turns all had confidence >= return_threshold

    # Number of turns that must stay "away from topic" after evidence threshold
    # before we commit. 0 = commit immediately when evidence is met.
    commit_persistence: int = 1

    # Threshold for "returned to topic" - if confidence drops below this after
    # evidence threshold is met, we ABORT. Reuses abort_threshold by default.
    # Set to None to use abort_threshold.
    return_threshold: Optional[float] = None

    # Alternative return detection: relative drop from peak confidence.
    # If confidence drops to less than (peak * return_drop_ratio), ABORT.
    # This is more adaptive than an absolute threshold.
    # Example: peak=0.9, ratio=0.55 → abort if conf < 0.495
    # Set to None to use return_threshold instead.
    return_drop_ratio: Optional[float] = None

    # === Conditional cooldown ===
    # Don't apply persistence uniformly. High-confidence neural spikes are likely
    # real boundaries; only apply cooldown to intermediate cases that look like
    # potential digressions.
    #
    # Cooldown (K) applies when:
    #   - suspect_cause == "drift" (drift is noisy, needs confirmation), OR
    #   - confidence < high_conf_commit_threshold (intermediate band)
    #
    # Cooldown bypassed (immediate commit) when:
    #   - suspect_cause == "neural" AND confidence >= high_conf_commit_threshold
    #
    # Set to None to apply K uniformly (original behavior).
    high_conf_commit_threshold: Optional[float] = None

    # === Drift-triggered SUSPECT: confirmation requirements ===
    # Drift fires on surface-level changes (new entities, tangents) that may
    # not be real topic changes. Neural model must still confirm.

    # Minimum evidence for drift-triggered SUSPECT
    # Same as neural-triggered (1.2) since drift just triggers faster entry
    # while neural must still build evidence to commit
    drift_min_evidence: float = 1.2

    # Abort threshold for drift-triggered SUSPECT (higher = faster abort)
    # Default: 0.4 vs 0.3 for neural-triggered
    drift_abort_threshold: float = 0.4

    # Abort streak for drift-triggered SUSPECT
    # Higher than neural (4 vs 3) to give more time for neural to confirm
    # when drift triggers early on a genuine topic change
    drift_abort_streak: int = 4

    # === Drift + neural fast commit ===
    # When drift triggers SUSPECT AND neural confidence exceeds this threshold,
    # commit immediately (subject to min_gap). This handles cases where:
    # 1. Both signals agree (double confirmation)
    # 2. Neural model is sensitive to phrasing, so subsequent turns may give
    #    lower confidence despite the topic having clearly changed
    # Set to None to disable fast commit (require normal evidence accumulation)
    drift_neural_fast_commit_threshold: float = 0.9


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

    # Cause of SUSPECT entry: "neural" or "drift"
    # Used to apply cause-conditioned policy (stricter for drift)
    suspect_cause: Optional[str] = None

    # Recent confidence values for persistence check (two-sided test)
    # After evidence threshold is met, we track confidence values to ensure
    # the user hasn't "returned" to the original topic before committing.
    recent_confidences: List[float] = field(default_factory=list)

    # Whether evidence threshold has been met (waiting for persistence)
    evidence_met: bool = False

    # Peak confidence during SUSPECT for relative drop detection
    peak_confidence: float = 0.0

    # Entry confidence when SUSPECT was entered (for bypass decision)
    # Bypass uses entry confidence, not commit-time confidence
    suspect_entry_confidence: float = 0.0


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
        current_thread: Optional[Thread] = None,
        semantic_drift: Optional[float] = None,
        **kwargs
    ) -> TopicDecision:
        """
        Make topic decision with commitment policy using frozen reference state machine.

        State Machine:
        - STABLE: Normal sliding window comparison. If confidence >= suspect_threshold,
          transition to SUSPECT and freeze the "before" context.
        - SUSPECT: Compare new messages against frozen pre-change context. Accumulate
          evidence. COMMIT if evidence >= min_evidence. ABORT if confidence stays
          below abort_threshold for abort_streak consecutive turns.

        Hybrid Trigger:
        - If semantic_drift >= drift_suspect_threshold, enter SUSPECT immediately
          even if neural confidence is low. This catches sharp topic transitions
          that the neural model misses on the first turn.

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
            if current_node_id is None:
                debug_print(
                    f"WARNING: messages[{self._state.current_idx}] has no node_id! "
                    f"Keys: {list(messages[self._state.current_idx].keys())}",
                    category="topic"
                )

        # === STATE MACHINE ===

        if self._state.state == CommitState.STABLE:
            # Normal operation: sliding window comparison
            base_decision = self.base_strategy.get_decision(query, messages, current_thread)
            confidence = base_decision.confidence_score

            # Check for drift fast-path (hybrid trigger)
            drift_triggered = False
            if (semantic_drift is not None and
                self.policy.drift_suspect_threshold is not None and
                semantic_drift >= self.policy.drift_suspect_threshold):
                drift_triggered = True

            debug_print(
                f"[STABLE] idx={self._state.current_idx} conf={confidence:.3f} "
                f"threshold={self.policy.suspect_threshold}"
                + (f" drift={semantic_drift:.3f}" if semantic_drift is not None else ""),
                category="topic"
            )

            # Enter SUSPECT via drift fast-path OR neural confidence
            # Drift catches sharp topic shifts that neural model misses on first turn
            if drift_triggered:
                # DRIFT-triggered SUSPECT entry: freeze context and seed with neural conf
                self._enter_suspect_drift(messages, current_node_id, semantic_drift, confidence)
                debug_print(
                    f"[STABLE→SUSPECT] Drift fast-path! drift={semantic_drift:.3f} "
                    f">= {self.policy.drift_suspect_threshold}, neural_conf={confidence:.3f}",
                    category="topic"
                )

                # === Drift + neural fast commit ===
                # When both drift AND neural give high confidence on the same turn,
                # commit immediately (subject to min_gap). This handles cases where
                # the neural model is sensitive to phrasing on subsequent turns.
                fast_commit_thresh = self.policy.drift_neural_fast_commit_threshold
                if fast_commit_thresh is not None and confidence >= fast_commit_thresh:
                    gap = self._turns_since_boundary()
                    if gap >= self.policy.min_gap:
                        debug_print(
                            f"[STABLE→COMMIT] Drift+neural fast commit! "
                            f"drift={semantic_drift:.3f}, neural={confidence:.3f} >= {fast_commit_thresh}, gap={gap}",
                            category="topic"
                        )
                        return self._commit_boundary(base_decision, start_time)
                    else:
                        debug_print(
                            f"[STABLE→SUSPECT] Drift+neural high but gap={gap} < min_gap={self.policy.min_gap}",
                            category="topic"
                        )

                return self._build_suspect_decision(
                    base_decision, start_time, "entered SUSPECT via drift",
                    semantic_drift=semantic_drift, drift_triggered=True
                )

            elif confidence >= self.policy.suspect_threshold:
                # NEURAL-triggered SUSPECT entry: use confidence as initial evidence
                self._enter_suspect(messages, current_node_id, confidence)

                # Check if we can commit immediately (single high signal + min_gap satisfied)
                if self._state.accumulated_evidence >= self.policy.min_evidence:
                    gap = self._turns_since_boundary()
                    if gap >= self.policy.min_gap:
                        debug_print(
                            f"[STABLE→COMMIT] Immediate commit! evidence={self._state.accumulated_evidence:.3f} "
                            f">= {self.policy.min_evidence}, gap={gap}",
                            category="topic"
                        )
                        return self._commit_boundary(base_decision, start_time)
                    else:
                        debug_print(
                            f"[STABLE→SUSPECT] Evidence sufficient but gap={gap} < min_gap={self.policy.min_gap}",
                            category="topic"
                        )

                # Not enough evidence yet, stay in SUSPECT
                debug_print(
                    f"[STABLE→SUSPECT] Entered SUSPECT, evidence={self._state.accumulated_evidence:.3f} "
                    f"< {self.policy.min_evidence}",
                    category="topic"
                )
                return self._build_suspect_decision(
                    base_decision, start_time, "entered SUSPECT",
                    semantic_drift=semantic_drift, drift_triggered=False
                )

            # No detection, stay STABLE
            return self._build_stable_decision(
                base_decision, start_time,
                semantic_drift=semantic_drift, drift_triggered=False
            )

        else:  # SUSPECT state
            # Compare against frozen reference context with frozen straddle message
            # This preserves the training format: after = [straddle_msg, query]
            base_decision = self.base_strategy.get_decision(
                query, messages, current_thread,
                frozen_before_context=self._state.frozen_before,
                frozen_straddle_msg=self._state.frozen_straddle_msg
            )
            confidence = base_decision.confidence_score

            # Get cause-conditioned thresholds
            # Drift-triggered SUSPECT uses stricter requirements
            is_drift_caused = self._state.suspect_cause == "drift"
            if is_drift_caused:
                abort_threshold = self.policy.drift_abort_threshold
                abort_streak = self.policy.drift_abort_streak
                min_evidence = self.policy.drift_min_evidence
            else:
                abort_threshold = self.policy.abort_threshold
                abort_streak = self.policy.abort_streak
                min_evidence = self.policy.min_evidence

            # Return threshold for persistence check (two-sided test)
            return_threshold = (
                self.policy.return_threshold if self.policy.return_threshold is not None
                else abort_threshold
            )

            debug_print(
                f"[SUSPECT:{self._state.suspect_cause}] idx={self._state.current_idx} "
                f"conf_vs_frozen={confidence:.3f} (abort_thresh={abort_threshold}, "
                f"evidence_met={self._state.evidence_met})",
                category="topic"
            )

            # Track recent confidences for persistence check
            self._state.recent_confidences.append(confidence)
            # Keep only the last K+1 values (K for persistence check + current)
            max_history = self.policy.commit_persistence + 1
            if len(self._state.recent_confidences) > max_history:
                self._state.recent_confidences = self._state.recent_confidences[-max_history:]

            # Track peak confidence for relative drop detection
            if confidence > self._state.peak_confidence:
                self._state.peak_confidence = confidence

            # Accumulate evidence with decay
            prev_evidence = self._state.accumulated_evidence
            self._state.accumulated_evidence = (
                prev_evidence * self.policy.evidence_decay + confidence
            )

            debug_print(
                f"  evidence: {prev_evidence:.3f}*{self.policy.evidence_decay} + "
                f"{confidence:.3f} = {self._state.accumulated_evidence:.3f} "
                f"(need {min_evidence}), peak={self._state.peak_confidence:.3f}",
                category="topic"
            )

            # Check if evidence threshold is met
            if self._state.accumulated_evidence >= min_evidence:
                self._state.evidence_met = True

            # Calculate effective return threshold for abort decisions
            # Priority: return_drop_ratio > return_threshold > abort_threshold
            if self.policy.return_drop_ratio is not None:
                effective_return_thresh = self._state.peak_confidence * self.policy.return_drop_ratio
            else:
                effective_return_thresh = return_threshold

            # Check ABORT condition: confidence below threshold?
            # Two cases for ABORT:
            # 1. Standard: low confidence for abort_streak consecutive turns
            # 2. Return detection: evidence_met but confidence dropped below return_threshold
            if confidence < abort_threshold:
                self._state.low_confidence_streak += 1
                debug_print(
                    f"  low conf streak: {self._state.low_confidence_streak}/{abort_streak}",
                    category="topic"
                )
                if self._state.low_confidence_streak >= abort_streak:
                    # ABORT: return to original topic, false alarm
                    debug_print(
                        f"[SUSPECT→ABORT] {self._state.suspect_cause}-triggered, "
                        f"low confidence for {abort_streak} turns",
                        category="topic"
                    )
                    self._abort_suspect()
                    return self._build_stable_decision(
                        base_decision, start_time,
                        f"ABORT: {self._state.suspect_cause}-triggered false alarm",
                        semantic_drift=semantic_drift
                    )
            else:
                self._state.low_confidence_streak = 0

            # Return detection: if evidence was met but confidence dropped, ABORT
            # This catches transient digressions where user returns to original topic
            if self._state.evidence_met and confidence < effective_return_thresh:
                debug_print(
                    f"[SUSPECT→ABORT] Return detected: evidence_met but "
                    f"conf={confidence:.3f} < return_thresh={effective_return_thresh:.3f} "
                    f"(peak={self._state.peak_confidence:.3f})",
                    category="topic"
                )
                self._abort_suspect()
                return self._build_stable_decision(
                    base_decision, start_time,
                    f"ABORT: returned to topic (conf={confidence:.2f} < {effective_return_thresh:.2f})",
                    semantic_drift=semantic_drift
                )

            # Check COMMIT condition (two-sided test):
            # 1. accumulated_evidence >= min_evidence
            # 2. min_gap satisfied
            # 3. Persistence check (conditional cooldown)
            if self._state.evidence_met:
                gap = self._turns_since_boundary()
                if gap >= self.policy.min_gap:
                    # Determine effective K based on conditional cooldown
                    # High-confidence neural spikes bypass cooldown (K=0)
                    # Drift-triggered or intermediate confidence requires cooldown
                    #
                    # IMPORTANT: Bypass uses ENTRY confidence, not commit-time confidence.
                    # This prevents gaming by transient spikes during SUSPECT.
                    k = self.policy.commit_persistence
                    bypass_cooldown = False

                    if self.policy.high_conf_commit_threshold is not None:
                        # Conditional cooldown: bypass K for high-confidence neural ENTRY
                        entry_conf = self._state.suspect_entry_confidence
                        if (self._state.suspect_cause == "neural" and
                            entry_conf >= self.policy.high_conf_commit_threshold):
                            bypass_cooldown = True
                            debug_print(
                                f"  [COOLDOWN BYPASS] neural + entry_conf={entry_conf:.3f} >= "
                                f"{self.policy.high_conf_commit_threshold}",
                                category="topic"
                            )

                    if k == 0 or bypass_cooldown:
                        # No persistence required, commit immediately
                        persistence_met = True
                    else:
                        # Check last K confidences (excluding current which we just added)
                        recent = self._state.recent_confidences[:-1] if len(self._state.recent_confidences) > 1 else []
                        persistence_met = (
                            len(recent) >= k and
                            all(c >= effective_return_thresh for c in recent[-k:])
                        )

                    if persistence_met:
                        node_preview = self._state.suspect_start_node_id[:8] if self._state.suspect_start_node_id else "None"
                        debug_print(
                            f"[SUSPECT→COMMIT] {self._state.suspect_cause}-triggered, "
                            f"evidence={self._state.accumulated_evidence:.3f} >= {min_evidence}, "
                            f"gap={gap}, persistence={k} (bypass={bypass_cooldown}), boundary_node={node_preview}...",
                            category="topic"
                        )
                        return self._commit_boundary(base_decision, start_time)
                    else:
                        debug_print(
                            f"[SUSPECT] evidence met, waiting for persistence "
                            f"(need {k} turns >= {effective_return_thresh:.2f})",
                            category="topic"
                        )
                else:
                    debug_print(
                        f"[SUSPECT] evidence sufficient ({self._state.accumulated_evidence:.3f}) "
                        f"but gap={gap} < min_gap={self.policy.min_gap}",
                        category="topic"
                    )

            # Stay in SUSPECT, continue accumulating
            return self._build_suspect_decision(
                base_decision, start_time, "accumulating evidence",
                semantic_drift=semantic_drift
            )

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
        self._state.suspect_cause = "neural"
        self._state.suspect_entry_confidence = initial_confidence  # For bypass decision

        if node_id is None:
            debug_print(
                "⚠️ ENTER_SUSPECT: node_id is None! Boundary placement will be incorrect.",
                category="topic"
            )

        # Capture the "before" context - the 4 messages BEFORE the straddle
        # messages layout: [...history..., straddle, query]
        # straddle = messages[-2], query = messages[-1]
        # We want messages[-6:-2] (4 messages before straddle)
        if len(messages) >= 6:
            start_idx = len(messages) - 6
            # Try to start with a user message for better topic context
            if start_idx > 0 and messages[start_idx].get('role') == 'assistant':
                start_idx -= 1
            end_idx = len(messages) - 2  # Stop before straddle
            self._state.frozen_before = messages[max(0, start_idx):end_idx]
            # Ensure we don't take more than 4
            if len(self._state.frozen_before) > 4:
                self._state.frozen_before = self._state.frozen_before[-4:]
        else:
            # Not enough messages for full window
            self._state.frozen_before = messages[:-2] if len(messages) >= 2 else []

        # Capture the "straddle" message - the last message before the suspected change
        # This preserves the training format: after = [straddle_msg, query]
        # Note: messages[-1] is the CURRENT query (already appended by topic_management)
        # We want messages[-2] which is the last message of the OLD topic
        if len(messages) >= 2:
            self._state.frozen_straddle_msg = messages[-2]
        elif messages:
            self._state.frozen_straddle_msg = messages[-1]
        else:
            self._state.frozen_straddle_msg = None

        debug_print(
            f"[ENTER_SUSPECT] node={node_id[:8] if node_id else 'None'}... "
            f"initial_evidence={initial_confidence:.3f}, "
            f"frozen {len(self._state.frozen_before)} msgs",
            category="topic"
        )

    def _enter_suspect_drift(
        self,
        messages: List[Dict[str, Any]],
        node_id: Optional[str],
        drift_value: float,
        neural_confidence: float = 0.0
    ):
        """
        Enter SUSPECT state via drift fast-path.

        Drift-triggered entry seeds evidence with the neural confidence at entry.
        This gives the neural model a head start on evidence accumulation while
        still requiring confirmation.

        Drift-triggered SUSPECT uses stricter thresholds:
        - Higher min_evidence (drift_min_evidence vs min_evidence)
        - Higher abort threshold (drift_abort_threshold vs abort_threshold)
        - Lower abort streak (drift_abort_streak vs abort_streak)
        """
        self._state.state = CommitState.SUSPECT
        self._state.suspect_start_node_id = node_id
        self._state.accumulated_evidence = neural_confidence  # Seed with neural conf
        self._state.low_confidence_streak = 0
        self._state.suspect_cause = "drift"
        self._state.suspect_entry_confidence = neural_confidence

        if node_id is None:
            debug_print(
                "⚠️ ENTER_SUSPECT_DRIFT: node_id is None! Boundary placement will be incorrect.",
                category="topic"
            )

        # Capture the "before" context - same logic as _enter_suspect
        if len(messages) >= 6:
            start_idx = len(messages) - 6
            if start_idx > 0 and messages[start_idx].get('role') == 'assistant':
                start_idx -= 1
            end_idx = len(messages) - 2
            self._state.frozen_before = messages[max(0, start_idx):end_idx]
            if len(self._state.frozen_before) > 4:
                self._state.frozen_before = self._state.frozen_before[-4:]
        else:
            self._state.frozen_before = messages[:-2] if len(messages) >= 2 else []

        # Capture the "straddle" message
        if len(messages) >= 2:
            self._state.frozen_straddle_msg = messages[-2]
        elif messages:
            self._state.frozen_straddle_msg = messages[-1]
        else:
            self._state.frozen_straddle_msg = None

        debug_print(
            f"[ENTER_SUSPECT_DRIFT] node={node_id[:8] if node_id else 'None'}... "
            f"drift={drift_value:.3f}, evidence={neural_confidence:.3f} (seeded), "
            f"frozen {len(self._state.frozen_before)} msgs",
            category="topic"
        )

    def _abort_suspect(self):
        """Abort SUSPECT state: false alarm, return to STABLE."""
        debug_print(
            f"ABORT: {self._state.suspect_cause}-triggered, "
            f"low confidence for {self._state.low_confidence_streak} turns, "
            f"evidence was {self._state.accumulated_evidence:.3f}",
            category="topic"
        )
        self._state.state = CommitState.STABLE
        self._state.frozen_before = None
        self._state.frozen_straddle_msg = None
        self._state.suspect_start_node_id = None
        self._state.accumulated_evidence = 0.0
        self._state.low_confidence_streak = 0
        self._state.suspect_cause = None
        self._state.recent_confidences = []
        self._state.evidence_met = False
        self._state.peak_confidence = 0.0
        self._state.suspect_entry_confidence = 0.0

    def _commit_boundary(self, base_decision: TopicDecision, start_time: float) -> TopicDecision:
        """Commit to the boundary detected when entering SUSPECT."""
        import time
        boundary_node_id = self._state.suspect_start_node_id
        evidence = self._state.accumulated_evidence

        if boundary_node_id is None:
            debug_print(
                "⚠️ COMMIT: boundary_node_id is None! Will fall back to current message.",
                category="topic"
            )
        else:
            debug_print(
                f"COMMIT: boundary_node_id={boundary_node_id[:8]}...",
                category="topic"
            )

        # Update tracking
        self._state.last_boundary_idx = self._state.current_idx

        # Capture cause before reset for signals
        suspect_cause = self._state.suspect_cause

        # Reset to STABLE
        self._state.state = CommitState.STABLE
        self._state.frozen_before = None
        self._state.frozen_straddle_msg = None
        self._state.suspect_start_node_id = None
        self._state.accumulated_evidence = 0.0
        self._state.low_confidence_streak = 0
        self._state.suspect_cause = None
        self._state.recent_confidences = []
        self._state.evidence_met = False
        self._state.peak_confidence = 0.0
        self._state.suspect_entry_confidence = 0.0

        # Use cause-conditioned min_evidence for reasoning
        actual_min_evidence = (
            self.policy.drift_min_evidence if suspect_cause == "drift"
            else self.policy.min_evidence
        )
        reasoning = f"COMMIT: {suspect_cause}-triggered, evidence={evidence:.2f} >= {actual_min_evidence}"
        if boundary_node_id:
            reasoning += f", boundary at {boundary_node_id[:8]}..."

        signals = dict(base_decision.signals)
        signals.update({
            'state': 'COMMIT',
            'accumulated_evidence': evidence,
            'min_evidence': actual_min_evidence,
            'boundary_node_id': boundary_node_id,
            'committed': True,
            'suspect_cause': suspect_cause,
        })

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
        extra_reason: str = "",
        semantic_drift: Optional[float] = None,
        drift_triggered: bool = False
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
        if semantic_drift is not None:
            signals['semantic_drift'] = semantic_drift
            signals['drift_triggered'] = drift_triggered

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
        reason: str,
        semantic_drift: Optional[float] = None,
        drift_triggered: bool = False
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
        if semantic_drift is not None:
            signals['semantic_drift'] = semantic_drift
            signals['drift_triggered'] = drift_triggered

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
