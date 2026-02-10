"""
Commitment policy wrapper for topic detection strategies.

Adds hysteresis to prevent oversegmentation by requiring:
1. Minimum gap between committed boundaries
2. Evidence accumulation across multiple turns
3. Higher threshold for commitment than detection

This separates salience detection (what the underlying strategy does)
from commitment (deciding when to materialize a boundary).
"""

import logging
from typing import Dict, List, Any, Optional

from episodic.topics.strategy import (
    TopicStrategy,
    TopicDecision,
    Thread,
    ThreadLink,
    RetrievedContext,
    Confidence,
)
from episodic.topics.strategies.commitment_policy import (
    CommitmentPolicy, CommitState, CommitmentState,
)
from episodic.debug_system import debug_print

logger = logging.getLogger(__name__)


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
        user_turns_in_topic: Optional[int] = None,
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

        Commit Gate:
        - min_user_turns_for_commit blocks COMMIT until enough user history exists.
        - SUSPECT entry and evidence accumulation still happen normally.

        This ensures "are we still far from the old topic?" rather than "did we
        change topics again?" - the frozen reference gives stable score semantics
        during evidence accumulation.
        """
        import time
        start_time = time.time()

        # Update state tracking
        self._state.current_idx = len(messages) - 1 if messages else 0

        # Increment internal user turns counter (anchored to confirmed boundaries)
        # This counts user turns since last confirmed boundary, NOT since topic naming
        self._state.user_turns_since_boundary += 1
        user_turns = self._state.user_turns_since_boundary

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
                # commit immediately (subject to min_gap only). This is an "obvious boundary"
                # case that bypasses the cold-start user_turns gate entirely.
                fast_commit_thresh = self.policy.drift_neural_fast_commit_threshold
                if fast_commit_thresh is not None and confidence >= fast_commit_thresh:
                    gap = self._turns_since_boundary()
                    if gap >= self.policy.min_gap:
                        debug_print(
                            f"[STABLE→COMMIT] Drift+neural fast commit! "
                            f"drift={semantic_drift:.3f}, neural={confidence:.3f} >= {fast_commit_thresh}, "
                            f"gap={gap} (user_turns gate bypassed for obvious boundary)",
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
                # Cold start gate: only blocks before first committed boundary
                if self._state.accumulated_evidence >= self.policy.min_evidence:
                    gap = self._turns_since_boundary()
                    min_turns = self.policy.min_user_turns_for_commit
                    # Cold start gate: only apply user_turns check before first commit
                    cold_start_blocked = (
                        not self._state.has_committed_boundary and
                        user_turns < min_turns
                    )
                    if gap >= self.policy.min_gap and not cold_start_blocked:
                        debug_print(
                            f"[STABLE→COMMIT] Immediate commit! evidence={self._state.accumulated_evidence:.3f} "
                            f">= {self.policy.min_evidence}, gap={gap}",
                            category="topic"
                        )
                        return self._commit_boundary(base_decision, start_time)
                    else:
                        reason_parts = []
                        if gap < self.policy.min_gap:
                            reason_parts.append(f"gap={gap} < min_gap={self.policy.min_gap}")
                        if cold_start_blocked:
                            reason_parts.append(f"cold_start: user_turns={user_turns} < min_turns={min_turns}")
                        debug_print(
                            f"[STABLE→SUSPECT] Evidence sufficient but {', '.join(reason_parts)}",
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
                    abort_cause = self._state.suspect_cause
                    debug_print(
                        f"[SUSPECT→ABORT] {abort_cause}-triggered, "
                        f"low confidence for {abort_streak} turns",
                        category="topic"
                    )
                    self._abort_suspect()
                    return self._build_stable_decision(
                        base_decision, start_time,
                        f"ABORT: {abort_cause}-triggered false alarm",
                        semantic_drift=semantic_drift,
                        aborted=True,
                        abort_reason=f"low_confidence_streak:{abort_cause}",
                    )
            else:
                self._state.low_confidence_streak = 0

            # Return detection: if evidence was met but confidence dropped, ABORT
            # This catches transient digressions where user returns to original topic
            if self._state.evidence_met and confidence < effective_return_thresh:
                abort_cause = self._state.suspect_cause
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
                    semantic_drift=semantic_drift,
                    aborted=True,
                    abort_reason=f"return_detected:{abort_cause}",
                )

            # Check COMMIT condition (two-sided test):
            # 1. accumulated_evidence >= min_evidence
            # 2. min_gap satisfied
            # 3. Cold start gate (only before first committed boundary)
            # 4. Persistence check (conditional cooldown)
            if self._state.evidence_met:
                gap = self._turns_since_boundary()
                min_turns = self.policy.min_user_turns_for_commit
                # Cold start gate: only apply user_turns check before first commit
                cold_start_blocked = (
                    not self._state.has_committed_boundary and
                    user_turns < min_turns
                )
                if gap >= self.policy.min_gap and not cold_start_blocked:
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
                        # Neural commit drift gate: for neural-triggered SUSPECT,
                        # require sufficient drift to confirm this is a real topic change
                        # (not just a subtopic shift like carbonara within pasta topic)
                        if (self._state.suspect_cause == "neural" and
                            self.policy.neural_commit_drift_threshold is not None):
                            drift_val = semantic_drift if semantic_drift is not None else 0.0
                            if drift_val < self.policy.neural_commit_drift_threshold:
                                # Drift gate blocks: this is a false alarm (subtopic, not topic change)
                                # ABORT instead of staying in SUSPECT to avoid getting stuck
                                # If drift rises later, we'll re-enter SUSPECT via drift trigger
                                debug_print(
                                    f"[SUSPECT→ABORT] neural drift gate: drift={drift_val:.3f} < "
                                    f"threshold={self.policy.neural_commit_drift_threshold}, "
                                    f"treating as false alarm",
                                    category="topic"
                                )
                                self._abort_suspect()
                                return self._build_stable_decision(
                                    base_decision, start_time,
                                    f"ABORT: neural drift gate (drift={drift_val:.3f} < {self.policy.neural_commit_drift_threshold})",
                                    semantic_drift=semantic_drift,
                                    aborted=True,
                                    abort_reason="neural_drift_gate",
                                )

                        node_preview = self._state.suspect_start_node_id[:8] if self._state.suspect_start_node_id else "None"
                        debug_print(
                            f"[SUSPECT→COMMIT] {self._state.suspect_cause}-triggered, "
                            f"evidence={self._state.accumulated_evidence:.3f} >= {min_evidence}, "
                            f"gap={gap}, user_turns={user_turns}, persistence={k} (bypass={bypass_cooldown}), "
                            f"boundary_node={node_preview}...",
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
                    reason_parts = []
                    if gap < self.policy.min_gap:
                        reason_parts.append(f"gap={gap} < min_gap={self.policy.min_gap}")
                    if cold_start_blocked:
                        reason_parts.append(f"cold_start: user_turns={user_turns} < min_turns={min_turns}")
                    debug_print(
                        f"[SUSPECT] evidence sufficient ({self._state.accumulated_evidence:.3f}) "
                        f"but {', '.join(reason_parts)}",
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
        self._state.has_committed_boundary = True  # Cold start gate no longer applies

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
        # Reset user turns counter - anchored to this confirmed boundary
        # Starts at 1 because this message triggered the commit
        self._state.user_turns_since_boundary = 1

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
        drift_triggered: bool = False,
        aborted: bool = False,
        abort_reason: Optional[str] = None,
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
            'aborted': aborted,
        })
        if abort_reason:
            signals['abort_reason'] = abort_reason
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
