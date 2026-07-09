"""Commitment-policy helper methods (suspect transitions, decision builders).

Mixin split out of commitment_strategy.py; CommitmentPolicyStrategy inherits it,
so these run on the instance (self._policy/_state/... and callbacks from
get_decision resolve via inheritance).
"""

import logging
from typing import Dict, List, Any, Optional

from episodic.topics.strategy import (
    TopicStrategy, TopicDecision, Thread, ThreadLink, RetrievedContext, Confidence,
)
from episodic.topics.strategies.commitment_policy import (
    CommitmentPolicy, CommitState, CommitmentState,
)
from episodic.debug_system import debug_print

logger = logging.getLogger(__name__)


class _CommitmentHelpersMixin:
    """SUSPECT state transitions and STABLE/SUSPECT/COMMIT decision builders."""

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
