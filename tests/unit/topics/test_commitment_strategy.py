"""
Unit tests for CommitmentPolicyStrategy.

Tests the hysteresis mechanism that prevents oversegmentation by:
1. Enforcing minimum gap between boundaries
2. Accumulating evidence before committing
3. Applying higher commitment threshold than detection threshold
"""

import pytest
from episodic.topics.strategy import TopicDecision, Thread, Confidence
from episodic.topics.strategies.commitment_strategy import (
    CommitmentPolicyStrategy,
    CommitmentPolicy,
    CommitmentState,
)


class MockDetectionStrategy:
    """Mock strategy that returns configured detection results."""

    def __init__(self, detections: dict = None):
        """
        Args:
            detections: Dict mapping message index to (detected, confidence)
        """
        self.name = "MockStrategy"
        self.version = "1.0.0"
        self.detections = detections or {}
        self._call_count = 0

    def get_decision(self, query: str, messages: list, current_thread=None):
        """Return predetermined detection based on message count."""
        idx = len(messages)
        detected, confidence = self.detections.get(idx, (False, 0.1))
        self._call_count += 1

        return TopicDecision(
            topic_changed=detected,
            new_thread=None,
            thread_links=[],
            retrieved_context=None,
            confidence=Confidence.HIGH if detected else Confidence.LOW,
            confidence_score=confidence,
            reasoning=f"Mock: detected={detected}",
            signals={'mock_idx': idx},
            strategy_name=self.name,
            strategy_version=self.version,
        )

    def segment_conversation(self, messages):
        return [Thread(id="t0", name="mock", start_node_id="0", messages=messages)]

    def detect_thread_link(self, query, recent_context, past_threads):
        return []

    def retrieve_context(self, query, thread_links, threads, max_tokens=2000):
        return None


class TestCommitmentPolicyBasics:
    """Test CommitmentPolicy dataclass."""

    def test_default_values(self):
        """CommitmentPolicy has sensible defaults."""
        policy = CommitmentPolicy()
        assert policy.min_gap == 3
        assert policy.evidence_window == 2
        assert policy.commitment_threshold is None
        assert policy.evidence_decay == 0.8
        assert policy.min_evidence == 1.2

    def test_custom_values(self):
        """CommitmentPolicy accepts custom values."""
        policy = CommitmentPolicy(
            min_gap=5,
            evidence_window=3,
            commitment_threshold=0.7,
            evidence_decay=0.9,
            min_evidence=2.0
        )
        assert policy.min_gap == 5
        assert policy.evidence_window == 3
        assert policy.commitment_threshold == 0.7
        assert policy.evidence_decay == 0.9
        assert policy.min_evidence == 2.0


class TestMinGapEnforcement:
    """Test that minimum gap between boundaries is enforced."""

    def test_first_boundary_allowed(self):
        """First boundary can be committed immediately."""
        mock = MockDetectionStrategy({
            3: (True, 0.9),
        })
        wrapped = CommitmentPolicyStrategy(
            mock,
            CommitmentPolicy(min_gap=3, min_evidence=0.5)
        )

        # Simulate 3 messages then detection
        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision = wrapped.get_decision("new topic?", messages)

        assert decision.topic_changed is True
        assert 'committed' in decision.signals
        assert decision.signals['committed'] is True

    def test_second_boundary_blocked_if_too_soon(self):
        """Second boundary blocked if within min_gap."""
        mock = MockDetectionStrategy({
            3: (True, 0.9),
            4: (True, 0.9),  # Too soon
        })
        policy = CommitmentPolicy(min_gap=3, min_evidence=0.5)
        wrapped = CommitmentPolicyStrategy(mock, policy)

        # First boundary at message 3
        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision1 = wrapped.get_decision("new topic", messages)
        assert decision1.topic_changed is True

        # Second boundary at message 4 (gap=1, less than 3)
        messages.append({'role': 'assistant', 'content': 'response'})
        decision2 = wrapped.get_decision("another topic", messages)
        assert decision2.topic_changed is False
        assert "too soon" in decision2.reasoning

    def test_second_boundary_allowed_after_gap(self):
        """Second boundary allowed after sufficient gap."""
        mock = MockDetectionStrategy({
            3: (True, 0.9),
            7: (True, 0.9),  # Gap of 4 >= min_gap of 3
        })
        policy = CommitmentPolicy(min_gap=3, min_evidence=0.5)
        wrapped = CommitmentPolicyStrategy(mock, policy)

        # First boundary
        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision1 = wrapped.get_decision("topic 1", messages)
        assert decision1.topic_changed is True

        # Add more messages
        for i in range(4):
            messages.append({'role': 'user', 'content': f'msg{i+3}'})

        # Second boundary should be allowed (gap = 4)
        decision2 = wrapped.get_decision("topic 2", messages)
        assert decision2.topic_changed is True


class TestEvidenceAccumulation:
    """Test evidence accumulation before commitment."""

    def test_single_weak_detection_not_committed(self):
        """Single weak detection doesn't meet evidence threshold."""
        mock = MockDetectionStrategy({
            3: (True, 0.5),  # Below min_evidence of 1.2
        })
        policy = CommitmentPolicy(min_gap=1, min_evidence=1.2)
        wrapped = CommitmentPolicyStrategy(mock, policy)

        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision = wrapped.get_decision("weak topic", messages)

        assert decision.topic_changed is False
        assert "insufficient evidence" in decision.reasoning

    def test_accumulated_detections_committed(self):
        """Multiple detections accumulate to meet evidence threshold."""
        # Two detections of 0.7 each should accumulate > 1.2
        mock = MockDetectionStrategy({
            3: (True, 0.7),
            4: (True, 0.7),
        })
        policy = CommitmentPolicy(min_gap=1, evidence_window=3, min_evidence=1.2)
        wrapped = CommitmentPolicyStrategy(mock, policy)

        # First detection adds to buffer
        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision1 = wrapped.get_decision("maybe topic", messages)
        # Might not commit yet (only 0.7)
        assert decision1.signals['accumulated_evidence'] == pytest.approx(0.7, rel=0.1)

        # Second detection should push over threshold
        messages.append({'role': 'assistant', 'content': 'response'})
        decision2 = wrapped.get_decision("confirm topic", messages)
        # Now we have two detections (0.7 + 0.7*0.8 = 1.26 > 1.2)
        assert decision2.topic_changed is True

    def test_evidence_decays_over_turns(self):
        """Evidence decays if not reinforced."""
        mock = MockDetectionStrategy({
            3: (True, 0.8),
            4: (False, 0.1),  # No reinforcement
            5: (False, 0.1),
            6: (False, 0.1),
        })
        policy = CommitmentPolicy(
            min_gap=1,
            evidence_window=2,
            evidence_decay=0.5,
            min_evidence=0.5
        )
        wrapped = CommitmentPolicyStrategy(mock, policy)

        # Initial detection
        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision1 = wrapped.get_decision("topic", messages)

        # Several turns without reinforcement
        for i in range(3):
            messages.append({'role': 'user', 'content': f'filler{i}'})
            decision = wrapped.get_decision("continuing", messages)

        # Evidence should have decayed/expired
        assert wrapped._calculate_evidence() < 0.5


class TestCommitmentThreshold:
    """Test commitment threshold enforcement."""

    def test_detection_below_threshold_not_committed(self):
        """Detection below commitment threshold not committed."""
        mock = MockDetectionStrategy({
            3: (True, 0.6),  # Below threshold
        })
        policy = CommitmentPolicy(
            min_gap=1,
            commitment_threshold=0.7,
            min_evidence=0.5
        )
        wrapped = CommitmentPolicyStrategy(mock, policy)

        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision = wrapped.get_decision("topic", messages)

        assert decision.topic_changed is False
        assert "below threshold" in decision.reasoning

    def test_detection_above_threshold_committed(self):
        """Detection above commitment threshold committed."""
        mock = MockDetectionStrategy({
            3: (True, 0.8),
        })
        policy = CommitmentPolicy(
            min_gap=1,
            commitment_threshold=0.7,
            min_evidence=0.5
        )
        wrapped = CommitmentPolicyStrategy(mock, policy)

        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision = wrapped.get_decision("topic", messages)

        assert decision.topic_changed is True


class TestStateManagement:
    """Test state management across decisions."""

    def test_reset_clears_state(self):
        """reset() clears all state."""
        mock = MockDetectionStrategy({3: (True, 0.9)})
        wrapped = CommitmentPolicyStrategy(
            mock,
            CommitmentPolicy(min_gap=1, min_evidence=0.5)
        )

        # Make a decision to create state
        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        wrapped.get_decision("topic", messages)

        # Reset
        wrapped.reset()

        assert wrapped._state.last_boundary_idx is None
        assert wrapped._state.evidence_buffer == []
        assert wrapped._state.current_idx == 0

    def test_signals_include_commitment_info(self):
        """Decision signals include commitment-specific info."""
        mock = MockDetectionStrategy({3: (True, 0.9)})
        wrapped = CommitmentPolicyStrategy(
            mock,
            CommitmentPolicy(min_gap=2, min_evidence=0.5)
        )

        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(3)]
        decision = wrapped.get_decision("topic", messages)

        assert 'accumulated_evidence' in decision.signals
        assert 'evidence_buffer_size' in decision.signals
        assert 'turns_since_boundary' in decision.signals
        assert 'min_gap' in decision.signals
        assert 'base_detected' in decision.signals
        assert 'committed' in decision.signals


class TestStrategyNaming:
    """Test strategy naming and versioning."""

    def test_name_includes_base_strategy(self):
        """Name indicates wrapped strategy."""
        mock = MockDetectionStrategy()
        mock.name = "TestBase"
        wrapped = CommitmentPolicyStrategy(mock, CommitmentPolicy())

        assert "Committed" in wrapped.name
        assert "TestBase" in wrapped.name

    def test_version_includes_base_version(self):
        """Version includes base strategy version."""
        mock = MockDetectionStrategy()
        mock.version = "2.0.0"
        wrapped = CommitmentPolicyStrategy(mock, CommitmentPolicy())

        assert "2.0.0" in wrapped.version


class TestParamOverrides:
    """Test parameter overrides via constructor."""

    def test_params_override_policy(self):
        """Constructor params override policy values."""
        policy = CommitmentPolicy(min_gap=3, evidence_window=2)
        mock = MockDetectionStrategy()
        wrapped = CommitmentPolicyStrategy(
            mock,
            policy,
            params={'min_gap': 5, 'evidence_window': 4}
        )

        assert wrapped.policy.min_gap == 5
        assert wrapped.policy.evidence_window == 4
