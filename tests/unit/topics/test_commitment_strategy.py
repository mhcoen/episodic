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


# =============================================================================
# Adaptive Commitment Strategy Tests
# =============================================================================

from episodic.topics.strategies.commitment_strategy import (
    AdaptiveCommitmentStrategy,
    AdaptivePolicy,
)


class TestAdaptivePolicyBasics:
    """Test AdaptivePolicy configuration."""

    def test_default_values(self):
        """AdaptivePolicy has sensible defaults."""
        policy = AdaptivePolicy()
        assert policy.target_rate == 0.1
        assert policy.rate_window == 50
        assert policy.adaptation_rate == 0.15  # Lower for stability
        assert policy.tolerance == 0.25
        assert policy.fixed_min_gap == 2  # Single-knob: gap is fixed
        assert policy.warmup_messages == 10
        assert policy.warmup_calibrate is True

    def test_custom_values(self):
        """AdaptivePolicy accepts custom values."""
        policy = AdaptivePolicy(
            target_rate=0.15,
            rate_window=30,
            adaptation_rate=0.2,
            tolerance=0.3,
            fixed_min_gap=3,
            warmup_messages=15,
        )
        assert policy.target_rate == 0.15
        assert policy.rate_window == 30
        assert policy.fixed_min_gap == 3
        assert policy.warmup_messages == 15


class TestAdaptiveRateTracking:
    """Test rate calculation and tracking."""

    def test_rate_starts_at_target(self):
        """Initially returns target rate (not enough data)."""
        mock = MockDetectionStrategy()
        adaptive = AdaptiveCommitmentStrategy(
            mock,
            AdaptivePolicy(target_rate=0.1)
        )

        # Not enough messages yet
        assert adaptive._current_rate() == 0.1

    def test_rate_updates_with_messages(self):
        """Rate calculation reflects actual boundaries."""
        mock = MockDetectionStrategy({
            i: (True, 0.9) for i in range(3, 20, 4)  # Boundaries every 4 messages
        })
        policy = AdaptivePolicy(target_rate=0.1, rate_window=20)
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        # Process 15 messages
        messages = []
        for i in range(15):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            adaptive.get_decision(f"query{i}", messages)

        # Should have some boundaries tracked
        rate = adaptive._current_rate()
        assert rate > 0  # Should have detected something


class TestAdaptivePolicyAdjustment:
    """Test policy self-adjustment behavior (single-knob: only min_evidence)."""

    def test_single_knob_only_evidence_changes(self):
        """Only min_evidence is adapted, min_gap stays fixed."""
        mock = MockDetectionStrategy({i: (True, 0.9) for i in range(2, 40)})
        policy = AdaptivePolicy(
            target_rate=0.05,
            rate_window=15,
            adaptation_rate=0.3,
            tolerance=0.1,
            fixed_min_gap=2,
            warmup_messages=5,
        )
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        initial_min_gap = adaptive.policy.min_gap

        # Process many messages
        messages = []
        for i in range(35):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            adaptive.get_decision(f"query{i}", messages)

        # min_gap should remain fixed
        assert adaptive.policy.min_gap == initial_min_gap == 2
        # min_evidence should have increased (tightened)
        assert adaptive.policy.min_evidence > 0.7

    def test_tightens_on_oversegmentation(self):
        """min_evidence increases when rate exceeds target."""
        mock = MockDetectionStrategy({i: (True, 0.9) for i in range(2, 40)})
        policy = AdaptivePolicy(
            target_rate=0.05,
            rate_window=15,
            adaptation_rate=0.3,
            tolerance=0.1,
            warmup_messages=5,
        )
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        initial_min_evidence = adaptive.policy.min_evidence

        # Process many messages to trigger adaptation
        messages = []
        for i in range(35):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            adaptive.get_decision(f"query{i}", messages)

        # Policy should have tightened (min_evidence increased)
        assert adaptive.policy.min_evidence >= initial_min_evidence

    def test_no_adjustment_within_tolerance(self):
        """No adjustment when rate is within tolerance band."""
        mock = MockDetectionStrategy()  # No detections
        policy = AdaptivePolicy(
            target_rate=0.1,
            tolerance=0.5,
            warmup_messages=5,
        )
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        # Process messages past warmup
        messages = []
        for i in range(20):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            adaptive.get_decision(f"query{i}", messages)

        # Should have minimal adjustments
        assert adaptive._adjustment_count < 3


class TestAdaptiveReset:
    """Test state management for adaptive strategy."""

    def test_reset_clears_tracking(self):
        """reset() clears rate tracking state."""
        mock = MockDetectionStrategy({3: (True, 0.9)})
        adaptive = AdaptiveCommitmentStrategy(mock, AdaptivePolicy())

        # Make some decisions
        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(15)]
        for i in range(15):
            adaptive.get_decision(f"q{i}", messages[:i+1])

        # Reset
        adaptive.reset()

        assert adaptive._message_count == 0
        assert adaptive._boundary_count == 0
        assert len(adaptive._recent_boundaries) == 0
        assert adaptive._warmup_complete is False
        assert len(adaptive._warmup_saliences) == 0
        assert adaptive._adjustment_count == 0


class TestAdaptiveWarmup:
    """Test cold-start warmup behavior."""

    def test_warmup_collects_saliences(self):
        """During warmup, saliences are collected."""
        mock = MockDetectionStrategy({i: (True, 0.5 + i * 0.02) for i in range(3, 15)})
        policy = AdaptivePolicy(warmup_messages=10, warmup_calibrate=True)
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        messages = []
        for i in range(8):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            adaptive.get_decision(f"q{i}", messages)

        # Should still be in warmup
        assert not adaptive._warmup_complete
        assert len(adaptive._warmup_saliences) == 8

    def test_warmup_calibrates_threshold(self):
        """After warmup, min_evidence is calibrated from salience distribution."""
        # Create mock with known salience distribution
        mock = MockDetectionStrategy({i: (True, 0.8) for i in range(3, 20)})
        policy = AdaptivePolicy(
            warmup_messages=10,
            warmup_calibrate=True,
            target_rate=0.1,
        )
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        messages = []
        for i in range(12):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            adaptive.get_decision(f"q{i}", messages)

        # Warmup should be complete
        assert adaptive._warmup_complete


class TestAdaptiveVolatility:
    """Test volatility tracking metrics."""

    def test_adjustment_count_tracked(self):
        """Number of policy adjustments is tracked."""
        mock = MockDetectionStrategy({i: (True, 0.9) for i in range(2, 50)})
        policy = AdaptivePolicy(
            target_rate=0.05,
            rate_window=15,
            adaptation_rate=0.3,
            tolerance=0.1,
            warmup_messages=5,
        )
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        messages = []
        for i in range(40):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            adaptive.get_decision(f"q{i}", messages)

        # Should have made some adjustments
        assert adaptive._adjustment_count > 0

    def test_rate_volatility_computed(self):
        """Rate volatility (std dev) is computed."""
        mock = MockDetectionStrategy({i: (True, 0.9) for i in range(2, 50)})
        policy = AdaptivePolicy(warmup_messages=5)
        adaptive = AdaptiveCommitmentStrategy(mock, policy)

        messages = []
        for i in range(30):
            messages.append({'role': 'user', 'content': f'msg{i}'})
            decision = adaptive.get_decision(f"q{i}", messages)

        # Volatility should be in signals
        assert 'rate_volatility' in decision.signals


class TestAdaptiveSignals:
    """Test that adaptive strategy includes useful signals."""

    def test_signals_include_rate_info(self):
        """Decision signals include rate, adaptation, and volatility info."""
        mock = MockDetectionStrategy({3: (True, 0.9)})
        adaptive = AdaptiveCommitmentStrategy(
            mock,
            AdaptivePolicy(target_rate=0.1, warmup_messages=2)
        )

        messages = [{'role': 'user', 'content': f'msg{i}'} for i in range(4)]
        decision = adaptive.get_decision("query", messages)

        assert 'current_rate' in decision.signals
        assert 'target_rate' in decision.signals
        assert 'min_gap' in decision.signals  # Fixed, not current_min_gap
        assert 'current_min_evidence' in decision.signals
        assert 'rate_volatility' in decision.signals
        assert 'adjustment_count' in decision.signals
        assert 'warmup_complete' in decision.signals
