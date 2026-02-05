"""
Unit tests for DefaultStrategy.

Tests the default topic detection strategy which combines
Neural(fine) salience detection with Commitment(medium) filtering.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from episodic.topics.strategy import TopicDecision, Thread, Confidence
from episodic.topics.strategies.default_strategy import (
    DefaultStrategy,
    DEFAULT_COMMITMENT_POLICY,
)


class TestDefaultStrategyInitialization:
    """Test DefaultStrategy initialization and configuration."""

    def test_default_initialization(self):
        """DefaultStrategy initializes with sensible defaults."""
        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy'):
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy'):
                strategy = DefaultStrategy()

        assert strategy.name == "DefaultStrategy"
        assert strategy.version == "1.0.0"

    def test_granularity_config(self):
        """Granularity can be configured."""
        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy') as MockNeural:
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy'):
                strategy = DefaultStrategy({'granularity': 'coarse'})

        # NeuralStrategy should be initialized with coarse granularity
        MockNeural.assert_called_once()
        call_args = MockNeural.call_args[0][0]
        assert call_args['granularity'] == 'coarse'

    def test_commitment_disabled(self):
        """Commitment can be disabled."""
        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy') as MockNeural:
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy') as MockCommit:
                mock_neural = MagicMock()
                MockNeural.return_value = mock_neural

                strategy = DefaultStrategy({'commitment': False})

        # Should use raw neural, not wrapped in commitment
        MockCommit.assert_not_called()
        assert strategy._strategy == mock_neural
        assert strategy._has_commitment is False

    def test_custom_commitment_policy(self):
        """Custom commitment parameters can be provided."""
        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy'):
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy') as MockCommit:
                strategy = DefaultStrategy({
                    'commitment': {'min_gap': 10, 'min_evidence': 2.5}
                })

        # CommitmentPolicyStrategy should be called with custom policy
        MockCommit.assert_called_once()


class TestDefaultPolicyConstants:
    """Test the default commitment policy constants."""

    def test_default_policy_values(self):
        """DEFAULT_COMMITMENT_POLICY has expected values."""
        assert DEFAULT_COMMITMENT_POLICY.min_gap == 4
        assert DEFAULT_COMMITMENT_POLICY.abort_threshold == 0.3
        assert DEFAULT_COMMITMENT_POLICY.abort_streak == 3
        assert DEFAULT_COMMITMENT_POLICY.evidence_decay == 0.7
        assert DEFAULT_COMMITMENT_POLICY.min_evidence == 1.2


class TestDefaultStrategyDelegation:
    """Test that DefaultStrategy properly delegates to underlying strategy."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock underlying strategy."""
        mock = MagicMock()
        mock.get_decision.return_value = TopicDecision(
            topic_changed=False,
            new_thread=None,
            thread_links=[],
            retrieved_context=None,
            confidence=Confidence.LOW,
            confidence_score=0.3,
            reasoning="Mock decision",
            signals={},
            strategy_name="MockStrategy",
            strategy_version="1.0.0",
        )
        mock.segment_conversation.return_value = []
        mock.detect_thread_link.return_value = []
        mock.retrieve_context.return_value = None
        return mock

    def test_get_decision_delegates(self, mock_strategy):
        """get_decision delegates to underlying strategy."""
        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy'):
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy') as MockCommit:
                MockCommit.return_value = mock_strategy
                strategy = DefaultStrategy()

        messages = [{'role': 'user', 'content': 'hello'}]
        decision = strategy.get_decision("test query", messages)

        mock_strategy.get_decision.assert_called_once()
        assert decision.strategy_name == "DefaultStrategy"

    def test_segment_conversation_delegates(self, mock_strategy):
        """segment_conversation delegates to underlying strategy."""
        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy'):
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy') as MockCommit:
                MockCommit.return_value = mock_strategy
                strategy = DefaultStrategy()

        messages = [{'role': 'user', 'content': 'hello'}]
        strategy.segment_conversation(messages)

        mock_strategy.segment_conversation.assert_called_once_with(messages)

    def test_reset_delegates(self, mock_strategy):
        """reset delegates to underlying strategy."""
        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy'):
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy') as MockCommit:
                MockCommit.return_value = mock_strategy
                strategy = DefaultStrategy()

        strategy.reset()

        mock_strategy.reset.assert_called_once()


class TestDefaultStrategyDecisionOverrides:
    """Test that DefaultStrategy overrides decision metadata."""

    def test_strategy_name_override(self):
        """Decision strategy_name is overridden to DefaultStrategy."""
        mock_decision = TopicDecision(
            topic_changed=True,
            new_thread=None,
            thread_links=[],
            retrieved_context=None,
            confidence=Confidence.HIGH,
            confidence_score=0.9,
            reasoning="Neural detected",
            signals={'boundary_probability': 0.9},
            strategy_name="NeuralStrategy",
            strategy_version="1.0.0",
        )

        mock_underlying = MagicMock()
        mock_underlying.get_decision.return_value = mock_decision

        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy'):
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy') as MockCommit:
                MockCommit.return_value = mock_underlying
                strategy = DefaultStrategy()

        messages = [{'role': 'user', 'content': 'hello'}]
        decision = strategy.get_decision("test", messages)

        assert decision.strategy_name == "DefaultStrategy"
        assert decision.strategy_version == "1.0.0"

    def test_diagnostics_added_to_signals(self):
        """Decision signals include diagnostics."""
        mock_decision = TopicDecision(
            topic_changed=False,
            new_thread=None,
            thread_links=[],
            retrieved_context=None,
            confidence=Confidence.LOW,
            confidence_score=0.2,
            reasoning="No change",
            signals={},
            strategy_name="MockStrategy",
            strategy_version="1.0.0",
        )

        mock_underlying = MagicMock()
        mock_underlying.get_decision.return_value = mock_decision

        with patch('episodic.topics.strategies.default_strategy.NeuralStrategy'):
            with patch('episodic.topics.strategies.default_strategy.CommitmentPolicyStrategy') as MockCommit:
                MockCommit.return_value = mock_underlying
                strategy = DefaultStrategy()

        messages = [{'role': 'user', 'content': 'hello'}]
        decision = strategy.get_decision("test", messages)

        assert 'diagnostics' in decision.signals
