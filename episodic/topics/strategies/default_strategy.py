"""
Default topic detection strategy for Episodic.

Combines Neural(fine) salience detection with Commitment(medium) filtering
to provide stable, high-quality topic boundary detection out of the box.

This is the recommended strategy for general use.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from episodic.topics.strategy import (
    TopicStrategy,
    Thread,
    ThreadLink,
    RetrievedContext,
    TopicDecision,
)
from episodic.topics.strategies.neural_strategy import NeuralStrategy
from episodic.topics.strategies.commitment_strategy import (
    CommitmentPolicyStrategy,
    CommitmentPolicy,
)
from episodic.topics.diagnostics import (
    DiagnosticsCollector,
    record_decision_diagnostics,
)

logger = logging.getLogger(__name__)


# Default commitment policy - INTERNAL parameters only
#
# User-facing parameters come from config_template.json (single source of truth):
# - suspect_threshold: Neural confidence to enter SUSPECT
# - drift_suspect_threshold: Drift threshold to enter SUSPECT
# - min_messages_before_topic_change: Commit gate
#
# These INTERNAL parameters are implementation details users won't typically change:
DEFAULT_COMMITMENT_POLICY = CommitmentPolicy(
    min_gap=4,              # Minimum messages between boundaries
    abort_threshold=0.3,    # ABORT if confidence stays below this
    abort_streak=3,         # ABORT after this many low-confidence turns
    evidence_decay=0.7,     # Decay factor for accumulated evidence
    min_evidence=1.2,       # Evidence needed to COMMIT
)


class DefaultStrategy(TopicStrategy):
    """
    Default topic detection: Neural(fine) + Commitment(medium).

    This strategy combines:
    - Neural salience detection with fine granularity (high sensitivity)
    - Commitment filtering with medium strictness (prevents oversegmentation)

    The result is stable boundary detection that:
    - Detects meaningful topic shifts
    - Avoids boundary spam
    - Works across diverse conversation types

    Configuration options:
        granularity: 'fine' (default), 'medium', or 'coarse'
        commitment: dict with CommitmentPolicy parameters, or False to disable

    Example:
        # Default (recommended)
        strategy = DefaultStrategy()

        # Custom granularity
        strategy = DefaultStrategy({'granularity': 'coarse'})

        # Custom commitment
        strategy = DefaultStrategy({
            'commitment': {'min_gap': 3, 'min_evidence': 0.8}
        })

        # Disable commitment (raw neural, not recommended)
        strategy = DefaultStrategy({'commitment': False})
    """

    def __init__(self, strategy_config: Dict[str, Any] = None):
        """
        Initialize default strategy.

        Args:
            strategy_config: Optional configuration:
                - granularity: 'fine', 'medium', 'coarse' (default: 'fine')
                - commitment: dict of CommitmentPolicy params, or False to disable
        """
        super().__init__(strategy_config)
        strategy_config = strategy_config or {}

        self.name = "DefaultStrategy"
        self.version = "1.0.0"

        # Build neural base - respect user's config granularity
        from episodic.config import config
        granularity = strategy_config.get('granularity') or config.get('topic_granularity', 'medium')
        self._neural = NeuralStrategy({'granularity': granularity})

        # Build commitment wrapper (unless disabled)
        commitment_config = strategy_config.get('commitment', {})

        if commitment_config is False:
            # Disabled - use raw neural
            self._strategy = self._neural
            self._has_commitment = False
            logger.info(f"DefaultStrategy: Neural({granularity}) without commitment")
        else:
            # Get drift trigger settings from user config
            # Defaults come from config_template.json (single source of truth)
            use_drift_trigger = config.get('use_drift_trigger')
            drift_suspect_threshold = config.get('drift_suspect_threshold')

            # If drift trigger is disabled, set threshold to None
            if not use_drift_trigger:
                drift_suspect_threshold = None

            # Get neural SUSPECT threshold from config
            suspect_threshold = config.get('suspect_threshold')

            # Get min_user_turns_for_commit from config (commit gate, not detection gate)
            min_user_turns = config.get('min_messages_before_topic_change')

            # Get neural commit drift gate threshold from config
            neural_commit_drift_threshold = config.get('neural_commit_drift_threshold')

            # Build commitment policy from config or defaults
            if isinstance(commitment_config, dict) and commitment_config:
                policy = CommitmentPolicy(
                    min_gap=commitment_config.get('min_gap', DEFAULT_COMMITMENT_POLICY.min_gap),
                    suspect_threshold=commitment_config.get('suspect_threshold', suspect_threshold),
                    abort_threshold=commitment_config.get('abort_threshold', DEFAULT_COMMITMENT_POLICY.abort_threshold),
                    abort_streak=commitment_config.get('abort_streak', DEFAULT_COMMITMENT_POLICY.abort_streak),
                    evidence_decay=commitment_config.get('evidence_decay', DEFAULT_COMMITMENT_POLICY.evidence_decay),
                    min_evidence=commitment_config.get('min_evidence', DEFAULT_COMMITMENT_POLICY.min_evidence),
                    drift_suspect_threshold=commitment_config.get('drift_suspect_threshold', drift_suspect_threshold),
                    min_user_turns_for_commit=commitment_config.get('min_user_turns_for_commit', min_user_turns),
                    neural_commit_drift_threshold=commitment_config.get('neural_commit_drift_threshold', neural_commit_drift_threshold),
                )
            else:
                # Use default policy but override from config
                policy = CommitmentPolicy(
                    min_gap=DEFAULT_COMMITMENT_POLICY.min_gap,
                    suspect_threshold=suspect_threshold,
                    abort_threshold=DEFAULT_COMMITMENT_POLICY.abort_threshold,
                    abort_streak=DEFAULT_COMMITMENT_POLICY.abort_streak,
                    evidence_decay=DEFAULT_COMMITMENT_POLICY.evidence_decay,
                    min_evidence=DEFAULT_COMMITMENT_POLICY.min_evidence,
                    drift_suspect_threshold=drift_suspect_threshold,
                    min_user_turns_for_commit=min_user_turns,
                    neural_commit_drift_threshold=neural_commit_drift_threshold,
                )

            self._strategy = CommitmentPolicyStrategy(self._neural, policy)
            self._has_commitment = True
            drift_info = f", drift_threshold={policy.drift_suspect_threshold}" if policy.drift_suspect_threshold else ", drift=disabled"
            logger.info(f"DefaultStrategy: Neural({granularity}) + Commitment(min_evidence={policy.min_evidence}, min_turns={policy.min_user_turns_for_commit}{drift_info})")

        # Diagnostics collector for observability
        self._diagnostics = DiagnosticsCollector()
        self._last_message_time: Optional[datetime] = None

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None,
        semantic_drift: Optional[float] = None,
        **kwargs
    ) -> TopicDecision:
        """Get topic decision using neural + commitment pipeline."""
        decision = self._strategy.get_decision(
            query, messages, current_thread, semantic_drift=semantic_drift, **kwargs
        )

        # Compute time gap for diagnostics
        now = datetime.now()
        time_gap_seconds = 0.0
        if self._last_message_time is not None:
            time_gap_seconds = (now - self._last_message_time).total_seconds()
        self._last_message_time = now

        # Collect diagnostics
        snapshot = record_decision_diagnostics(
            decision.signals,
            time_gap_seconds=time_gap_seconds
        )

        # Add diagnostics to decision signals
        decision.signals['diagnostics'] = snapshot.to_dict()

        # Override strategy name to show it's the default
        decision.strategy_name = self.name
        decision.strategy_version = self.version

        return decision

    def reset(self) -> None:
        """Reset strategy state."""
        if hasattr(self._strategy, 'reset'):
            self._strategy.reset()
        self._diagnostics.reset()
        self._last_message_time = None

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Segment conversation using the underlying strategy."""
        return self._strategy.segment_conversation(messages)

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Detect thread links using the underlying strategy."""
        return self._strategy.detect_thread_link(query, recent_context, past_threads)

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Retrieve context using the underlying strategy."""
        return self._strategy.retrieve_context(query, thread_links, threads, max_tokens)
