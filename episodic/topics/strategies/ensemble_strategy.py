"""
Ensemble strategy combining multiple topic detection approaches.

Uses keyword detection for high-confidence explicit transitions,
neural detection as primary signal, and embedding similarity as backup.
"""

import time
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

logger = logging.getLogger(__name__)


class EnsembleStrategy(TopicStrategy):
    """
    Ensemble topic detection combining multiple strategies.

    Priority order:
    1. Keyword detection - explicit transitions ("by the way", "changing topics")
       are high-confidence and override other signals
    2. Neural detection - primary signal from fine-tuned model
    3. Embedding similarity - backup signal for edge cases

    Can be configured to use different combination rules.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble strategy.

        Args:
            params: Optional parameters:
                - use_keyword: Enable keyword detection (default: True)
                - use_neural: Enable neural detection (default: True)
                - use_embedding: Enable embedding detection (default: False)
                - neural_threshold: Confidence threshold for neural (default: 0.5)
                - keyword_explicit_threshold: Threshold for explicit transitions (default: 0.5)
                - require_agreement: Require multiple signals to agree (default: False)
                - agreement_count: Number of signals that must agree (default: 2)
        """
        params = params or {}
        self.use_keyword = params.get('use_keyword', True)
        self.use_neural = params.get('use_neural', True)
        self.use_embedding = params.get('use_embedding', False)
        self.neural_threshold = params.get('neural_threshold', 0.5)
        self.keyword_explicit_threshold = params.get('keyword_explicit_threshold', 0.5)
        self.require_agreement = params.get('require_agreement', False)
        self.agreement_count = params.get('agreement_count', 2)

        # Lazy-loaded strategies
        self._keyword_strategy = None
        self._neural_strategy = None
        self._embedding_strategy = None

    @property
    def name(self) -> str:
        return "EnsembleStrategy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def _get_keyword_strategy(self):
        """Lazy load keyword strategy."""
        if self._keyword_strategy is None:
            from episodic.topics.strategies.keyword_strategy import KeywordStrategy
            self._keyword_strategy = KeywordStrategy()
        return self._keyword_strategy

    def _get_neural_strategy(self):
        """Lazy load neural strategy."""
        if self._neural_strategy is None:
            from episodic.topics.strategies.neural_strategy import NeuralStrategy
            self._neural_strategy = NeuralStrategy({
                'confidence_threshold': self.neural_threshold
            })
        return self._neural_strategy

    def _get_embedding_strategy(self):
        """Lazy load embedding strategy."""
        if self._embedding_strategy is None:
            from episodic.topics.strategies.relative_embedding_strategy import (
                RelativeEmbeddingStrategy
            )
            self._embedding_strategy = RelativeEmbeddingStrategy()
        return self._embedding_strategy

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Segment conversation using neural strategy."""
        if self.use_neural:
            return self._get_neural_strategy().segment_conversation(messages)
        return [Thread(
            id="thread_0",
            name="conversation",
            start_node_id=messages[0].get('node_id', '0') if messages else '0',
            messages=messages
        )]

    def detect_thread_link(
        self,
        query: str,
        threads: List[Thread],
        current_thread: Optional[Thread] = None
    ) -> List[ThreadLink]:
        """Detect thread links using embedding strategy."""
        if self.use_embedding:
            return self._get_embedding_strategy().detect_thread_link(
                query, threads, current_thread
            )
        return []

    def retrieve_context(
        self,
        query: str,
        threads: List[Thread],
        current_thread: Optional[Thread] = None,
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Retrieve context using embedding strategy."""
        if self.use_embedding:
            return self._get_embedding_strategy().retrieve_context(
                query, threads, current_thread, max_tokens
            )
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="ensemble_no_retrieval"
        )

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make topic change decision using ensemble of strategies.

        Logic:
        1. If keyword detects explicit transition → return True (high confidence)
        2. If require_agreement → need N signals to agree
        3. Otherwise → use neural as primary signal
        """
        start_time = time.time()

        decisions = {}
        signals = {}

        # 1. Check keyword detection first (explicit transitions are definitive)
        if self.use_keyword:
            keyword_decision = self._get_keyword_strategy().get_decision(
                query, messages, current_thread
            )
            decisions['keyword'] = keyword_decision

            # Explicit transition phrases are high-confidence
            explicit_score = keyword_decision.signals.get('explicit_transition', 0)
            signals['keyword_explicit'] = explicit_score
            signals['keyword_domain'] = keyword_decision.signals.get('domain_shift', 0)

            if explicit_score >= self.keyword_explicit_threshold:
                # Explicit transition detected - high confidence boundary
                processing_time = (time.time() - start_time) * 1000
                return TopicDecision(
                    topic_changed=True,
                    new_thread=None,
                    thread_links=[],
                    retrieved_context=None,
                    confidence=Confidence.HIGH,
                    confidence_score=explicit_score,
                    reasoning=f"Explicit transition detected: {keyword_decision.reasoning}",
                    signals={
                        **signals,
                        'decision_source': 'keyword_explicit',
                    },
                    strategy_name=self.name,
                    strategy_version=self.version,
                    processing_time_ms=processing_time,
                )

        # 2. Get neural detection
        if self.use_neural:
            try:
                neural_decision = self._get_neural_strategy().get_decision(
                    query, messages, current_thread
                )
                decisions['neural'] = neural_decision
                signals['neural_boundary_prob'] = neural_decision.confidence_score
                signals['neural_changed'] = 1.0 if neural_decision.topic_changed else 0.0
            except Exception as e:
                logger.warning(f"Neural detection failed: {e}")
                signals['neural_error'] = str(e)

        # 3. Get embedding detection (optional)
        if self.use_embedding:
            try:
                embedding_decision = self._get_embedding_strategy().get_decision(
                    query, messages, current_thread
                )
                decisions['embedding'] = embedding_decision
                signals['embedding_z_score'] = embedding_decision.signals.get('z_score', 0)
                signals['embedding_changed'] = 1.0 if embedding_decision.topic_changed else 0.0
            except Exception as e:
                logger.warning(f"Embedding detection failed: {e}")
                signals['embedding_error'] = str(e)

        # 4. Combine decisions
        if self.require_agreement:
            # Count how many strategies agree on topic change
            votes = sum(
                1 for d in decisions.values()
                if d.topic_changed
            )
            topic_changed = votes >= self.agreement_count
            reasoning = f"Agreement voting: {votes}/{len(decisions)} strategies detected change"
            confidence_score = votes / len(decisions) if decisions else 0
        else:
            # Use neural as primary, fall back to others
            if 'neural' in decisions:
                topic_changed = decisions['neural'].topic_changed
                reasoning = f"Neural: {decisions['neural'].reasoning}"
                confidence_score = decisions['neural'].confidence_score
            elif 'embedding' in decisions:
                topic_changed = decisions['embedding'].topic_changed
                reasoning = f"Embedding fallback: {decisions['embedding'].reasoning}"
                confidence_score = decisions['embedding'].confidence_score
            elif 'keyword' in decisions:
                topic_changed = decisions['keyword'].topic_changed
                reasoning = f"Keyword fallback: {decisions['keyword'].reasoning}"
                confidence_score = decisions['keyword'].confidence_score
            else:
                topic_changed = False
                reasoning = "No strategies available"
                confidence_score = 0.0

        # Determine confidence level
        if confidence_score >= 0.8:
            confidence = Confidence.HIGH
        elif confidence_score >= 0.5:
            confidence = Confidence.MEDIUM
        elif confidence_score >= 0.3:
            confidence = Confidence.LOW
        else:
            confidence = Confidence.UNCERTAIN

        processing_time = (time.time() - start_time) * 1000
        signals['decision_source'] = 'ensemble'

        return TopicDecision(
            topic_changed=topic_changed,
            new_thread=None,
            thread_links=[],
            retrieved_context=None,
            confidence=confidence,
            confidence_score=confidence_score,
            reasoning=reasoning,
            signals=signals,
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=processing_time,
        )
