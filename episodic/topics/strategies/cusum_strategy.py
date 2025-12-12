"""
CUSUM (Cumulative Sum) topic detection strategy.

Uses classic change-point detection to accumulate drift evidence over time,
triggering boundaries when accumulated drift exceeds a threshold.

Advantages over single-step threshold:
- Detects gradual topic shifts
- More robust to noise (single low-similarity message doesn't trigger)
- Self-resetting after boundary detection
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from episodic.topics.strategy import (
    TopicStrategy,
    Thread,
    ThreadLink,
    RetrievedContext,
    TopicDecision,
    Confidence
)
from episodic.ml.drift import ConversationalDrift
from episodic.config import config


class CUSUMStrategy(TopicStrategy):
    """
    CUSUM-based topic boundary detection.

    Algorithm:
    1. Compute drift d_t = 1 - similarity(query, recent_context)
    2. Update cumulative sum: S_t = max(0, S_{t-1} + d_t - μ)
       where μ is the expected (baseline) drift
    3. Trigger boundary when S_t > τ (threshold)
    4. Reset S_t = 0 after boundary

    This accumulates evidence of drift, catching gradual shifts
    that single-step thresholds miss.
    """

    def __init__(self, strategy_config: Dict[str, Any] = None):
        """
        Initialize CUSUM strategy.

        Args:
            strategy_config: Optional parameters:
                - threshold: CUSUM threshold τ (default: 0.3)
                - baseline_drift: Expected drift μ (default: 0.02)
                - adaptive_baseline: Learn μ from conversation (default: True)
                - min_history: Minimum messages before detecting (default: 3)
        """
        super().__init__(strategy_config)
        strategy_config = strategy_config or {}

        self.name = "CUSUMStrategy"
        self.version = "1.0.0"

        # CUSUM parameters
        self.threshold = strategy_config.get('threshold', 0.3)
        self.baseline_drift = strategy_config.get('baseline_drift', 0.02)
        self.adaptive_baseline = strategy_config.get('adaptive_baseline', True)
        self.min_history = strategy_config.get('min_history', 3)

        # State
        self._cusum = 0.0
        self._drift_history: List[float] = []
        self._current_baseline = self.baseline_drift

        # Drift calculator for embeddings
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")
        self._drift_calc = ConversationalDrift(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )

    def _compute_drift(
        self,
        query: str,
        messages: List[Dict[str, Any]]
    ) -> float:
        """Compute drift between query and recent context."""
        if not messages:
            return 0.0

        # Get recent context (last few messages)
        recent = messages[-4:] if len(messages) >= 4 else messages
        context_text = " ".join(m.get('content', '') for m in recent)

        # Compute embeddings
        query_emb = self._drift_calc.embedding_provider.embed(query)
        context_emb = self._drift_calc.embedding_provider.embed(context_text)

        # Cosine similarity
        dot = sum(a * b for a, b in zip(query_emb, context_emb))
        norm_q = sum(a * a for a in query_emb) ** 0.5
        norm_c = sum(b * b for b in context_emb) ** 0.5

        if norm_q == 0 or norm_c == 0:
            return 0.0

        similarity = dot / (norm_q * norm_c)

        # Drift = 1 - similarity (higher = more different)
        return 1.0 - similarity

    def _update_baseline(self, drift: float) -> None:
        """Update adaptive baseline from drift history."""
        self._drift_history.append(drift)

        # Keep last 20 drifts
        if len(self._drift_history) > 20:
            self._drift_history = self._drift_history[-20:]

        if self.adaptive_baseline and len(self._drift_history) >= 5:
            # Use median as robust baseline
            sorted_drifts = sorted(self._drift_history)
            mid = len(sorted_drifts) // 2
            self._current_baseline = sorted_drifts[mid]

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make topic decision using CUSUM.
        """
        import time
        start_time = time.time()

        # Not enough history
        if len(messages) < self.min_history:
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.LOW,
                confidence_score=0.2,
                strategy_name=self.name,
                strategy_version=self.version,
                reasoning="Not enough conversation history for CUSUM",
                signals={'cusum': self._cusum, 'drift': 0.0},
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={}
            )

        # Compute drift
        drift = self._compute_drift(query, messages)

        # Update CUSUM: S_t = max(0, S_{t-1} + d_t - μ)
        self._cusum = max(0.0, self._cusum + drift - self._current_baseline)

        # Check for boundary
        topic_changed = self._cusum > self.threshold

        # Build signals
        signals = {
            'drift': drift,
            'cusum': self._cusum,
            'threshold': self.threshold,
            'baseline': self._current_baseline,
        }

        # Determine confidence
        if topic_changed:
            # How far above threshold?
            excess = self._cusum - self.threshold
            if excess > 0.2:
                confidence = Confidence.HIGH
                confidence_score = min(1.0, 0.7 + excess)
            elif excess > 0.1:
                confidence = Confidence.MEDIUM
                confidence_score = 0.6
            else:
                confidence = Confidence.LOW
                confidence_score = 0.4

            reasoning = (
                f"CUSUM triggered: accumulated drift {self._cusum:.3f} > "
                f"threshold {self.threshold:.3f} (current drift={drift:.3f})"
            )

            # Reset CUSUM after boundary
            self._cusum = 0.0

            # Create new thread
            new_thread = Thread(
                id=str(uuid.uuid4()),
                name=None,  # Will be named by topic extraction
                start_node_id="",
                end_node_id=None,
                message_count=1,
                created_at=datetime.now(),
                metadata={'cusum_excess': excess, 'drift': drift}
            )
        else:
            confidence = Confidence.LOW
            confidence_score = 0.3
            reasoning = (
                f"CUSUM below threshold: {self._cusum:.3f} <= {self.threshold:.3f} "
                f"(current drift={drift:.3f})"
            )
            new_thread = None

        # Update baseline after decision
        self._update_baseline(drift)

        processing_time = (time.time() - start_time) * 1000

        return TopicDecision(
            topic_changed=topic_changed,
            new_thread=new_thread,
            thread_links=[],
            retrieved_context=None,
            confidence=confidence,
            confidence_score=confidence_score,
            strategy_name=self.name,
            strategy_version=self.version,
            reasoning=reasoning,
            signals=signals,
            processing_time_ms=processing_time,
            metadata={'drift_history_len': len(self._drift_history)}
        )

    def reset(self) -> None:
        """Reset CUSUM state for new conversation."""
        self._cusum = 0.0
        self._drift_history = []
        self._current_baseline = self.baseline_drift

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Not implemented - CUSUM is incremental, not batch."""
        return []

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Not implemented for CUSUM strategy."""
        return []

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Not implemented for CUSUM strategy."""
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="CUSUM strategy does not support context retrieval"
        )
