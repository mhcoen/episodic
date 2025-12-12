"""
Delta-representation topic detection strategy.

Models topic boundaries as *changes* in discourse representation,
not as absolute semantic content.

Given embeddings h_t, computes:
    Δh_t = h_t - h_{t-k}

Then classifies based on the magnitude and characteristics of this delta.

Advantages:
- Shift-invariant (works regardless of absolute topic)
- Directly models what we care about (transitions)
- Can distinguish gradual drift from abrupt shifts
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


class DeltaStrategy(TopicStrategy):
    """
    Delta-representation based topic boundary detection.

    Instead of comparing query to context (state-based),
    compares the *change* in representation from one window to the next.

    Algorithm:
    1. Compute embedding for recent window (messages t-k to t-1)
    2. Compute embedding for window including query (messages t-k+1 to t)
    3. Compute delta: Δh = h_new - h_old
    4. Boundary if ||Δh|| > threshold or direction shifts significantly
    """

    def __init__(self, strategy_config: Dict[str, Any] = None):
        """
        Initialize delta strategy.

        Args:
            strategy_config: Optional parameters:
                - window_size: Size of comparison windows (default: 4)
                - magnitude_threshold: Delta magnitude threshold (default: 0.3)
                - use_direction: Also consider direction change (default: True)
                - direction_threshold: Cosine distance for direction (default: 0.5)
                - adaptive: Learn thresholds from conversation (default: True)
        """
        super().__init__(strategy_config)
        strategy_config = strategy_config or {}

        self.name = "DeltaStrategy"
        self.version = "1.0.0"

        # Parameters
        self.window_size = strategy_config.get('window_size', 4)
        self.magnitude_threshold = strategy_config.get('magnitude_threshold', 0.3)
        self.use_direction = strategy_config.get('use_direction', True)
        self.direction_threshold = strategy_config.get('direction_threshold', 0.5)
        self.adaptive = strategy_config.get('adaptive', True)

        # State for adaptive thresholds
        self._magnitude_history: List[float] = []
        self._direction_history: List[float] = []
        self._prev_delta: Optional[List[float]] = None

        # Drift calculator for embeddings
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")
        self._drift_calc = ConversationalDrift(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )

    def _get_window_embedding(self, messages: List[Dict[str, Any]]) -> List[float]:
        """Compute centroid embedding for a window of messages."""
        if not messages:
            return []

        # Concatenate messages
        text = " ".join(m.get('content', '') for m in messages)
        return self._drift_calc.embedding_provider.embed(text)

    def _compute_magnitude(self, delta: List[float]) -> float:
        """Compute L2 norm of delta vector."""
        return sum(d * d for d in delta) ** 0.5

    def _compute_direction_change(
        self,
        delta: List[float],
        prev_delta: List[float]
    ) -> float:
        """
        Compute how much the direction of change has shifted.
        Returns 1 - cosine_similarity (0 = same direction, 2 = opposite).
        """
        if not prev_delta:
            return 0.0

        dot = sum(a * b for a, b in zip(delta, prev_delta))
        norm_d = sum(a * a for a in delta) ** 0.5
        norm_p = sum(b * b for b in prev_delta) ** 0.5

        if norm_d == 0 or norm_p == 0:
            return 0.0

        cosine = dot / (norm_d * norm_p)
        return 1.0 - cosine  # 0 = same direction, 2 = opposite

    def _get_adaptive_threshold(self) -> float:
        """Compute adaptive magnitude threshold."""
        if not self._magnitude_history or len(self._magnitude_history) < 5:
            return self.magnitude_threshold

        # Use mean + 1.5 * std as threshold
        import numpy as np
        mean = np.mean(self._magnitude_history)
        std = np.std(self._magnitude_history)
        return float(mean + 1.5 * std)

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make topic decision using delta representation.
        """
        import time
        start_time = time.time()

        # Need enough messages for two windows
        min_messages = self.window_size + 1
        if len(messages) < min_messages:
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.LOW,
                confidence_score=0.2,
                strategy_name=self.name,
                strategy_version=self.version,
                reasoning=f"Need at least {min_messages} messages for delta comparison",
                signals={},
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={}
            )

        # Old window: messages[-window_size-1:-1]
        old_window = messages[-(self.window_size + 1):-1]
        old_emb = self._get_window_embedding(old_window)

        # New window: messages[-window_size:] + query
        new_window_msgs = messages[-self.window_size:]
        new_window_text = " ".join(m.get('content', '') for m in new_window_msgs) + " " + query
        new_emb = self._drift_calc.embedding_provider.embed(new_window_text)

        # Compute delta
        delta = [n - o for n, o in zip(new_emb, old_emb)]
        magnitude = self._compute_magnitude(delta)

        # Compute direction change from previous delta
        direction_change = 0.0
        if self.use_direction and self._prev_delta:
            direction_change = self._compute_direction_change(delta, self._prev_delta)

        # Get threshold
        threshold = self._get_adaptive_threshold() if self.adaptive else self.magnitude_threshold

        # Decision logic
        magnitude_triggered = magnitude > threshold
        direction_triggered = self.use_direction and direction_change > self.direction_threshold

        topic_changed = magnitude_triggered or direction_triggered

        # Build signals
        signals = {
            'delta_magnitude': magnitude,
            'threshold': threshold,
            'direction_change': direction_change,
            'magnitude_triggered': 1.0 if magnitude_triggered else 0.0,
            'direction_triggered': 1.0 if direction_triggered else 0.0,
        }

        # Determine confidence and reasoning
        if topic_changed:
            triggers = []
            if magnitude_triggered:
                triggers.append(f"magnitude {magnitude:.3f} > {threshold:.3f}")
            if direction_triggered:
                triggers.append(f"direction change {direction_change:.3f} > {self.direction_threshold}")

            reasoning = f"Delta triggered: {', '.join(triggers)}"

            if magnitude_triggered and direction_triggered:
                confidence = Confidence.HIGH
                confidence_score = 0.85
            elif magnitude_triggered:
                confidence = Confidence.MEDIUM
                confidence_score = 0.65
            else:
                confidence = Confidence.LOW
                confidence_score = 0.45

            new_thread = Thread(
                id=str(uuid.uuid4()),
                name=None,
                start_node_id="",
                end_node_id=None,
                message_count=1,
                created_at=datetime.now(),
                metadata={'delta_magnitude': magnitude, 'direction_change': direction_change}
            )
        else:
            confidence = Confidence.LOW
            confidence_score = 0.3
            reasoning = (
                f"Delta below thresholds: magnitude={magnitude:.3f} "
                f"(threshold={threshold:.3f}), direction={direction_change:.3f}"
            )
            new_thread = None

        # Update state
        self._magnitude_history.append(magnitude)
        if len(self._magnitude_history) > 20:
            self._magnitude_history = self._magnitude_history[-20:]

        self._direction_history.append(direction_change)
        if len(self._direction_history) > 20:
            self._direction_history = self._direction_history[-20:]

        self._prev_delta = delta

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
            metadata={'history_len': len(self._magnitude_history)}
        )

    def reset(self) -> None:
        """Reset state for new conversation."""
        self._magnitude_history = []
        self._direction_history = []
        self._prev_delta = None

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Not implemented - Delta is incremental, not batch."""
        return []

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Not implemented for Delta strategy."""
        return []

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Not implemented for Delta strategy."""
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="Delta strategy does not support context retrieval"
        )
