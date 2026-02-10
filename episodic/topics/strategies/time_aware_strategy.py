"""
Time-aware topic detection strategy.

Incorporates message timing into boundary detection. Long gaps between
messages often indicate topic changes, especially combined with semantic drift.

Implements a Bayesian-style combination:
    P(boundary | time_gap, semantic_drift) ∝ P(time_gap | boundary) * P(drift | boundary)

This reduces false positives from irregular conversation rhythms and
captures the common pattern of "returning after a break = new topic".
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


class TimeAwareStrategy(TopicStrategy):
    """
    Time-aware topic boundary detection.

    Combines temporal signals with semantic drift:
    - Long gaps strongly suggest new topics
    - Short gaps require stronger semantic evidence
    - Very short gaps (< 1 min) rarely indicate boundaries

    Designed for production use where timestamps are available.
    """

    def __init__(self, strategy_config: Dict[str, Any] = None):
        """
        Initialize time-aware strategy.

        Args:
            strategy_config: Optional parameters:
                - long_gap_minutes: Gap that strongly suggests boundary (default: 30)
                - medium_gap_minutes: Gap that moderately suggests boundary (default: 10)
                - short_gap_minutes: Gap below which boundaries are unlikely (default: 1)
                - drift_threshold: Semantic drift threshold (default: 0.4)
                - time_weight: Weight of time signal vs drift (default: 0.4)
        """
        super().__init__(strategy_config)
        strategy_config = strategy_config or {}

        self.name = "TimeAwareStrategy"
        self.version = "1.0.0"

        # Time thresholds (in minutes)
        self.long_gap = strategy_config.get('long_gap_minutes', 30)
        self.medium_gap = strategy_config.get('medium_gap_minutes', 10)
        self.short_gap = strategy_config.get('short_gap_minutes', 1)

        # Semantic parameters
        self.drift_threshold = strategy_config.get('drift_threshold', 0.4)
        self.time_weight = strategy_config.get('time_weight', 0.4)

        # Drift calculator
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")
        self._drift_calc = ConversationalDrift(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )

    def _compute_time_gap(
        self,
        messages: List[Dict[str, Any]],
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Compute time gap in minutes from last message.

        Returns 0 if no timestamp available.
        """
        if not messages:
            return 0.0

        last_msg = messages[-1]

        # Try to get timestamp from message
        last_time = last_msg.get('created_at') or last_msg.get('timestamp')

        if last_time is None:
            return 0.0

        # Parse if string
        if isinstance(last_time, str):
            try:
                last_time = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                return 0.0

        # Current time
        if current_time is None:
            current_time = datetime.now()

        # Handle timezone-aware vs naive
        if last_time.tzinfo is not None and current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=last_time.tzinfo)
        elif last_time.tzinfo is None and current_time.tzinfo is not None:
            last_time = last_time.replace(tzinfo=current_time.tzinfo)

        gap = current_time - last_time
        return gap.total_seconds() / 60.0  # Convert to minutes

    def _compute_drift(
        self,
        query: str,
        messages: List[Dict[str, Any]]
    ) -> float:
        """Compute semantic drift between query and recent context."""
        if not messages:
            return 0.0

        recent = messages[-4:] if len(messages) >= 4 else messages
        context_text = " ".join(m.get('content', '') for m in recent)

        query_emb = self._drift_calc.embedding_provider.embed(query)
        context_emb = self._drift_calc.embedding_provider.embed(context_text)

        dot = sum(a * b for a, b in zip(query_emb, context_emb))
        norm_q = sum(a * a for a in query_emb) ** 0.5
        norm_c = sum(b * b for b in context_emb) ** 0.5

        if norm_q == 0 or norm_c == 0:
            return 0.0

        similarity = dot / (norm_q * norm_c)
        return 1.0 - similarity

    def _time_score(self, gap_minutes: float) -> float:
        """
        Convert time gap to boundary probability.

        Returns value in [0, 1]:
        - 0 for very short gaps
        - 0.5 for medium gaps
        - 1.0 for long gaps
        """
        if gap_minutes <= 0:
            return 0.0

        if gap_minutes >= self.long_gap:
            return 1.0
        elif gap_minutes >= self.medium_gap:
            # Linear interpolation from 0.5 to 1.0
            t = (gap_minutes - self.medium_gap) / (self.long_gap - self.medium_gap)
            return 0.5 + 0.5 * t
        elif gap_minutes >= self.short_gap:
            # Linear interpolation from 0.0 to 0.5
            t = (gap_minutes - self.short_gap) / (self.medium_gap - self.short_gap)
            return 0.5 * t
        else:
            return 0.0

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None,
        current_time: Optional[datetime] = None
    ) -> TopicDecision:
        """
        Make topic decision using time and semantic signals.
        """
        import time
        start_time = time.time()

        if not messages:
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.LOW,
                confidence_score=0.1,
                strategy_name=self.name,
                strategy_version=self.version,
                reasoning="No messages to compare",
                signals={},
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={}
            )

        # Compute signals
        time_gap = self._compute_time_gap(messages, current_time)
        drift = self._compute_drift(query, messages)

        time_score = self._time_score(time_gap)
        drift_score = drift / self.drift_threshold  # Normalize to ~1.0 at threshold

        # Combined score
        combined = (
            self.time_weight * time_score +
            (1 - self.time_weight) * drift_score
        )

        signals = {
            'time_gap_minutes': time_gap,
            'time_score': time_score,
            'drift': drift,
            'drift_score': drift_score,
            'combined_score': combined,
        }

        # Decision logic
        # Long gap alone can trigger
        if time_gap >= self.long_gap:
            topic_changed = True
            confidence = Confidence.HIGH
            confidence_score = 0.85
            reasoning = f"Long time gap: {time_gap:.1f} minutes >= {self.long_gap}"
        # Medium gap + drift
        elif time_gap >= self.medium_gap and drift > self.drift_threshold * 0.7:
            topic_changed = True
            confidence = Confidence.MEDIUM
            confidence_score = 0.65
            reasoning = f"Medium gap ({time_gap:.1f}min) + semantic drift ({drift:.3f})"
        # Combined threshold
        elif combined >= 0.7:
            topic_changed = True
            confidence = Confidence.MEDIUM
            confidence_score = combined
            reasoning = f"Combined time+drift score: {combined:.3f} >= 0.7"
        # Short gap suppresses even with drift
        elif time_gap < self.short_gap and time_gap > 0:
            topic_changed = False
            confidence = Confidence.LOW
            confidence_score = 0.2
            reasoning = f"Short gap ({time_gap:.1f}min) suppresses boundary"
        # No timing data, fall back to drift only
        elif time_gap == 0:
            topic_changed = drift > self.drift_threshold
            confidence = Confidence.LOW if topic_changed else Confidence.LOW
            confidence_score = drift_score * 0.5  # Lower confidence without time
            reasoning = f"No timing data, drift-only: {drift:.3f}"
        else:
            topic_changed = False
            confidence = Confidence.LOW
            confidence_score = 0.3
            reasoning = f"Below thresholds: gap={time_gap:.1f}min, drift={drift:.3f}"

        new_thread = None
        if topic_changed:
            new_thread = Thread(
                id=str(uuid.uuid4()),
                name=None,
                start_node_id="",
                end_node_id=None,
                message_count=1,
                created_at=datetime.now(),
                metadata={'time_gap': time_gap, 'drift': drift}
            )

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
            metadata={'time_gap': time_gap}
        )

    def reset(self) -> None:
        """Reset state for new conversation."""
        pass  # Stateless

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Not implemented - Time-aware is incremental, not batch."""
        return []

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Not implemented for Time-aware strategy."""
        return []

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Not implemented for Time-aware strategy."""
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="Time-aware strategy does not support context retrieval"
        )
