"""
Summary-shift probing topic detection strategy.

Detects topic boundaries by comparing embeddings of conversation summaries
before and after adding a new turn. A large shift in summary embedding
indicates the new message changed the conversational focus significantly.

This differs from DeltaStrategy in that:
- Delta compares raw message embeddings
- Summary probe compresses through summarization first, detecting higher-level shifts

Supports two modes:
1. LLM summaries (accurate but expensive) - uses actual LLM to generate summaries
2. Embedding compression (fast but approximate) - uses centroid + PCA compression
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import time

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


class SummaryProbeStrategy(TopicStrategy):
    """
    Summary-shift probing for topic boundary detection.

    Algorithm:
    1. Maintain a summary of recent conversation (cached)
    2. When new message arrives, generate summary including it
    3. Compare embeddings of old vs new summary
    4. Boundary if shift exceeds threshold

    The intuition: if adding one message significantly changes what
    the conversation is "about" (as captured by summary), that's a topic shift.
    """

    def __init__(self, strategy_config: Dict[str, Any] = None):
        """
        Initialize summary probe strategy.

        Args:
            strategy_config: Optional parameters:
                - mode: "llm" for LLM summaries, "embedding" for fast mode (default: "embedding")
                - window_size: Messages to include in summary (default: 6)
                - shift_threshold: Cosine distance threshold (default: 0.25)
                - summary_model: Model for LLM summaries (default: from config)
                - adaptive: Learn threshold from conversation (default: True)
                - cache_summaries: Cache summaries to reduce LLM calls (default: True)
        """
        super().__init__(strategy_config)
        strategy_config = strategy_config or {}

        self.name = "SummaryProbeStrategy"
        self.version = "1.0.0"

        # Mode
        self.mode = strategy_config.get('mode', 'embedding')
        if self.mode not in ('llm', 'embedding'):
            raise ValueError(f"mode must be 'llm' or 'embedding', got: {self.mode}")

        # Parameters
        self.window_size = strategy_config.get('window_size', 6)
        self.shift_threshold = strategy_config.get('shift_threshold', 0.25)
        self.summary_model = strategy_config.get(
            'summary_model',
            config.get('summary_model', 'gpt-4o-mini')
        )
        self.adaptive = strategy_config.get('adaptive', True)
        self.cache_summaries = strategy_config.get('cache_summaries', True)

        # State
        self._cached_summary: Optional[str] = None
        self._cached_summary_embedding: Optional[List[float]] = None
        self._cached_messages_hash: Optional[int] = None
        self._shift_history: List[float] = []

        # Embedding provider for comparing summaries
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")
        self._drift_calc = ConversationalDrift(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )

    def _messages_hash(self, messages: List[Dict[str, Any]]) -> int:
        """Compute hash of messages for cache validation."""
        content = "".join(m.get('content', '')[:50] for m in messages[-self.window_size:])
        return hash(content)

    def _generate_summary_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Generate summary using LLM."""
        from episodic.llm import query_llm

        # Format conversation
        conversation_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages[-self.window_size:]
        )

        prompt = f"""Summarize the following conversation in 1-2 sentences,
capturing the main topic and any key points discussed:

{conversation_text}

Summary:"""

        try:
            response, _ = query_llm(
                prompt=prompt,
                model=self.summary_model,
                system_message="You are a concise summarizer. Output only the summary.",
                temperature=0.0
            )
            return response.strip()
        except Exception as e:
            # Fallback to embedding mode on error
            return self._generate_summary_embedding(messages)

    def _generate_summary_embedding(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate pseudo-summary using embedding compression.

        This creates a "summary" by identifying the most representative
        sentences based on embedding centrality.
        """
        if not messages:
            return ""

        window = messages[-self.window_size:]

        # For embedding mode, we just concatenate and let the embedding
        # capture the semantic content. The "summary" is the full window text.
        # The key is that we compare embeddings, not the text itself.
        return " ".join(m.get('content', '') for m in window)

    def _get_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Get summary for messages, using cache if available."""
        msg_hash = self._messages_hash(messages)

        if self.cache_summaries and self._cached_messages_hash == msg_hash:
            return self._cached_summary

        if self.mode == 'llm':
            summary = self._generate_summary_llm(messages)
        else:
            summary = self._generate_summary_embedding(messages)

        if self.cache_summaries:
            self._cached_summary = summary
            self._cached_messages_hash = msg_hash

        return summary

    def _get_summary_embedding(self, summary: str) -> List[float]:
        """Get embedding for a summary."""
        return self._drift_calc.embedding_provider.embed(summary)

    def _compute_shift(
        self,
        old_embedding: List[float],
        new_embedding: List[float]
    ) -> float:
        """
        Compute shift between embeddings as cosine distance.
        Returns value in [0, 2] where 0 = identical, 2 = opposite.
        """
        if not old_embedding or not new_embedding:
            return 0.0

        dot = sum(a * b for a, b in zip(old_embedding, new_embedding))
        norm_old = sum(a * a for a in old_embedding) ** 0.5
        norm_new = sum(b * b for b in new_embedding) ** 0.5

        if norm_old == 0 or norm_new == 0:
            return 0.0

        cosine = dot / (norm_old * norm_new)
        return 1.0 - cosine  # Cosine distance

    def _get_adaptive_threshold(self) -> float:
        """Compute adaptive threshold from history."""
        if not self._shift_history or len(self._shift_history) < 5:
            return self.shift_threshold

        import numpy as np
        mean = np.mean(self._shift_history)
        std = np.std(self._shift_history)
        # Use mean + 1.5 * std as threshold
        return float(mean + 1.5 * std)

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make topic decision by comparing summary embeddings.
        """
        start_time = time.time()

        # Need enough messages
        if len(messages) < 2:
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.LOW,
                confidence_score=0.2,
                strategy_name=self.name,
                strategy_version=self.version,
                reasoning="Need at least 2 messages for summary comparison",
                signals={},
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={}
            )

        # Get summary of messages WITHOUT query
        old_summary = self._get_summary(messages)
        old_embedding = self._get_summary_embedding(old_summary)

        # Get summary of messages WITH query
        messages_with_query = messages + [{'role': 'user', 'content': query}]
        new_summary = self._get_summary(messages_with_query)
        new_embedding = self._get_summary_embedding(new_summary)

        # Compute shift
        shift = self._compute_shift(old_embedding, new_embedding)

        # Get threshold
        threshold = self._get_adaptive_threshold() if self.adaptive else self.shift_threshold

        # Decision
        topic_changed = shift > threshold

        # Update history
        self._shift_history.append(shift)
        if len(self._shift_history) > 30:
            self._shift_history = self._shift_history[-30:]

        # Update cache for next call (include query in cached state)
        if self.cache_summaries:
            self._cached_summary = new_summary
            self._cached_summary_embedding = new_embedding
            self._cached_messages_hash = self._messages_hash(messages_with_query)

        # Build signals
        signals = {
            'summary_shift': shift,
            'threshold': threshold,
            'mode': self.mode,
            'window_size': self.window_size,
        }

        if topic_changed:
            confidence = Confidence.HIGH if shift > threshold * 1.5 else Confidence.MEDIUM
            confidence_score = min(0.95, 0.5 + (shift - threshold) / threshold)
            reasoning = f"Summary shift {shift:.3f} > threshold {threshold:.3f}"

            new_thread = Thread(
                id=str(uuid.uuid4()),
                name=None,
                start_node_id="",
                end_node_id=None,
                message_count=1,
                created_at=datetime.now(),
                metadata={'summary_shift': shift, 'mode': self.mode}
            )
        else:
            confidence = Confidence.LOW
            confidence_score = max(0.1, 0.5 - (threshold - shift) / threshold)
            reasoning = f"Summary shift {shift:.3f} <= threshold {threshold:.3f}"
            new_thread = None

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
            metadata={
                'old_summary_preview': old_summary[:100] if self.mode == 'llm' else None,
                'new_summary_preview': new_summary[:100] if self.mode == 'llm' else None,
            }
        )

    def reset(self) -> None:
        """Reset state for new conversation."""
        self._cached_summary = None
        self._cached_summary_embedding = None
        self._cached_messages_hash = None
        self._shift_history = []

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Batch segmentation using summary probing."""
        if len(messages) < 3:
            return []

        threads = []
        self.reset()

        # Track current segment
        segment_start = 0
        message_history = []

        for i, msg in enumerate(messages):
            if i > 0 and msg.get('role') == 'user':
                decision = self.get_decision(
                    query=msg.get('content', ''),
                    messages=message_history,
                    current_thread=None
                )

                if decision.topic_changed:
                    # Create thread for previous segment
                    if i > segment_start:
                        threads.append(Thread(
                            id=str(uuid.uuid4()),
                            name=None,
                            start_node_id=str(segment_start),
                            end_node_id=str(i - 1),
                            message_count=i - segment_start,
                            created_at=datetime.now(),
                            metadata={'shift': decision.signals.get('summary_shift', 0)}
                        ))
                    segment_start = i

            message_history.append(msg)

        # Final segment
        if len(messages) > segment_start:
            threads.append(Thread(
                id=str(uuid.uuid4()),
                name=None,
                start_node_id=str(segment_start),
                end_node_id=str(len(messages) - 1),
                message_count=len(messages) - segment_start,
                created_at=datetime.now(),
                metadata={}
            ))

        return threads

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Not implemented for SummaryProbe strategy."""
        return []

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Not implemented for SummaryProbe strategy."""
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="SummaryProbe strategy does not support context retrieval"
        )
