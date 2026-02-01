"""
Relative embedding strategy for topic detection.

Uses self-calibrating thresholds based on the conversation's own
similarity distribution, making it more robust across model changes.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from episodic.topics.strategy import (
    TopicStrategy,
    TopicDecision,
    Thread,
    ThreadLink,
    RetrievedContext,
    Confidence,
)


class RelativeEmbeddingStrategy(TopicStrategy):
    """
    Topic detection using relative embedding similarity.

    Instead of absolute thresholds (0.7, 0.85), uses relative measures:
    - Compares query similarity to recent vs older messages
    - Topic change if query is significantly more similar to older messages
    - Self-calibrates based on conversation's similarity distribution

    More robust to embedding model changes than absolute thresholds.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the relative embedding strategy.

        Args:
            params: Optional parameters:
                - similarity_ratio_threshold: Ratio of old/recent similarity
                  above which indicates topic return (default: 1.3)
                - drop_ratio_threshold: Ratio of recent/baseline similarity
                  below which indicates topic change (default: 0.7)
                - recent_window: Number of recent messages to compare (default: 4)
                - baseline_window: Number of baseline messages (default: 8)
                - embedding_model: Model for embeddings (default: from config)
        """
        params = params or {}
        self.similarity_ratio_threshold = params.get('similarity_ratio_threshold', 1.3)
        self.drop_ratio_threshold = params.get('drop_ratio_threshold', 0.7)
        self.recent_window = params.get('recent_window', 4)
        self.baseline_window = params.get('baseline_window', 8)

        self._embedder = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "RelativeEmbeddingStrategy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def _get_embedder(self):
        """Lazy load the embedding function."""
        if self._embedder is None:
            try:
                # Suppress noisy transformer model loading output
                import logging
                import warnings
                import os
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                logging.getLogger("transformers").setLevel(logging.ERROR)
                logging.getLogger("safetensors").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", message=".*not sharded.*")
                warnings.filterwarnings("ignore", message=".*position_ids.*")

                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise RuntimeError("sentence-transformers required for RelativeEmbeddingStrategy")
        return self._embedder

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with caching."""
        if text not in self._embedding_cache:
            embedder = self._get_embedder()
            self._embedding_cache[text] = embedder.encode(text, convert_to_numpy=True)
        return self._embedding_cache[text]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_baseline_similarity(
        self,
        messages: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Compute baseline similarity statistics for the conversation.

        Returns:
            Tuple of (mean_similarity, std_similarity)
        """
        # Use all messages, not just user messages
        if len(messages) < 2:
            return 0.5, 0.2  # Default baseline

        # Compute pairwise similarities between consecutive messages
        similarities = []
        for i in range(1, len(messages)):
            prev_emb = self._get_embedding(messages[i-1].get('content', ''))
            curr_emb = self._get_embedding(messages[i].get('content', ''))
            sim = self._cosine_similarity(prev_emb, curr_emb)
            similarities.append(sim)

        if not similarities:
            return 0.5, 0.2

        return float(np.mean(similarities)), float(np.std(similarities))

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Segment conversation using relative similarity drops."""
        threads = []
        current_messages = []
        thread_id = 0

        user_messages = [m for m in messages if m.get('role') == 'user']

        if len(user_messages) < 2:
            return [Thread(
                id="thread_0",
                name="conversation",
                start_node_id=messages[0].get('node_id', '0') if messages else '0',
                messages=messages
            )]

        # Compute baseline
        baseline_mean, baseline_std = self._compute_baseline_similarity(messages)
        threshold = baseline_mean - 1.5 * baseline_std  # 1.5 std below mean

        prev_emb = None
        msg_idx = 0

        for msg in messages:
            if msg.get('role') == 'user':
                curr_emb = self._get_embedding(msg.get('content', ''))

                if prev_emb is not None:
                    sim = self._cosine_similarity(prev_emb, curr_emb)

                    # Significant drop indicates boundary
                    if sim < threshold and current_messages:
                        threads.append(Thread(
                            id=f"thread_{thread_id}",
                            name=f"topic_{thread_id}",
                            start_node_id=current_messages[0].get('node_id', str(thread_id)),
                            messages=current_messages.copy()
                        ))
                        thread_id += 1
                        current_messages = []

                prev_emb = curr_emb

            current_messages.append(msg)
            msg_idx += 1

        # Add final thread
        if current_messages:
            threads.append(Thread(
                id=f"thread_{thread_id}",
                name=f"topic_{thread_id}",
                start_node_id=current_messages[0].get('node_id', str(thread_id)),
                messages=current_messages
            ))

        return threads

    def detect_thread_link(
        self,
        query: str,
        threads: List[Thread],
        current_thread: Optional[Thread] = None
    ) -> List[ThreadLink]:
        """Detect if query links to past threads via embedding similarity."""
        if not threads:
            return []

        query_emb = self._get_embedding(query)
        links = []

        # Get average similarity to current thread
        current_sim = 0.0
        if current_thread and current_thread.messages:
            current_sims = []
            for msg in current_thread.messages:
                if msg.get('role') == 'user':
                    msg_emb = self._get_embedding(msg.get('content', ''))
                    current_sims.append(self._cosine_similarity(query_emb, msg_emb))
            if current_sims:
                current_sim = np.mean(current_sims)

        # Check similarity to other threads
        for thread in threads:
            if current_thread and thread.id == current_thread.id:
                continue

            thread_sims = []
            for msg in thread.messages:
                if msg.get('role') == 'user':
                    msg_emb = self._get_embedding(msg.get('content', ''))
                    thread_sims.append(self._cosine_similarity(query_emb, msg_emb))

            if thread_sims:
                thread_sim = float(np.mean(thread_sims))

                # Link if more similar to old thread than current
                if current_sim > 0:
                    ratio = thread_sim / current_sim
                    if ratio > self.similarity_ratio_threshold:
                        links.append(ThreadLink(
                            from_thread_id=current_thread.id if current_thread else "current",
                            to_thread_id=thread.id,
                            weight=ratio - 1.0,  # How much more similar
                            link_type="embedding_similarity"
                        ))

        return links

    def retrieve_context(
        self,
        query: str,
        threads: List[Thread],
        current_thread: Optional[Thread] = None,
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Retrieve context from most similar threads."""
        links = self.detect_thread_link(query, threads, current_thread)

        retrieved_messages = []
        retrieved_threads = []

        for link in sorted(links, key=lambda l: l.weight, reverse=True):
            for thread in threads:
                if thread.id == link.to_thread_id:
                    retrieved_threads.append(thread)
                    retrieved_messages.extend(thread.messages)
                    break

        return RetrievedContext(
            threads=retrieved_threads,
            messages=retrieved_messages,
            relevance_scores={t.id: 1.0 for t in retrieved_threads},
            token_count=sum(len(m.get('content', '')) // 4 for m in retrieved_messages),
            retrieval_reason="embedding_similarity" if links else "none"
        )

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Decide if query represents a topic change using relative similarity.

        Compares query's similarity to recent messages vs baseline,
        detecting drops that indicate topic shifts.
        """
        start_time = time.time()

        # Need minimum history (use all messages, not just user)
        if len(messages) < 2:
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.UNCERTAIN,
                confidence_score=0.0,
                reasoning="Insufficient history for comparison",
                signals={'message_count': len(messages)},
                strategy_name=self.name,
                strategy_version=self.version,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        query_emb = self._get_embedding(query)

        # Compute baseline statistics
        baseline_mean, baseline_std = self._compute_baseline_similarity(messages)

        # Similarity to recent messages (use all messages, not just user)
        recent = messages[-self.recent_window:]
        recent_sims = []
        for msg in recent:
            msg_emb = self._get_embedding(msg.get('content', ''))
            recent_sims.append(self._cosine_similarity(query_emb, msg_emb))

        recent_sim = float(np.mean(recent_sims)) if recent_sims else 0.0

        # Compute ratio relative to baseline
        if baseline_mean > 0:
            drop_ratio = recent_sim / baseline_mean
        else:
            drop_ratio = 1.0

        # Z-score: how many std deviations below mean
        if baseline_std > 0:
            z_score = (recent_sim - baseline_mean) / baseline_std
        else:
            z_score = 0.0

        # Determine topic change
        # Topic changed if similarity to recent is significantly below baseline
        topic_changed = drop_ratio < self.drop_ratio_threshold or z_score < -1.5

        # Confidence based on how clear the signal is
        if z_score < -2.5:
            confidence = Confidence.HIGH
            confidence_score = min(1.0, abs(z_score) / 3.0)
        elif z_score < -1.5:
            confidence = Confidence.MEDIUM
            confidence_score = 0.5 + (abs(z_score) - 1.5) / 2.0
        elif z_score < -0.5:
            confidence = Confidence.LOW
            confidence_score = 0.3
        else:
            confidence = Confidence.UNCERTAIN
            confidence_score = 0.0

        # Build reasoning
        if topic_changed:
            reasoning = f"Similarity drop: {recent_sim:.3f} vs baseline {baseline_mean:.3f} (z={z_score:.2f})"
        else:
            reasoning = f"Within normal range: {recent_sim:.3f} vs baseline {baseline_mean:.3f} (z={z_score:.2f})"

        processing_time = (time.time() - start_time) * 1000

        return TopicDecision(
            topic_changed=topic_changed,
            new_thread=None,
            thread_links=[],
            retrieved_context=None,
            confidence=confidence,
            confidence_score=confidence_score,
            reasoning=reasoning,
            signals={
                'recent_similarity': recent_sim,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'drop_ratio': drop_ratio,
                'z_score': z_score,
            },
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=processing_time,
        )
