"""
Keyword-based topic detection strategy.

Uses explicit transition phrases and domain keyword matching
for simple, fast, interpretable topic boundary detection.
"""

import time
from typing import Dict, List, Any, Optional

from episodic.topics.strategy import (
    TopicStrategy,
    TopicDecision,
    Thread,
    ThreadLink,
    RetrievedContext,
    Confidence,
)
from episodic.topics.keywords import TransitionDetector


class KeywordStrategy(TopicStrategy):
    """
    Keyword-based topic detection using transition phrases and domain matching.

    Fast, interpretable baseline that detects:
    - Explicit transitions ("changing topics", "by the way", etc.)
    - Domain shifts (cooking → programming keywords)

    Good for explicit topic changes, less effective for gradual drift.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the keyword strategy.

        Args:
            params: Optional parameters:
                - explicit_threshold: Threshold for explicit transitions (default: 0.5)
                - domain_threshold: Threshold for domain shifts (default: 0.5)
                - combined_threshold: Threshold for combined score (default: 0.4)
        """
        params = params or {}
        self.explicit_threshold = params.get('explicit_threshold', 0.5)
        self.domain_threshold = params.get('domain_threshold', 0.5)
        self.combined_threshold = params.get('combined_threshold', 0.4)

        # Create fresh detector for each evaluation
        self._detector: Optional[TransitionDetector] = None

    @property
    def name(self) -> str:
        return "KeywordStrategy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def _get_detector(self) -> TransitionDetector:
        """Get or create a transition detector."""
        if self._detector is None:
            self._detector = TransitionDetector()
        return self._detector

    def reset(self) -> None:
        """Reset detector state (for evaluation)."""
        self._detector = None

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Segment a conversation into threads based on keywords."""
        threads = []
        current_messages = []
        thread_id = 0

        detector = TransitionDetector()  # Fresh detector

        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                result = detector.detect_transition_keywords(msg.get('content', ''))

                # Check if this is a topic boundary
                is_boundary = (
                    result['explicit_transition'] >= self.explicit_threshold or
                    result['domain_shift'] >= self.domain_threshold
                )

                if is_boundary and current_messages:
                    # Save current thread
                    threads.append(Thread(
                        id=f"thread_{thread_id}",
                        name=result.get('dominant_domain', f'topic_{thread_id}'),
                        start_node_id=current_messages[0].get('node_id', str(thread_id)),
                        messages=current_messages.copy()
                    ))
                    thread_id += 1
                    current_messages = []

            current_messages.append(msg)

        # Add final thread
        if current_messages:
            threads.append(Thread(
                id=f"thread_{thread_id}",
                name=f'topic_{thread_id}',
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
        """Detect if the query links to any previous threads."""
        # Simple implementation: check for domain overlap
        links = []
        detector = TransitionDetector()

        query_result = detector.detect_transition_keywords(query)
        query_domains = set(query_result.get('detected_domains', {}).keys())

        for thread in threads:
            if current_thread and thread.id == current_thread.id:
                continue

            # Check domain overlap with thread
            thread_detector = TransitionDetector()
            thread_domains = set()
            for msg in thread.messages:
                if msg.get('role') == 'user':
                    result = thread_detector.detect_transition_keywords(msg.get('content', ''))
                    thread_domains.update(result.get('detected_domains', {}).keys())

            overlap = query_domains & thread_domains
            if overlap:
                links.append(ThreadLink(
                    from_thread_id=current_thread.id if current_thread else "current",
                    to_thread_id=thread.id,
                    weight=len(overlap) / max(len(query_domains), 1),
                    link_type="domain_overlap"
                ))

        return links

    def retrieve_context(
        self,
        query: str,
        threads: List[Thread],
        current_thread: Optional[Thread] = None,
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Retrieve context from linked threads."""
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
            retrieval_reason="domain_keyword_match" if links else "none"
        )

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Decide if the query represents a topic change.

        Uses keyword detection for explicit transitions and domain shifts.
        """
        start_time = time.time()

        detector = self._get_detector()
        result = detector.detect_transition_keywords(query)

        # Calculate combined score
        explicit_score = result['explicit_transition']
        domain_score = result['domain_shift']
        combined_score = max(explicit_score, domain_score * 0.8)

        # Determine if topic changed
        topic_changed = (
            explicit_score >= self.explicit_threshold or
            domain_score >= self.domain_threshold or
            combined_score >= self.combined_threshold
        )

        # Determine confidence level
        if explicit_score >= 0.8:
            confidence = Confidence.HIGH
        elif explicit_score >= 0.5 or domain_score >= 0.5:
            confidence = Confidence.MEDIUM
        elif combined_score > 0:
            confidence = Confidence.LOW
        else:
            confidence = Confidence.UNCERTAIN

        # Build reasoning
        reasoning_parts = []
        if result.get('found_phrase'):
            reasoning_parts.append(f"Explicit transition: '{result['found_phrase']}'")
        if result.get('dominant_domain'):
            reasoning_parts.append(f"Domain: {result['dominant_domain']}")
        if domain_score > 0:
            reasoning_parts.append(f"Domain shift detected")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No transition signals"

        processing_time = (time.time() - start_time) * 1000

        return TopicDecision(
            topic_changed=topic_changed,
            new_thread=None,
            thread_links=[],
            retrieved_context=None,
            confidence=confidence,
            confidence_score=combined_score,
            reasoning=reasoning,
            signals={
                'explicit_transition': explicit_score,
                'domain_shift': domain_score,
                'combined': combined_score,
                'detected_domains': list(result.get('detected_domains', {}).keys()),
            },
            strategy_name=self.name,
            strategy_version=self.version,
            processing_time_ms=processing_time,
        )
