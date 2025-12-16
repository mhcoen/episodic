"""
Topic strategy abstraction for pluggable topic detection and retrieval.

This module provides the base infrastructure for experimenting with different
topic segmentation and context retrieval approaches. Strategies are pluggable
and configurable, enabling empirical comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class Confidence(Enum):
    """Confidence levels for topic decisions."""
    HIGH = "high"        # Strong signal, act on it
    MEDIUM = "medium"    # Moderate signal, consider acting
    LOW = "low"          # Weak signal, likely noise
    UNCERTAIN = "uncertain"  # Not enough information


@dataclass
class Thread:
    """
    A contiguous segment of conversation about a coherent topic.

    Threads are the unit of topic organization - messages are linear
    within a thread, but threads can link to multiple parent threads.
    """
    id: str
    name: Optional[str]  # Human-readable topic name, may be None initially
    start_node_id: str
    end_node_id: Optional[str] = None  # None if thread is ongoing
    message_count: int = 0
    created_at: Optional[datetime] = None

    # Optional: the actual messages in this thread
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # Optional: embedding/centroid for semantic matching
    embedding: Optional[List[float]] = None

    # Optional: summary of thread content
    summary: Optional[str] = None

    # Metadata for debugging/analysis
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadLink:
    """
    A weighted link between threads in the conversation DAG.

    Links can be:
    - Linear (immediate predecessor in time)
    - Semantic (continuation of an earlier topic)
    """
    from_thread_id: str
    to_thread_id: str
    weight: float  # 0.0 to 1.0, strength of connection
    link_type: str  # "linear", "semantic", "explicit" (user stated)
    confidence: Confidence

    # What triggered this link detection
    trigger: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedContext:
    """
    Context retrieved from past threads for inclusion in LLM prompt.
    """
    threads: List[Thread]
    messages: List[Dict[str, Any]]  # The actual message content

    # How relevant each thread is to the current query
    relevance_scores: Dict[str, float]  # thread_id -> score

    # Total tokens in retrieved context
    token_count: int

    # Why this context was retrieved
    retrieval_reason: str

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicDecision:
    """
    A decision made by a topic strategy.

    Captures both the decision and the reasoning, enabling
    debugging, logging, and evaluation.
    """
    # What was decided
    topic_changed: bool
    new_thread: Optional[Thread]
    thread_links: List[ThreadLink]
    retrieved_context: Optional[RetrievedContext]

    # Confidence in the decision
    confidence: Confidence
    confidence_score: float  # 0.0 to 1.0

    # What strategy made this decision
    strategy_name: str
    strategy_version: str

    # Why this decision was made (for debugging)
    reasoning: str
    signals: Dict[str, float]  # Individual signal values

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


class TopicStrategy(ABC):
    """
    Abstract base class for topic detection and retrieval strategies.

    Strategies are pluggable - implement this interface to create
    a new approach that can be swapped in via configuration.

    The interface separates concerns:
    - segment_conversation: Identify thread boundaries
    - detect_thread_link: Detect when current query relates to past threads
    - retrieve_context: Get relevant context from past threads
    - get_decision: Combined decision-making (calls the above)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy-specific parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"

    @abstractmethod
    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """
        Identify thread boundaries in a conversation.

        Args:
            messages: List of messages with content, role, node_id, timestamp

        Returns:
            List of Thread objects representing conversation segments
        """
        pass

    @abstractmethod
    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """
        Detect if the current query relates to any past threads.

        Args:
            query: The current user message
            recent_context: Recent messages in current thread
            past_threads: Previously identified threads

        Returns:
            List of ThreadLinks if connections detected, empty list otherwise
        """
        pass

    @abstractmethod
    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """
        Retrieve relevant context from linked threads.

        Args:
            query: The current user message
            thread_links: Detected links to past threads
            threads: All available threads
            max_tokens: Maximum tokens to retrieve

        Returns:
            RetrievedContext with messages from relevant threads
        """
        pass

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None,
        semantic_drift: Optional[float] = None,
        **kwargs
    ) -> TopicDecision:
        """
        Make a complete topic decision for the current query.

        This is the main entry point that combines segmentation,
        link detection, and retrieval into a single decision.

        Args:
            query: The current user message
            messages: Full conversation history
            current_thread: The currently active thread, if any
            semantic_drift: Pre-computed embedding drift score (0-1), used for
                           hybrid trigger to fast-path into SUSPECT state
            **kwargs: Additional strategy-specific parameters

        Returns:
            TopicDecision capturing what to do and why
        """
        import time
        start_time = time.time()

        # Segment conversation into threads
        threads = self.segment_conversation(messages)

        # Separate current thread from past threads
        past_threads = [t for t in threads if t.id != (current_thread.id if current_thread else None)]

        # Get recent context (last few messages)
        recent_context = messages[-10:] if messages else []

        # Detect links to past threads
        thread_links = self.detect_thread_link(query, recent_context, past_threads)

        # Determine if topic changed
        topic_changed = self._detect_topic_change(query, recent_context, threads)

        # Retrieve context if we found relevant past threads
        retrieved_context = None
        if thread_links:
            retrieved_context = self.retrieve_context(query, thread_links, threads)

        # Build the decision
        processing_time = (time.time() - start_time) * 1000

        return TopicDecision(
            topic_changed=topic_changed,
            new_thread=self._create_new_thread(query) if topic_changed else None,
            thread_links=thread_links,
            retrieved_context=retrieved_context,
            confidence=self._aggregate_confidence(thread_links),
            confidence_score=self._calculate_confidence_score(thread_links),
            strategy_name=self.name,
            strategy_version=self.version,
            reasoning=self._build_reasoning(topic_changed, thread_links),
            signals=self._collect_signals(query, recent_context, threads),
            processing_time_ms=processing_time
        )

    def _detect_topic_change(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        threads: List[Thread]
    ) -> bool:
        """
        Detect if the query represents a topic change.

        Override in subclasses for custom logic.
        """
        # Default: no topic change
        return False

    def _create_new_thread(self, query: str) -> Thread:
        """Create a new thread for a topic change."""
        import uuid
        return Thread(
            id=str(uuid.uuid4()),
            name=None,  # To be extracted later
            start_node_id="",  # Set by caller
            end_node_id=None,
            message_count=1,
            created_at=datetime.now()
        )

    def _aggregate_confidence(self, thread_links: List[ThreadLink]) -> Confidence:
        """Aggregate confidence from multiple thread links."""
        if not thread_links:
            return Confidence.UNCERTAIN

        # Use highest confidence from links
        confidences = [link.confidence for link in thread_links]
        if Confidence.HIGH in confidences:
            return Confidence.HIGH
        elif Confidence.MEDIUM in confidences:
            return Confidence.MEDIUM
        elif Confidence.LOW in confidences:
            return Confidence.LOW
        return Confidence.UNCERTAIN

    def _calculate_confidence_score(self, thread_links: List[ThreadLink]) -> float:
        """Calculate numeric confidence score from thread links."""
        if not thread_links:
            return 0.0

        # Average weight of links
        return sum(link.weight for link in thread_links) / len(thread_links)

    def _build_reasoning(
        self,
        topic_changed: bool,
        thread_links: List[ThreadLink]
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts = []

        if topic_changed:
            parts.append("Topic change detected.")
        else:
            parts.append("Continuing current topic.")

        if thread_links:
            links_desc = ", ".join(
                f"{link.to_thread_id} ({link.link_type}, weight={link.weight:.2f})"
                for link in thread_links
            )
            parts.append(f"Found links to past threads: {links_desc}")
        else:
            parts.append("No links to past threads detected.")

        return " ".join(parts)

    def _collect_signals(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        threads: List[Thread]
    ) -> Dict[str, float]:
        """
        Collect signal values for logging/debugging.

        Override in subclasses to add strategy-specific signals.
        """
        return {
            "query_length": len(query),
            "recent_context_length": len(recent_context),
            "num_threads": len(threads)
        }


class NullStrategy(TopicStrategy):
    """
    A no-op strategy that does nothing.

    Useful as a baseline and for testing.
    """

    def segment_conversation(self, messages: List[Dict[str, Any]]) -> List[Thread]:
        """Return empty list - no segmentation."""
        return []

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Return empty list - no link detection."""
        return []

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Return empty context."""
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="NullStrategy retrieves nothing"
        )
