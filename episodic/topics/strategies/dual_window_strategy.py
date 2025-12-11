"""
Dual-window topic strategy.

Wraps the existing DualWindowDetector as a pluggable TopicStrategy,
enabling comparison with alternative approaches.
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
from episodic.topics.dual_window_detector import DualWindowDetector
from episodic.config import config
from episodic.debug_utils import debug_print


class DualWindowStrategy(TopicStrategy):
    """
    Topic strategy using dual-window (4,1) + (4,2) detection.

    This wraps the existing DualWindowDetector implementation to
    conform to the TopicStrategy interface for pluggable experimentation.
    """

    def __init__(self, strategy_config: Dict[str, Any] = None):
        """Initialize with optional configuration overrides."""
        super().__init__(strategy_config)
        self.name = "DualWindowStrategy"
        self.version = "1.0.0"

        # Initialize the underlying detector
        self.detector = DualWindowDetector()

        # Track detected threads (in-memory for now)
        self._threads: Dict[str, Thread] = {}
        self._current_thread_id: Optional[str] = None

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """
        Segment conversation into threads based on topic boundaries.

        Note: The dual-window detector works incrementally (message by message),
        so this reconstructs threads from the existing topic table or
        re-processes the full conversation.
        """
        # For now, return threads from database if available
        # This is a limitation of the dual-window approach - it's incremental,
        # not batch-oriented
        try:
            from episodic.db_topics import get_all_topics

            db_topics = get_all_topics()
            threads = []

            for topic in db_topics:
                thread = Thread(
                    id=str(topic.get('id', uuid.uuid4())),
                    name=topic.get('name'),
                    start_node_id=topic.get('start_node_id', ''),
                    end_node_id=topic.get('end_node_id'),
                    message_count=topic.get('message_count', 0),
                    created_at=topic.get('created_at', datetime.now()),
                    metadata={
                        'confidence': topic.get('confidence'),
                        'source': 'database'
                    }
                )
                threads.append(thread)
                self._threads[thread.id] = thread

            return threads

        except Exception as e:
            debug_print(f"Error loading threads from database: {e}", category="topic")
            return list(self._threads.values())

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """
        Detect if query relates to past threads.

        The dual-window detector doesn't natively support thread linking,
        so this is a basic implementation that could be enhanced.
        """
        # For now, the dual-window detector doesn't detect thread links
        # This is a gap in the current implementation that the new
        # architecture is designed to address

        # Future: Could use embedding similarity between query and
        # thread summaries/centroids to detect links

        return []

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """
        Retrieve context from linked threads.

        Since detect_thread_link returns empty for dual-window,
        this will typically return empty context.
        """
        if not thread_links:
            return RetrievedContext(
                threads=[],
                messages=[],
                relevance_scores={},
                token_count=0,
                retrieval_reason="No thread links detected by DualWindowStrategy"
            )

        # If we had thread links, we'd retrieve messages from those threads
        # For now, return empty since dual-window doesn't support this
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="DualWindowStrategy does not support context retrieval (yet)"
        )

    def _detect_topic_change(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        threads: List[Thread]
    ) -> bool:
        """
        Detect if query represents a topic change using dual-window approach.
        """
        if not recent_context:
            return False

        # Call the underlying detector
        topic_changed, _, detection_info = self.detector.detect_topic_change(
            recent_messages=recent_context,
            new_message=query
        )

        # Store detection info for signals
        self._last_detection_info = detection_info

        return topic_changed

    def _collect_signals(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        threads: List[Thread]
    ) -> Dict[str, float]:
        """Collect signal values from dual-window detection."""
        signals = super()._collect_signals(query, recent_context, threads)

        # Add dual-window specific signals if available
        if hasattr(self, '_last_detection_info') and self._last_detection_info:
            info = self._last_detection_info

            if info.get('high_precision'):
                hp = info['high_precision']
                signals['high_precision_drift'] = hp.get('drift_score', 0.0)
                signals['high_precision_similarity'] = hp.get('similarity', 0.0)
                signals['high_precision_threshold'] = hp.get('threshold', 0.0)
                signals['high_precision_boundary'] = 1.0 if hp.get('is_boundary') else 0.0

            if info.get('safety_net'):
                sn = info['safety_net']
                signals['safety_net_drift'] = sn.get('drift_score', 0.0)
                signals['safety_net_similarity'] = sn.get('similarity', 0.0)
                signals['safety_net_threshold'] = sn.get('threshold', 0.0)
                signals['safety_net_boundary'] = 1.0 if sn.get('is_boundary') else 0.0

            signals['detection_type'] = {
                'high_precision': 1.0,
                'safety_net': 0.5,
                None: 0.0
            }.get(info.get('detection_type'), 0.0)

        return signals

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make a topic decision using dual-window detection.

        Overrides base implementation to use the detector directly
        for better integration with existing code.
        """
        import time
        start_time = time.time()

        # Segment conversation to get threads
        threads = self.segment_conversation(messages)

        # Recent context for detection (newest first, as detector expects)
        recent_context = messages[-10:][::-1] if messages else []

        # Run detection
        topic_changed, new_topic_name, detection_info = self.detector.detect_topic_change(
            recent_messages=recent_context,
            new_message=query,
            current_topic=None
        )

        # Store for signal collection
        self._last_detection_info = detection_info

        # Build signals
        signals = self._collect_signals(query, recent_context, threads)

        # Determine confidence
        confidence = Confidence.UNCERTAIN
        confidence_score = 0.0

        if detection_info:
            detection_type = detection_info.get('detection_type')
            if detection_type == 'high_precision':
                confidence = Confidence.HIGH
                hp = detection_info.get('high_precision', {})
                confidence_score = 1.0 - hp.get('similarity', 0.5)
            elif detection_type == 'safety_net':
                confidence = Confidence.MEDIUM
                sn = detection_info.get('safety_net', {})
                confidence_score = 1.0 - sn.get('similarity', 0.5)
            else:
                confidence = Confidence.LOW
                confidence_score = 0.3

        # Build reasoning
        reasoning = self._build_reasoning_from_detection(detection_info, topic_changed)

        # Create new thread if topic changed
        new_thread = None
        if topic_changed:
            new_thread = Thread(
                id=str(uuid.uuid4()),
                name=new_topic_name,
                start_node_id="",  # Will be set by caller
                end_node_id=None,
                message_count=1,
                created_at=datetime.now(),
                metadata={'detection_info': detection_info}
            )

        processing_time = (time.time() - start_time) * 1000

        return TopicDecision(
            topic_changed=topic_changed,
            new_thread=new_thread,
            thread_links=[],  # Dual-window doesn't detect thread links
            retrieved_context=None,  # Dual-window doesn't retrieve context
            confidence=confidence,
            confidence_score=confidence_score,
            strategy_name=self.name,
            strategy_version=self.version,
            reasoning=reasoning,
            signals=signals,
            processing_time_ms=processing_time,
            metadata={'detection_info': detection_info}
        )

    def _build_reasoning_from_detection(
        self,
        detection_info: Optional[Dict[str, Any]],
        topic_changed: bool
    ) -> str:
        """Build reasoning string from detection info."""
        if not detection_info:
            return "No detection info available."

        parts = []

        if topic_changed:
            detection_type = detection_info.get('detection_type', 'unknown')
            if detection_type == 'high_precision':
                hp = detection_info.get('high_precision', {})
                parts.append(
                    f"Topic change detected by high precision (4,1) window. "
                    f"Drift={hp.get('drift_score', 0):.3f}, "
                    f"Similarity={hp.get('similarity', 0):.3f} < threshold={hp.get('threshold', 0):.3f}"
                )
            elif detection_type == 'safety_net':
                sn = detection_info.get('safety_net', {})
                parts.append(
                    f"Topic change detected by safety net (4,2) window. "
                    f"Drift={sn.get('drift_score', 0):.3f}, "
                    f"Similarity={sn.get('similarity', 0):.3f} < threshold={sn.get('threshold', 0):.3f}"
                )
        else:
            parts.append("No topic change detected.")
            hp = detection_info.get('high_precision', {})
            if hp:
                parts.append(
                    f"High precision similarity={hp.get('similarity', 0):.3f} >= threshold={hp.get('threshold', 0):.3f}"
                )

        return " ".join(parts)
