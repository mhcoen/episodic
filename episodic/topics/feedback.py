"""
User feedback capture for topic retrieval.

Allows users to signal when topic context retrieval was helpful
or missed expected context, feeding back into evaluation and tuning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from episodic.config import config

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback users can provide."""
    HELPFUL = "helpful"           # Retrieved context was useful
    NOT_HELPFUL = "not_helpful"   # Retrieved context was irrelevant
    MISSING = "missing"           # Expected context wasn't retrieved
    FALSE_POSITIVE = "false_positive"  # Topic change detected incorrectly
    FALSE_NEGATIVE = "false_negative"  # Topic change missed


@dataclass
class TopicFeedback:
    """A single piece of user feedback about topic handling."""
    feedback_type: FeedbackType
    timestamp: datetime
    query: str

    # Context about what happened
    strategy_name: Optional[str] = None
    topic_changed_detected: Optional[bool] = None
    context_retrieved: bool = False
    retrieved_thread_names: List[str] = None

    # User's correction/note
    expected_topic: Optional[str] = None
    user_note: Optional[str] = None

    # Session context
    session_id: Optional[str] = None
    node_id: Optional[str] = None

    def __post_init__(self):
        if self.retrieved_thread_names is None:
            self.retrieved_thread_names = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feedback_type': self.feedback_type.value,
            'timestamp': self.timestamp.isoformat(),
            'query': self.query,
            'strategy_name': self.strategy_name,
            'topic_changed_detected': self.topic_changed_detected,
            'context_retrieved': self.context_retrieved,
            'retrieved_thread_names': self.retrieved_thread_names,
            'expected_topic': self.expected_topic,
            'user_note': self.user_note,
            'session_id': self.session_id,
            'node_id': self.node_id,
        }


class FeedbackStore:
    """
    Stores and retrieves user feedback about topic handling.

    Feedback is stored in a JSONL file for easy analysis and
    potential use as training/evaluation data.
    """

    def __init__(self, feedback_path: Optional[str] = None):
        """
        Initialize the feedback store.

        Args:
            feedback_path: Path to feedback file. If None, uses config default.
        """
        if feedback_path is None:
            feedback_path = config.get(
                'topic_feedback_path',
                'logs/topic_feedback.jsonl'
            )

        base_dir = Path.home() / ".episodic"
        self.feedback_path = base_dir / feedback_path
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)

        # Track recent context for feedback correlation
        self._recent_context: Optional[Dict[str, Any]] = None

    def set_recent_context(
        self,
        query: str,
        strategy_name: str,
        topic_changed: bool,
        context_retrieved: bool,
        retrieved_threads: List[str],
        node_id: Optional[str] = None
    ) -> None:
        """
        Set context from most recent topic decision for feedback correlation.

        Called after each topic decision so feedback can be linked.
        """
        self._recent_context = {
            'query': query,
            'strategy_name': strategy_name,
            'topic_changed': topic_changed,
            'context_retrieved': context_retrieved,
            'retrieved_threads': retrieved_threads,
            'node_id': node_id,
            'timestamp': datetime.now()
        }

    def record_feedback(
        self,
        feedback_type: FeedbackType,
        user_note: Optional[str] = None,
        expected_topic: Optional[str] = None,
        query: Optional[str] = None
    ) -> TopicFeedback:
        """
        Record user feedback about topic handling.

        Args:
            feedback_type: Type of feedback
            user_note: Optional user explanation
            expected_topic: What topic user expected (for MISSING feedback)
            query: Override query (uses recent context if None)

        Returns:
            The recorded feedback object
        """
        ctx = self._recent_context or {}

        feedback = TopicFeedback(
            feedback_type=feedback_type,
            timestamp=datetime.now(),
            query=query or ctx.get('query', ''),
            strategy_name=ctx.get('strategy_name'),
            topic_changed_detected=ctx.get('topic_changed'),
            context_retrieved=ctx.get('context_retrieved', False),
            retrieved_thread_names=ctx.get('retrieved_threads', []),
            expected_topic=expected_topic,
            user_note=user_note,
            node_id=ctx.get('node_id'),
        )

        self._write_feedback(feedback)
        logger.info(f"Recorded topic feedback: {feedback_type.value}")

        return feedback

    def _write_feedback(self, feedback: TopicFeedback) -> None:
        """Write feedback to the JSONL file."""
        with open(self.feedback_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback.to_dict(), ensure_ascii=False) + '\n')

    def read_feedback(
        self,
        limit: Optional[int] = None,
        feedback_type: Optional[FeedbackType] = None
    ) -> List[TopicFeedback]:
        """
        Read feedback entries from the file.

        Args:
            limit: Maximum entries to return (most recent first)
            feedback_type: Filter by feedback type

        Returns:
            List of TopicFeedback objects
        """
        if not self.feedback_path.exists():
            return []

        entries = []
        with open(self.feedback_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    fb_type = FeedbackType(data['feedback_type'])

                    if feedback_type and fb_type != feedback_type:
                        continue

                    entries.append(TopicFeedback(
                        feedback_type=fb_type,
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        query=data['query'],
                        strategy_name=data.get('strategy_name'),
                        topic_changed_detected=data.get('topic_changed_detected'),
                        context_retrieved=data.get('context_retrieved', False),
                        retrieved_thread_names=data.get('retrieved_thread_names', []),
                        expected_topic=data.get('expected_topic'),
                        user_note=data.get('user_note'),
                        session_id=data.get('session_id'),
                        node_id=data.get('node_id'),
                    ))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse feedback entry: {e}")
                    continue

        # Most recent first
        entries.reverse()

        if limit:
            entries = entries[:limit]

        return entries

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.

        Returns:
            Dict with feedback statistics
        """
        entries = self.read_feedback()

        if not entries:
            return {
                'total': 0,
                'by_type': {},
                'by_strategy': {},
            }

        by_type = {}
        by_strategy = {}

        for entry in entries:
            # Count by type
            type_name = entry.feedback_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count by strategy
            if entry.strategy_name:
                by_strategy[entry.strategy_name] = by_strategy.get(entry.strategy_name, 0) + 1

        # Calculate helpful ratio
        helpful = by_type.get('helpful', 0)
        not_helpful = by_type.get('not_helpful', 0)
        total_rated = helpful + not_helpful
        helpful_ratio = helpful / total_rated if total_rated > 0 else None

        return {
            'total': len(entries),
            'by_type': by_type,
            'by_strategy': by_strategy,
            'helpful_ratio': helpful_ratio,
        }

    def clear_feedback(self) -> None:
        """Clear all stored feedback."""
        if self.feedback_path.exists():
            self.feedback_path.unlink()


# Singleton instance
_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Get the singleton feedback store instance."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store


def record_helpful() -> TopicFeedback:
    """Quick helper to record that context retrieval was helpful."""
    return get_feedback_store().record_feedback(FeedbackType.HELPFUL)


def record_not_helpful(note: Optional[str] = None) -> TopicFeedback:
    """Quick helper to record that context retrieval wasn't helpful."""
    return get_feedback_store().record_feedback(FeedbackType.NOT_HELPFUL, user_note=note)


def record_missing_context(expected_topic: str, note: Optional[str] = None) -> TopicFeedback:
    """Quick helper to record that expected context was missing."""
    return get_feedback_store().record_feedback(
        FeedbackType.MISSING,
        expected_topic=expected_topic,
        user_note=note
    )
