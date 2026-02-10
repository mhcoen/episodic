"""
Decision logging for topic strategies.

Logs all topic decisions to enable debugging, evaluation,
and building training data for future improvements.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from episodic.topics.strategy import TopicDecision, Confidence
from episodic.config import config

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Logs topic decisions to a JSONL file.

    Each line in the log file is a JSON object containing:
    - The decision details
    - Input context (query, recent messages)
    - Strategy information
    - Timestamp
    """

    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize the decision logger.

        Args:
            log_path: Path to the log file. If None, uses config value.
                      Path is relative to ~/.episodic/
        """
        if log_path is None:
            log_path = config.get('topic_decision_log_path', 'logs/topic_decisions.jsonl')

        # Make path relative to ~/.episodic/
        base_dir = Path.home() / ".episodic"
        self.log_path = base_dir / log_path

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._enabled = config.get('topic_decision_logging', False)

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable decision logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable decision logging."""
        self._enabled = False

    def log_decision(
        self,
        decision: TopicDecision,
        query: str,
        recent_context: list,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a topic decision.

        Args:
            decision: The TopicDecision object
            query: The user query that triggered this decision
            recent_context: Recent messages used for detection
            additional_context: Any additional context to log
        """
        if not self._enabled:
            return

        try:
            log_entry = self._build_log_entry(
                decision, query, recent_context, additional_context
            )
            self._write_log_entry(log_entry)
        except Exception as e:
            logger.error(f"Failed to log topic decision: {e}")

    def _build_log_entry(
        self,
        decision: TopicDecision,
        query: str,
        recent_context: list,
        additional_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build a log entry from a decision."""
        # Convert decision to dict, handling special types
        decision_dict = {
            'topic_changed': decision.topic_changed,
            'confidence': decision.confidence.value if isinstance(decision.confidence, Confidence) else str(decision.confidence),
            'confidence_score': decision.confidence_score,
            'strategy_name': decision.strategy_name,
            'strategy_version': decision.strategy_version,
            'reasoning': decision.reasoning,
            'signals': decision.signals,
            'processing_time_ms': decision.processing_time_ms,
        }

        # Add new thread info if present
        if decision.new_thread:
            decision_dict['new_thread'] = {
                'id': decision.new_thread.id,
                'name': decision.new_thread.name,
            }

        # Add thread links if present
        if decision.thread_links:
            decision_dict['thread_links'] = [
                {
                    'from_thread_id': link.from_thread_id,
                    'to_thread_id': link.to_thread_id,
                    'weight': link.weight,
                    'link_type': link.link_type,
                }
                for link in decision.thread_links
            ]

        # Add retrieved context summary if present
        if decision.retrieved_context:
            decision_dict['retrieved_context'] = {
                'num_threads': len(decision.retrieved_context.threads),
                'num_messages': len(decision.retrieved_context.messages),
                'token_count': decision.retrieved_context.token_count,
                'retrieval_reason': decision.retrieved_context.retrieval_reason,
            }

        # Build full log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'recent_context_length': len(recent_context),
            'recent_context_preview': self._preview_context(recent_context),
            'decision': decision_dict,
        }

        if additional_context:
            log_entry['additional_context'] = additional_context

        return log_entry

    def _preview_context(self, context: list, max_items: int = 3) -> list:
        """Create a preview of the context (first few messages, truncated)."""
        preview = []
        for msg in context[:max_items]:
            content = msg.get('content', '')
            if len(content) > 100:
                content = content[:100] + '...'
            preview.append({
                'role': msg.get('role', 'unknown'),
                'content_preview': content
            })
        return preview

    def _write_log_entry(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to the file."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def read_logs(self, limit: Optional[int] = None) -> list:
        """
        Read log entries from the file.

        Args:
            limit: Maximum number of entries to read (most recent first)

        Returns:
            List of log entry dicts
        """
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Return most recent first
        entries.reverse()

        if limit:
            entries = entries[:limit]

        return entries

    def clear_logs(self) -> None:
        """Clear all logged decisions."""
        if self.log_path.exists():
            self.log_path.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logged decisions.

        Returns:
            Dict with statistics
        """
        entries = self.read_logs()

        if not entries:
            return {
                'total_decisions': 0,
                'topic_changes': 0,
                'no_changes': 0,
                'strategies_used': {},
                'avg_processing_time_ms': 0,
            }

        topic_changes = sum(1 for e in entries if e['decision']['topic_changed'])
        strategies = {}
        total_time = 0

        for entry in entries:
            strategy = entry['decision']['strategy_name']
            strategies[strategy] = strategies.get(strategy, 0) + 1
            total_time += entry['decision'].get('processing_time_ms', 0)

        return {
            'total_decisions': len(entries),
            'topic_changes': topic_changes,
            'no_changes': len(entries) - topic_changes,
            'topic_change_rate': topic_changes / len(entries) if entries else 0,
            'strategies_used': strategies,
            'avg_processing_time_ms': total_time / len(entries) if entries else 0,
        }


# Singleton instance
_decision_logger: Optional[DecisionLogger] = None


def get_decision_logger() -> DecisionLogger:
    """Get the singleton decision logger instance."""
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger()
    return _decision_logger


def log_topic_decision(
    decision: TopicDecision,
    query: str,
    recent_context: list,
    additional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to log a topic decision.

    Args:
        decision: The TopicDecision object
        query: The user query
        recent_context: Recent messages
        additional_context: Any additional context
    """
    get_decision_logger().log_decision(
        decision, query, recent_context, additional_context
    )
