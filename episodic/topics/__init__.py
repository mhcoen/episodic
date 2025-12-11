"""
Topic detection and management module for Episodic.

This module provides various approaches to topic detection and management:
- TopicManager: Main topic management class
- HybridTopicDetector: Multi-signal topic detection
- SlidingWindowDetector: Window-based drift detection
- TransitionDetector: Keyword and transition detection

New strategy-based architecture (experimental):
- TopicStrategy: Abstract base class for pluggable strategies
- DualWindowStrategy: Dual-window (4,1)+(4,2) strategy
- strategy_registry: Factory for creating strategies by name
- decision_logging: Log decisions for debugging/evaluation
"""

from .detector import TopicManager
from .hybrid import HybridTopicDetector
from .windows import SlidingWindowDetector
from .keywords import TransitionDetector
from .boundaries import analyze_topic_boundary, find_transition_point_heuristic
from .utils import (
    build_conversation_segment,
    is_node_in_topic_range,
    count_nodes_in_topic,
)

# New strategy-based architecture
from .strategy import (
    TopicStrategy,
    Thread,
    ThreadLink,
    RetrievedContext,
    TopicDecision,
    Confidence,
    NullStrategy,
)
from .strategy_registry import (
    get_strategy,
    get_current_strategy,
    list_strategies,
    register_strategy,
    reset_strategy,
)
from .decision_logging import (
    DecisionLogger,
    get_decision_logger,
    log_topic_decision,
)
from .topic_retrieval import (
    retrieve_topic_context,
    format_topic_context,
    get_topic_messages,
)

# Keep backward compatibility
from .detector import (
    detect_topic_change_separately,
    extract_topic_ollama,
    should_create_first_topic,
    topic_manager,  # Export the global instance
)

# Re-export for backward compatibility
from .utils import _display_topic_evolution


def detect_topic_change_hybrid(
    recent_messages,
    new_message,
    current_topic=None
):
    """
    Hybrid topic detection wrapper for backward compatibility.
    
    This function wraps the HybridTopicDetector class method.
    """
    from .hybrid import HybridTopicDetector
    detector = HybridTopicDetector()
    return detector.detect_topic_change(recent_messages, new_message, current_topic)


__all__ = [
    # Main classes (legacy)
    'TopicManager',
    'HybridTopicDetector',
    'SlidingWindowDetector',
    'TransitionDetector',

    # New strategy-based architecture
    'TopicStrategy',
    'Thread',
    'ThreadLink',
    'RetrievedContext',
    'TopicDecision',
    'Confidence',
    'NullStrategy',
    'get_strategy',
    'get_current_strategy',
    'list_strategies',
    'register_strategy',
    'reset_strategy',
    'DecisionLogger',
    'get_decision_logger',
    'log_topic_decision',
    'retrieve_topic_context',
    'format_topic_context',
    'get_topic_messages',

    # Functions (legacy)
    'detect_topic_change_separately',
    'extract_topic_ollama',
    'should_create_first_topic',
    'analyze_topic_boundary',
    'find_transition_point_heuristic',
    'build_conversation_segment',
    'is_node_in_topic_range',
    'count_nodes_in_topic',
    '_display_topic_evolution',
    'detect_topic_change_hybrid',
    'topic_manager',  # Export the global instance
]