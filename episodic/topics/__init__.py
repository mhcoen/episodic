"""
Topic detection and management module for Episodic.

This module provides various approaches to topic detection and management:
- TopicManager: Main topic management class
- HybridTopicDetector: Multi-signal topic detection
- SlidingWindowDetector: Window-based drift detection
- TransitionDetector: Keyword and transition detection
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
    # Main classes
    'TopicManager',
    'HybridTopicDetector', 
    'SlidingWindowDetector',
    'TransitionDetector',
    
    # Functions
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