"""
Topic detection and management functionality for Episodic.

This file maintains backward compatibility by importing from the new
topics module structure. All functionality has been reorganized into:
- episodic.topics.detector: Main TopicManager class
- episodic.topics.hybrid: Hybrid detection system
- episodic.topics.windows: Sliding window detection
- episodic.topics.keywords: Keyword-based detection
- episodic.topics.boundaries: Boundary analysis
- episodic.topics.utils: Utility functions
"""

# Import everything from the new module structure for backward compatibility
from episodic.topics import (
    # Main classes
    TopicManager,
    HybridTopicDetector,
    SlidingWindowDetector,
    TransitionDetector,
    
    # Functions
    detect_topic_change_separately,
    extract_topic_ollama,
    should_create_first_topic,
    analyze_topic_boundary,
    find_transition_point_heuristic,
    build_conversation_segment,
    is_node_in_topic_range,
    count_nodes_in_topic,
    _display_topic_evolution,
)

# Create global instance for backward compatibility

# Module-level functions that were previously defined here
# These are now imported from the detector module but we re-export them
# at module level for backward compatibility


def detect_topic_change_hybrid(
    recent_messages,
    new_message,
    current_topic=None
):
    """
    Hybrid topic detection wrapper for backward compatibility.
    
    This function wraps the HybridTopicDetector class method.
    """
    from episodic.topics.hybrid import HybridTopicDetector
    detector = HybridTopicDetector()
    return detector.detect_topic_change(recent_messages, new_message, current_topic)