"""
Hybrid topic detection system for Episodic.

This file maintains backward compatibility by importing from the new
topics.hybrid module. All functionality has been moved to:
- episodic.topics.hybrid: HybridTopicDetector and HybridScorer
- episodic.topics.keywords: TransitionDetector and TopicChangeSignals
"""

# Import everything from the new module structure for backward compatibility
from episodic.topics.keywords import TransitionDetector
from episodic.topics.hybrid import HybridScorer, HybridTopicDetector

# Re-export the main detection function
def detect_topic_change_hybrid(
    recent_messages,
    new_message,
    current_topic=None
):
    """
    Hybrid topic detection wrapper for backward compatibility.
    
    This function wraps the HybridTopicDetector class method.
    Returns only the first 3 values for backward compatibility.
    """
    detector = HybridTopicDetector()
    changed, new_topic, cost_info, debug_info = detector.detect_topic_change(
        recent_messages, new_message, current_topic
    )
    # Return only first 3 values for backward compatibility
    return changed, new_topic, cost_info

# For complete backward compatibility, create module-level instances
# (though these weren't used in the original, it's good practice)
_transition_detector = TransitionDetector()
_hybrid_scorer = HybridScorer()
_hybrid_detector = HybridTopicDetector()