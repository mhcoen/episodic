"""
Topic boundary analysis functionality.

This file maintains backward compatibility by importing from the new
topics.boundaries module. All functionality has been moved to:
- episodic.topics.boundaries: analyze_topic_boundary and find_transition_point_heuristic
"""

# Import everything from the new module structure for backward compatibility
from episodic.topics.boundaries import (
    analyze_topic_boundary,
    find_transition_point_heuristic
)