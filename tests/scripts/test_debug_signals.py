#!/usr/bin/env python3
"""Debug signal details."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.topics_hybrid import HybridTopicDetector
from episodic.config import config

# Configure
config.set("hybrid_topic_weights", {
    "semantic_drift": 0.6,
    "keyword_explicit": 0.25,
    "keyword_domain": 0.1,
    "message_gap": 0.025,
    "conversation_flow": 0.025
})

detector = HybridTopicDetector()

# Test explicit transition
messages = [
    {"role": "user", "content": "Tell me about Python"},
    {"role": "assistant", "content": "Python is..."},
]

new_msg = "Let's switch gears. What's the weather forecast?"
print(f"Testing: '{new_msg}'")

# Get keyword results first
keyword_results = detector.transition_detector.detect_transition_keywords(new_msg)
print(f"\nKeyword detection:")
print(f"  Explicit: {keyword_results['explicit_transition']}")
print(f"  Found phrase: {keyword_results.get('found_phrase')}")

# Run full detection
changed, _, metadata = detector.detect_topic_change(messages, new_msg)
print(f"\nFull detection:")
print(f"  Changed: {changed}")
print(f"  Score: {metadata.get('score')}")
print(f"  Signals: {metadata.get('signals')}")
print(f"  Explanation: {metadata.get('explanation')}")