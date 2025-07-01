#!/usr/bin/env python3
"""Debug keyword detection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.topics_hybrid import TransitionDetector

detector = TransitionDetector()

test_messages = [
    "Let's change topics. What's the weather like?",
    "Let me ask about something else",
    "By the way, how are you?",
    "Just a normal message",
    "Can you explain neural networks?",
]

for msg in test_messages:
    result = detector.detect_transition_keywords(msg)
    print(f"\nMessage: '{msg}'")
    print(f"  Explicit score: {result['explicit_transition']}")
    print(f"  Found phrase: {result.get('found_phrase')}")
    print(f"  Domain shift: {result['domain_shift']}")
    print(f"  Dominant domain: {result['dominant_domain']}")