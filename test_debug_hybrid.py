#!/usr/bin/env python3
"""Debug hybrid detection step by step."""

import sys
sys.path.insert(0, '.')

from episodic.topics_hybrid import HybridTopicDetector

detector = HybridTopicDetector()

# Test 1: Keywords should trigger
messages = [
    {"role": "user", "content": "Tell me about Python programming"},
    {"role": "assistant", "content": "Python is..."},
]

new_msg = "Let's change topics. What's the weather like?"
print("=== Test 1: Explicit transition ===")
print(f"Messages: {[m['content'][:30] + '...' for m in messages if m['role'] == 'user']}")
print(f"New message: '{new_msg}'")

# Manually check components
keyword_result = detector.transition_detector.detect_transition_keywords(new_msg)
print(f"\nKeyword detection:")
print(f"  Explicit: {keyword_result['explicit_transition']}")
print(f"  Found phrase: {keyword_result.get('found_phrase')}")

# Run full detection
result = detector.detect_topic_change(messages, new_msg)
print(f"\nFull detection:")
print(f"  Changed: {result[0]}")
print(f"  Score: {result[2].get('score')}")
print(f"  Signals: {result[2].get('signals')}")
print(f"  Explanation: {result[2].get('explanation')}")

# Test 2: Semantic drift
print("\n\n=== Test 2: Semantic drift ===")
messages2 = [
    {"role": "user", "content": "How do I make pasta?"},
    {"role": "assistant", "content": "To make pasta..."},
    {"role": "user", "content": "What sauce goes well with it?"},
    {"role": "assistant", "content": "For pasta sauces..."},
]

new_msg2 = "Can you explain quantum computing?"
print(f"Messages: {[m['content'][:30] + '...' for m in messages2 if m['role'] == 'user']}")
print(f"New message: '{new_msg2}'")

result2 = detector.detect_topic_change(messages2, new_msg2)
print(f"\nFull detection:")
print(f"  Changed: {result2[0]}")
print(f"  Score: {result2[2].get('score')}")
print(f"  Signals: {result2[2].get('signals')}")
print(f"  Explanation: {result2[2].get('explanation')}")