#!/usr/bin/env python3
"""Test edge cases and error handling."""

import sys
sys.path.insert(0, '.')

from episodic.topics_hybrid import detect_topic_change_hybrid
from episodic.config import config

# Configure
config.set("use_hybrid_topic_detection", True)
config.set("hybrid_topic_threshold", 0.55)

print("Edge Case Testing\n")

# Test 1: Empty history
print("Test 1: Empty message history")
try:
    changed, _, metadata = detect_topic_change_hybrid([], "Hello world")
    print(f"Result: Changed={changed}, Method={metadata.get('method')}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: No user messages
print("\nTest 2: No user messages in history")
messages = [
    {"role": "assistant", "content": "Welcome!"},
    {"role": "system", "content": "You are helpful"},
]
try:
    changed, _, metadata = detect_topic_change_hybrid(messages, "Hello")
    print(f"Result: Changed={changed}, Method={metadata.get('method')}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Very long message
print("\nTest 3: Very long message")
messages = [
    {"role": "user", "content": "Tell me about AI"},
    {"role": "assistant", "content": "AI is..."},
]
long_msg = "Let me ask about " + "something " * 100 + "different"
try:
    changed, _, metadata = detect_topic_change_hybrid(messages, long_msg)
    print(f"Result: Changed={changed}, Score={metadata.get('score'):.2f}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Special characters
print("\nTest 4: Special characters and emojis")
messages = [
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "4"},
]
special_msg = "BTW 🤔 what about weather? ☀️"
try:
    changed, _, metadata = detect_topic_change_hybrid(messages, special_msg)
    print(f"Result: Changed={changed}, Score={metadata.get('score'):.2f}")
    print(f"Found phrase: {metadata.get('transition_phrase')}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: None/empty content
print("\nTest 5: Empty new message")
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]
try:
    changed, _, metadata = detect_topic_change_hybrid(messages, "")
    print(f"Result: Changed={changed}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nEdge case testing completed!")