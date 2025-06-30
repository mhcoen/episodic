#!/usr/bin/env python3
"""Simple test of hybrid topic detection without database."""

import sys
sys.path.insert(0, '.')

from episodic.topics_hybrid import detect_topic_change_hybrid
from episodic.config import config

# Configure
config.set("use_hybrid_topic_detection", True)
config.set("debug", True)
config.set("hybrid_topic_weights", {
    "semantic_drift": 0.6,
    "keyword_explicit": 0.25,
    "keyword_domain": 0.1,
    "message_gap": 0.025,
    "conversation_flow": 0.025
})
config.set("hybrid_topic_threshold", 0.55)
config.set("hybrid_llm_threshold", 0.3)

print("Simple Hybrid Topic Detection Test\n")

# Test 1: Normal conversation flow
print("Test 1: Normal conversation (should NOT change)")
messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "How does it work?"},
    {"role": "assistant", "content": "It works by..."},
]
changed, _, metadata = detect_topic_change_hybrid(messages, "Can you give me an example?")
print(f"Result: Changed={changed}, Score={metadata.get('score'):.2f}")
print(f"Explanation: {metadata.get('explanation')}\n")

# Test 2: Topic change with transition phrase
print("Test 2: Explicit transition (SHOULD change)")
messages = [
    {"role": "user", "content": "Tell me about Python"},
    {"role": "assistant", "content": "Python is..."},
]
changed, _, metadata = detect_topic_change_hybrid(messages, "Let's switch gears. What's the weather forecast?")
print(f"Result: Changed={changed}, Score={metadata.get('score'):.2f}")
print(f"Explanation: {metadata.get('explanation')}\n")

# Test 3: Semantic drift
print("Test 3: Large semantic drift (SHOULD change)")
messages = [
    {"role": "user", "content": "How do I bake a cake?"},
    {"role": "assistant", "content": "To bake a cake..."},
    {"role": "user", "content": "What temperature should I use?"},
    {"role": "assistant", "content": "350°F is typical..."},
]
changed, _, metadata = detect_topic_change_hybrid(messages, "Explain quantum entanglement")
print(f"Result: Changed={changed}, Score={metadata.get('score'):.2f}")
print(f"Explanation: {metadata.get('explanation')}\n")

print("All tests completed!")