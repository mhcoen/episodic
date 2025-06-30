#!/usr/bin/env python3
"""Test manual topic indexing."""

import sys
sys.path.insert(0, '.')

from episodic.config import config
from episodic.commands.index_topics import index_topics

print("Testing Manual Topic Indexing")
print("=" * 60)

# First, disable automatic topic detection
print("\n1. Disabling automatic topic detection...")
config.set("automatic_topic_detection", False)
print(f"   automatic_topic_detection = {config.get('automatic_topic_detection')}")

# Test with different window sizes
print("\n2. Testing sliding window analysis...")
print("\nWindow size = 3 (comparing 3 messages at a time):")
print("-" * 40)
index_topics(window_size=3, apply=False, verbose=False)

print("\n\nWindow size = 5 (comparing 5 messages at a time):")
print("-" * 40)
index_topics(window_size=5, apply=False, verbose=False)

print("\n" + "=" * 60)
print("Configuration Summary:")
print(f"- Automatic topic detection: {config.get('automatic_topic_detection', True)}")
print(f"- Hybrid detection enabled: {config.get('use_hybrid_topic_detection', False)}")
print(f"- Topic threshold: {config.get('hybrid_topic_threshold', 0.55)}")
print("\nUse '/index n' command to manually detect topics with window size n")
print("Add --verbose to see all scores, --apply to create the topics")