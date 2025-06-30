#!/usr/bin/env python3
"""Check actual runtime configuration values."""

import sys
sys.path.insert(0, '.')

from episodic.config import config
import json

print("Runtime Configuration Check")
print("=" * 60)

# Check config object
print("\nFrom config object:")
print(f"min_messages_before_topic_change: {config.get('min_messages_before_topic_change', 'NOT SET')}")
print(f"topic_detection_model: {config.get('topic_detection_model', 'NOT SET')}")
print(f"use_hybrid_topic_detection: {config.get('use_hybrid_topic_detection', False)}")
print(f"hybrid_topic_threshold: {config.get('hybrid_topic_threshold', 'NOT SET')}")
print(f"hybrid_topic_weights: {config.get('hybrid_topic_weights', 'NOT SET')}")

# Show all config values
print("\nAll configuration values:")
for key, value in sorted(config.config.items()):
    if isinstance(value, dict):
        print(f"{key}: {json.dumps(value, indent=2)}")
    else:
        print(f"{key}: {value}")

# Look for where min_messages_before_topic_change might be defaulted
print("\n\nChecking topic detection defaults:")
from episodic.topics import topic_manager
print(f"TopicManager min_messages default: {getattr(topic_manager, 'min_messages_before_topic_change', 'NOT FOUND')}")

# Check topic detection behavior
print("\n\nTopic Detection Threshold Behavior:")
print("First topic: Always created after 3+ user messages")
print("Second topic: Uses min_messages/2 threshold")
print("Third+ topics: Uses full min_messages threshold")
print("\nThis means with min_messages=8:")
print("- First topic: 3+ messages")
print("- Second topic: 4+ messages") 
print("- Third+ topics: 8+ messages")