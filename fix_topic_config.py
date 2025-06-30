#!/usr/bin/env python3
"""Fix topic detection configuration."""

import sys
sys.path.insert(0, '.')

from episodic.config import config

print("Fixing Topic Detection Configuration")
print("=" * 60)

print(f"\nCurrent min_messages_before_topic_change: {config.get('min_messages_before_topic_change', 'NOT SET')}")

# Set it to the proper value
config.set('min_messages_before_topic_change', 8)

print(f"Updated min_messages_before_topic_change: {config.get('min_messages_before_topic_change')}")

# Also check if hybrid detection is causing issues
if config.get('use_hybrid_topic_detection', False):
    print("\nHybrid topic detection is ENABLED")
    print("Current weights favor semantic drift (60%), which can cause false positives")
    print("\nRecommended: Either disable hybrid detection or adjust weights")
    print("To disable: /set use_hybrid_topic_detection false")
    print("To adjust weights for less sensitivity:")
    print('/set hybrid_topic_weights {"semantic_drift": 0.3, "keyword_explicit": 0.4, "keyword_domain": 0.2, "message_gap": 0.05, "conversation_flow": 0.05}')
    print("/set hybrid_topic_threshold 0.7")
else:
    print("\nHybrid topic detection is DISABLED (using standard LLM detection)")

print("\n✅ Configuration updated! Future conversations will use the proper threshold.")
print("\nNote: This won't fix existing topics, but will prevent the issue going forward.")