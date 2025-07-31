#!/usr/bin/env python3
"""
Unit test for dual-window detector.
"""

from episodic.topics.dual_window_detector import DualWindowDetector
from episodic.config import config

# Set debug mode
config.set('debug', True)

# Create detector
detector = DualWindowDetector()

# Create test messages
test_messages = [
    # Python topic
    {"role": "user", "content": "How do I read a CSV file in Python?", "short_id": "msg1"},
    {"role": "assistant", "content": "You can use pandas...", "short_id": "msg2"},
    {"role": "user", "content": "Can you show me how to handle missing values in pandas?", "short_id": "msg3"},
    {"role": "assistant", "content": "Sure, pandas provides...", "short_id": "msg4"},
    {"role": "user", "content": "What's the difference between loc and iloc?", "short_id": "msg5"},
    {"role": "assistant", "content": "loc is label-based...", "short_id": "msg6"},
    
    # Topic change to cooking
    {"role": "user", "content": "What's the secret to making perfect carbonara?", "short_id": "msg7"},
    {"role": "assistant", "content": "The key to carbonara is...", "short_id": "msg8"},
]

print("Testing Dual-Window Detector")
print("=" * 60)

# Test at different points
test_points = [
    (6, "What's the secret to making perfect carbonara?", "Should detect topic change from Python to cooking"),
]

for idx, new_message, description in test_points:
    print(f"\nTest: {description}")
    print(f"Previous messages: {idx}")
    print(f"New message: '{new_message[:50]}...'")
    
    # Get recent messages (reversed for newest-first)
    recent = test_messages[:idx]
    recent.reverse()
    
    # Detect
    changed, topic, info = detector.detect_topic_change(
        recent_messages=recent,
        new_message=new_message,
        current_topic=("Python Programming", "msg1")
    )
    
    print(f"\nResult: {'TOPIC CHANGED' if changed else 'SAME TOPIC'}")
    if info:
        if info.get('high_precision'):
            hp = info['high_precision']
            print(f"  High precision (4,1): similarity={hp['similarity']:.3f}, boundary={hp['is_boundary']}")
        if info.get('safety_net'):
            sn = info['safety_net']
            print(f"  Safety net (4,2): similarity={sn['similarity']:.3f}, boundary={sn['is_boundary']}")
        print(f"  Detection type: {info.get('detection_type', 'none')}")