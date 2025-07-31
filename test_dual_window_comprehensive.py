#!/usr/bin/env python3
"""
Comprehensive test of dual-window detector with multiple topics.
"""

from episodic.topics.dual_window_detector import DualWindowDetector
from episodic.config import config

# Set configuration
config.set('debug', False)  # Less verbose
config.set('dual_window_high_precision_threshold', 0.2)
config.set('dual_window_safety_net_threshold', 0.25)

# Create detector
detector = DualWindowDetector()

# Test conversations from populate_database.txt
conversations = [
    # Topic 1: Python Programming (5 messages)
    ("user", "How do I read a CSV file in Python?"),
    ("assistant", "You can use pandas to read CSV files..."),
    ("user", "Can you show me how to handle missing values in pandas?"),
    ("assistant", "Sure, pandas provides several methods..."),
    ("user", "What's the difference between loc and iloc?"),
    ("assistant", "loc is label-based while iloc is integer position-based..."),
    ("user", "How do I merge two dataframes?"),
    ("assistant", "You can use pd.merge() or df.merge()..."),
    ("user", "Can you explain list comprehensions with an example?"),
    ("assistant", "List comprehensions provide a concise way..."),
    
    # Topic 2: Cooking Italian Food (4 messages)
    ("user", "What's the secret to making perfect carbonara?"),
    ("assistant", "The key to perfect carbonara is..."),
    ("user", "How much pasta water should I save for the sauce?"),
    ("assistant", "Reserve about 1 cup of pasta water..."),
    ("user", "What type of cheese is best - pecorino or parmesan?"),
    ("assistant", "Traditionally, pecorino romano is used..."),
    ("user", "Should I add cream to carbonara?"),
    ("assistant", "No, authentic carbonara never contains cream..."),
    
    # Topic 3: Machine Learning Basics (6 messages)
    ("user", "What's the difference between supervised and unsupervised learning?"),
    ("assistant", "Supervised learning uses labeled data..."),
    ("user", "Can you explain what overfitting means?"),
    ("assistant", "Overfitting occurs when a model..."),
    ("user", "How do I split data into training and test sets?"),
    ("assistant", "You can use train_test_split from sklearn..."),
    ("user", "What's cross-validation and why is it important?"),
    ("assistant", "Cross-validation is a technique..."),
    ("user", "Which algorithm should I use for classification?"),
    ("assistant", "The choice depends on your data..."),
    ("user", "How do I evaluate my model's performance?"),
    ("assistant", "Use metrics like accuracy, precision, recall..."),
]

print("Comprehensive Dual-Window Detection Test")
print("=" * 60)

# Process messages
messages = []
current_topic = None
topic_boundaries = []

for i, (role, content) in enumerate(conversations):
    # Only test on user messages
    if role == "user" and len(messages) >= 5:
        # Get recent messages (newest-first)
        recent = messages.copy()
        recent.reverse()
        
        # Detect topic change
        changed, _, info = detector.detect_topic_change(
            recent_messages=recent,
            new_message=content,
            current_topic=current_topic
        )
        
        if changed:
            topic_boundaries.append({
                'index': i,
                'message': content,
                'detection_type': info.get('detection_type'),
                'high_precision_sim': info['high_precision']['similarity'] if info.get('high_precision') else None,
                'safety_net_sim': info['safety_net']['similarity'] if info.get('safety_net') else None
            })
            print(f"\n🔄 Topic boundary detected at message {i+1}:")
            print(f"   Message: '{content[:50]}...'")
            print(f"   Detection: {info.get('detection_type')}")
            if info.get('high_precision'):
                print(f"   High precision similarity: {info['high_precision']['similarity']:.3f}")
            if info.get('safety_net'):
                print(f"   Safety net similarity: {info['safety_net']['similarity']:.3f}")
    
    # Add message to history
    messages.append({
        'role': role,
        'content': content,
        'short_id': f'msg{i+1}'
    })

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nTotal messages: {len(conversations)}")
print(f"Topic boundaries detected: {len(topic_boundaries)}")

print("\nExpected boundaries:")
print("  1. Message 11: 'What's the secret to making perfect carbonara?' (Python → Cooking)")
print("  2. Message 19: 'What's the difference between supervised and unsupervised learning?' (Cooking → ML)")

print("\nActual boundaries:")
for i, boundary in enumerate(topic_boundaries):
    print(f"  {i+1}. Message {boundary['index']+1}: '{boundary['message'][:50]}...'")
    print(f"     Detection type: {boundary['detection_type']}")

# Evaluate accuracy
expected_boundaries = [10, 18]  # 0-indexed
detected_boundaries = [b['index'] for b in topic_boundaries]

print(f"\nAccuracy evaluation:")
print(f"  Expected: {expected_boundaries}")
print(f"  Detected: {detected_boundaries}")

# Check for correct detections
correct = 0
for expected in expected_boundaries:
    if expected in detected_boundaries:
        correct += 1
        print(f"  ✓ Correctly detected boundary at message {expected+1}")
    else:
        print(f"  ✗ Missed boundary at message {expected+1}")

# Check for false positives
for detected in detected_boundaries:
    if detected not in expected_boundaries:
        print(f"  ⚠️  False positive at message {detected+1}")

print(f"\nPrecision: {correct}/{len(detected_boundaries)} = {correct/len(detected_boundaries)*100:.1f}%" if detected_boundaries else "N/A")
print(f"Recall: {correct}/{len(expected_boundaries)} = {correct/len(expected_boundaries)*100:.1f}%")