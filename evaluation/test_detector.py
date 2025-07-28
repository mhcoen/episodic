#!/usr/bin/env python3
"""Quick test of detector on sample data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.detector_adapters import create_detector

# Load a sample conversation
loader = SuperDialsegLoader()
dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/superseg")

# Load test data
conversations = loader.load_conversations(dataset_path, 'test')
conv = conversations[0]  # First conversation

# Parse it
messages, gold_boundaries = loader.parse_conversation(conv)

print(f"Conversation ID: {conv.get('dial_id', 'unknown')}")
print(f"Number of messages: {len(messages)}")
print(f"Gold boundaries: {gold_boundaries}")
print("\nFirst few messages:")
for i, msg in enumerate(messages[:6]):
    print(f"  {i}: [{msg['role']}] {msg['content'][:60]}...")

# Test sliding window detector
print("\n\nTesting sliding window detector:")
detector = create_detector('sliding_window', window_size=3, threshold=0.5)
predicted = detector.detect_boundaries(messages)
print(f"Predicted boundaries (threshold=0.5): {predicted}")

# Try with different threshold
detector2 = create_detector('sliding_window', window_size=3, threshold=0.3)
predicted2 = detector2.detect_boundaries(messages)
print(f"Predicted boundaries (threshold=0.3): {predicted2}")

# Test keywords detector
print("\n\nTesting keywords detector:")
keyword_detector = create_detector('keywords', threshold=0.3)
keyword_predicted = keyword_detector.detect_boundaries(messages)
print(f"Predicted boundaries (keywords): {keyword_predicted}")

# Check what the detector is seeing
print("\n\nChecking detector internals:")
from episodic.topics.realtime_windows import RealtimeWindowDetector
from episodic.ml.drift import ConversationalDrift

# Create drift calculator directly
drift_calc = ConversationalDrift()

# Test drift between first and last user messages
user_msgs = [m for m in messages if m['role'] == 'user']
if len(user_msgs) >= 2:
    node1 = {'content': user_msgs[0]['content']}
    node2 = {'content': user_msgs[-1]['content']}
    drift = drift_calc.calculate_drift(node1, node2, text_field='content')
    print(f"Drift between first and last user message: {drift:.3f}")
    print(f"First user msg: {user_msgs[0]['content'][:60]}...")
    print(f"Last user msg: {user_msgs[-1]['content'][:60]}...")