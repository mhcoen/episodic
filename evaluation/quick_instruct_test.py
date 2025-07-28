#!/usr/bin/env python3
"""Quick single dialogue test of instruct model."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.metrics import SegmentationMetrics

# Load one dialogue from SuperDialseg
with open("/Users/mhcoen/proj/episodic/datasets/superseg/segmentation_file_test.json", 'r') as f:
    data = json.load(f)

# Get first dialogue
dialogue = data['dial_data']['superseg-v2'][0]

# Extract messages and boundaries
messages = []
boundaries = []
prev_topic_id = None

for i, turn in enumerate(dialogue['turns']):
    messages.append({
        'role': turn['role'],
        'content': turn['utterance']
    })
    
    if 'topic_id' in turn:
        current_topic_id = turn['topic_id']
        if prev_topic_id is not None and current_topic_id != prev_topic_id:
            boundaries.append(i - 1)
        prev_topic_id = current_topic_id

print("Test Dialogue from SuperDialseg")
print(f"Length: {len(messages)} messages")
print(f"Gold boundaries: {boundaries}")
print(f"\nFirst few messages:")
for i in range(min(4, len(messages))):
    print(f"{i}: [{messages[i]['role']}] {messages[i]['content'][:60]}...")

# Test with mistral:instruct
print("\n" + "="*60)
print("Testing mistral:instruct")
print("="*60)

detector = OllamaInstructDetector(
    model_name="mistral:instruct",
    threshold=0.6,
    window_size=1,
    verbose=True  # Show drift scores
)

predicted = detector.detect_boundaries(messages)

# Calculate metrics
metrics = SegmentationMetrics()
results = metrics.calculate_exact_metrics(
    predicted,
    boundaries,
    len(messages)
)

print(f"\nPredicted boundaries: {predicted}")
print(f"\nMetrics:")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")

# Also test with windowed metrics
windowed_results = metrics.calculate_windowed_metrics(
    predicted,
    boundaries,
    len(messages),
    window=3
)

print(f"\nWith tolerance window=3:")
print(f"Precision: {windowed_results['precision_w3']:.3f}")
print(f"Recall: {windowed_results['recall_w3']:.3f}")
print(f"F1 Score: {windowed_results['f1_w3']:.3f}")