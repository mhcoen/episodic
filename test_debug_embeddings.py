#!/usr/bin/env python3
"""Debug embedding calculations."""

import sys
sys.path.insert(0, '.')

from episodic.ml.drift import ConversationalDrift

# Create drift calculator
drift_calc = ConversationalDrift(
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    distance_algorithm="cosine",
    peak_strategy="threshold",
    threshold=0.35
)

# Test similar messages
node1 = {"message": "How do I make pasta?"}
node2 = {"message": "What sauce goes well with pasta?"}
drift1 = drift_calc.calculate_drift(node1, node2)
print(f"Similar messages (pasta): {drift1:.3f}")

# Test different messages
node3 = {"message": "How do I make pasta?"}
node4 = {"message": "Can you explain quantum computing?"}
drift2 = drift_calc.calculate_drift(node3, node4)
print(f"Different messages (pasta->quantum): {drift2:.3f}")

# Test very similar
node5 = {"message": "What's the weather like?"}
node6 = {"message": "How's the weather today?"}
drift3 = drift_calc.calculate_drift(node5, node6)
print(f"Very similar (weather): {drift3:.3f}")

# Test sequence
nodes = [
    {"message": "Tell me about machine learning"},
    {"message": "What is deep learning?"},
    {"message": "How do neural networks work?"},
    {"message": "Let's talk about cooking instead"},
    {"message": "What's a good pasta recipe?"},
]

print("\nSequence drift:")
for i in range(len(nodes) - 1):
    drift = drift_calc.calculate_drift(nodes[i], nodes[i+1])
    print(f"  '{nodes[i]['message'][:30]}...' -> '{nodes[i+1]['message'][:30]}...': {drift:.3f}")