#!/usr/bin/env python3
"""
Quick test of sliding window detection on a subset of test data.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import re

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.9
MAX_SAMPLES = 200  # Process only first 200 samples for quick results

def parse_message_pair(input_text: str):
    """Parse the two messages from the input text."""
    match = re.match(r"Message 1: (.+?)\nMessage 2: (.+)", input_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Processing first {MAX_SAMPLES} samples...")

# Test different thresholds
thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]

for threshold in thresholds:
    predictions = []
    true_labels = []
    
    with open("/Users/mhcoen/proj/episodic/evaluation/finetuning_data/test_all_datasets.jsonl", 'r') as f:
        for idx, line in enumerate(f):
            if idx >= MAX_SAMPLES:
                break
                
            data = json.loads(line.strip())
            
            # Parse messages
            msg1, msg2 = parse_message_pair(data['input'])
            if msg1 is None or msg2 is None:
                continue
            
            # Get embeddings
            embed1 = model.encode(msg1, convert_to_numpy=True)
            embed2 = model.encode(msg2, convert_to_numpy=True)
            
            # Calculate similarity
            similarity = cosine_similarity(embed1, embed2)
            
            # Predict
            predicted = 1 if similarity < threshold else 0
            predictions.append(predicted)
            
            # True label
            true_label = int(data['output'])
            true_labels.append(true_label)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average='binary',
        zero_division=0
    )
    
    print(f"\nThreshold: {threshold}")
    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Predicted boundaries: {sum(predictions)}/{len(predictions)}")

print("\nNote: Results based on first", MAX_SAMPLES, "samples only")