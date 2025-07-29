#!/usr/bin/env python3
"""
Test the current sliding window detection on pre-processed test data.
This simulates how the sliding window would perform on pairs of messages.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from typing import List, Tuple
import re

# Configuration matching Episodic's defaults
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.9

def parse_message_pair(input_text: str) -> Tuple[str, str]:
    """Parse the two messages from the input text."""
    # Extract messages using regex
    match = re.match(r"Message 1: (.+?)\nMessage 2: (.+)", input_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def evaluate_sliding_window(test_file: str, threshold: float = SIMILARITY_THRESHOLD):
    """Evaluate sliding window approach on message pairs."""
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Loading test data...")
    predictions = []
    true_labels = []
    similarities = []
    
    with open(test_file, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print(f"Processing sample {idx}...")
                
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
            similarities.append(similarity)
            
            # Predict: if similarity < threshold, it's a topic change
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
    
    cm = confusion_matrix(true_labels, predictions)
    
    print("\n" + "="*60)
    print(f"SLIDING WINDOW DETECTION RESULTS (threshold={threshold})")
    print("="*60)
    print(f"Total samples: {len(predictions)}")
    print(f"True boundaries: {sum(true_labels)}")
    print(f"Predicted boundaries: {sum(predictions)}")
    print(f"\nF1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              No    Yes")
    print(f"Actual No  {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"       Yes {cm[1,0]:5d} {cm[1,1]:5d}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=['Same Topic', 'Topic Change']))
    
    # Analyze similarity distribution
    similarities = np.array(similarities)
    boundary_sims = [s for s, l in zip(similarities, true_labels) if l == 1]
    non_boundary_sims = [s for s, l in zip(similarities, true_labels) if l == 0]
    
    print("\nSimilarity Statistics:")
    print(f"Overall: mean={np.mean(similarities):.3f}, std={np.std(similarities):.3f}")
    print(f"Topic boundaries: mean={np.mean(boundary_sims):.3f}, std={np.std(boundary_sims):.3f}")
    print(f"Same topic: mean={np.mean(non_boundary_sims):.3f}, std={np.std(non_boundary_sims):.3f}")
    
    return {
        'threshold': threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': predictions,
        'true_labels': true_labels,
        'similarities': similarities
    }

def test_multiple_thresholds(test_file: str):
    """Test different similarity thresholds."""
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results = []
    
    print("\n" + "="*60)
    print("TESTING MULTIPLE THRESHOLDS")
    print("="*60)
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        result = evaluate_sliding_window(test_file, threshold)
        results.append(result)
        print(f"F1: {result['f1']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}")
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['f1'])
    print(f"\nBest threshold: {best_result['threshold']} with F1={best_result['f1']:.4f}")
    
    return results

if __name__ == "__main__":
    test_file = "/Users/mhcoen/proj/episodic/evaluation/finetuning_data/test_all_datasets.jsonl"
    
    # Test with default threshold
    print("Testing with Episodic's default threshold (0.9)...")
    default_result = evaluate_sliding_window(test_file)
    
    # Test multiple thresholds
    print("\n" + "="*80)
    all_results = test_multiple_thresholds(test_file)