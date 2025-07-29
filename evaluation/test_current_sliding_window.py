#!/usr/bin/env python3
"""
Test the current sliding window detection system on our dataset.
This evaluates the rule-based system that's actually deployed in Episodic.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import List, Tuple
import os

# Load the sentence transformer model used by Episodic
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.9  # Default from Episodic
WINDOW_SIZE = 3  # Default sliding window size

def load_dataset(filepath: str) -> Tuple[List[List[str]], List[int]]:
    """Load the conversation dataset from JSONL format."""
    conversations = []
    labels = []
    
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            # Extract messages from the conversation
            messages = []
            for turn in item['messages']:
                messages.append(turn['content'])
            conversations.append(messages)
            # The label is 1 for boundary, 0 for no boundary
            labels.append(item['boundary_position'] if 'boundary_position' in item else item.get('label', 0))
    
    return conversations, labels

def get_window_embedding(messages: List[str], model: SentenceTransformer) -> np.ndarray:
    """Get embedding for a window of messages."""
    # Concatenate messages in the window
    window_text = " ".join(messages)
    # Get embedding
    embedding = model.encode(window_text, convert_to_numpy=True)
    return embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def detect_boundaries_sliding_window(
    messages: List[str], 
    model: SentenceTransformer,
    window_size: int = WINDOW_SIZE,
    threshold: float = SIMILARITY_THRESHOLD,
    min_messages: int = 8
) -> List[int]:
    """
    Detect topic boundaries using sliding window approach.
    Returns list of indices where boundaries are detected.
    """
    boundaries = []
    
    # Need at least 2 * window_size messages to compare windows
    if len(messages) < 2 * window_size:
        return boundaries
    
    # Skip detection for first min_messages
    start_index = max(2 * window_size, min_messages)
    
    for i in range(start_index, len(messages) - window_size + 1):
        # Get two windows
        window1 = messages[i - window_size:i]
        window2 = messages[i:i + window_size]
        
        # Get embeddings
        embed1 = get_window_embedding(window1, model)
        embed2 = get_window_embedding(window2, model)
        
        # Calculate similarity
        similarity = cosine_similarity(embed1, embed2)
        
        # Check if similarity is below threshold
        if similarity < threshold:
            boundaries.append(i)
    
    return boundaries

def evaluate_sliding_window(test_file: str):
    """Evaluate the sliding window detection on test data."""
    print("Loading test dataset...")
    conversations, true_labels = load_dataset(test_file)
    
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    all_predictions = []
    all_true_labels = []
    
    print(f"\nEvaluating sliding window detection:")
    print(f"- Window size: {WINDOW_SIZE}")
    print(f"- Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"- Min messages before detection: 8")
    
    # Process each conversation example
    for conv_idx, (messages, true_label) in enumerate(zip(conversations, true_labels)):
        if conv_idx % 100 == 0:
            print(f"Processing example {conv_idx}/{len(conversations)}...")
        
        # For JSONL format, each example is a single window with a label
        # We need to check if there's a boundary in this window
        # The boundary_position indicates where in the window the boundary is
        
        # Check if we have enough messages
        if len(messages) < 2 * WINDOW_SIZE:
            continue
            
        # Try to detect a boundary in the middle of the conversation
        middle_idx = len(messages) // 2
        
        # Get windows around the middle
        if middle_idx >= WINDOW_SIZE and middle_idx + WINDOW_SIZE <= len(messages):
            window1 = messages[middle_idx - WINDOW_SIZE:middle_idx]
            window2 = messages[middle_idx:middle_idx + WINDOW_SIZE]
            
            # Get embeddings
            embed1 = get_window_embedding(window1, model)
            embed2 = get_window_embedding(window2, model)
            
            # Calculate similarity
            similarity = cosine_similarity(embed1, embed2)
            
            # Predict boundary if similarity is below threshold
            predicted = 1 if similarity < SIMILARITY_THRESHOLD else 0
            
            all_predictions.append(predicted)
            all_true_labels.append(true_label)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, 
        all_predictions, 
        average='binary',
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    print("\n" + "="*50)
    print("SLIDING WINDOW DETECTION RESULTS")
    print("="*50)
    print(f"Total samples evaluated: {len(all_predictions)}")
    print(f"True boundaries: {sum(all_true_labels)}")
    print(f"Predicted boundaries: {sum(all_predictions)}")
    print(f"\nF1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:6d}  FP: {cm[0,1]:6d}")
    print(f"FN: {cm[1,0]:6d}  TP: {cm[1,1]:6d}")
    
    # Test different thresholds
    print("\n" + "="*50)
    print("TESTING DIFFERENT THRESHOLDS")
    print("="*50)
    
    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    for threshold in thresholds:
        all_predictions = []
        all_true_labels = []
        
        for conv_idx, (messages, true_label) in enumerate(zip(conversations, true_labels)):
            if len(messages) < 2 * WINDOW_SIZE:
                continue
                
            middle_idx = len(messages) // 2
            
            if middle_idx >= WINDOW_SIZE and middle_idx + WINDOW_SIZE <= len(messages):
                window1 = messages[middle_idx - WINDOW_SIZE:middle_idx]
                window2 = messages[middle_idx:middle_idx + WINDOW_SIZE]
                
                embed1 = get_window_embedding(window1, model)
                embed2 = get_window_embedding(window2, model)
                
                similarity = cosine_similarity(embed1, embed2)
                predicted = 1 if similarity < threshold else 0
                
                all_predictions.append(predicted)
                all_true_labels.append(true_label)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, 
            all_predictions, 
            average='binary',
            zero_division=0
        )
        
        print(f"\nThreshold: {threshold}")
        print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  Predicted boundaries: {sum(all_predictions)}")

if __name__ == "__main__":
    # Test on the same test set used for ML models
    test_file = "/Users/mhcoen/proj/episodic/evaluation/finetuning_data/test_all_datasets.jsonl"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        print("Please ensure the test dataset is available.")
    else:
        evaluate_sliding_window(test_file)