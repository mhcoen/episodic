#!/usr/bin/env python3
"""Quick test of instruct models on real datasets."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.metrics import SegmentationMetrics

# Load a few examples from SuperDialseg
superseg_path = "/Users/mhcoen/proj/episodic/datasets/superseg/segmentation_file_test.json"
tiage_path = "/Users/mhcoen/proj/episodic/datasets/tiage/segmentation_file_test.json"


def load_dialogues(file_path, num_dialogues=5):
    """Load a few dialogues from dataset."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    dialogues = []
    
    # Handle different dataset formats
    if 'dial_data' in data:  # SuperDialseg/TIAGE format
        dataset_key = list(data['dial_data'].keys())[0]
        raw_dialogues = data['dial_data'][dataset_key][:num_dialogues]
        
        for dialogue in raw_dialogues:
            messages = []
            boundaries = []
            prev_topic_id = None
            
            for i, turn in enumerate(dialogue['turns']):
                messages.append({
                    'role': turn['role'],
                    'content': turn['utterance']
                })
                
                # Detect boundaries from topic_id changes
                if 'topic_id' in turn:
                    current_topic_id = turn['topic_id']
                    if prev_topic_id is not None and current_topic_id != prev_topic_id:
                        boundaries.append(i - 1)
                    prev_topic_id = current_topic_id
            
            dialogues.append((messages, boundaries))
    
    return dialogues


def test_on_dataset(dataset_name, file_path, detector):
    """Test detector on a dataset."""
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name}")
    print('='*60)
    
    dialogues = load_dialogues(file_path, num_dialogues=5)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    metrics_calc = SegmentationMetrics()
    
    for i, (messages, gold_boundaries) in enumerate(dialogues):
        print(f"\nDialogue {i+1}: {len(messages)} messages, boundaries at {gold_boundaries}")
        
        # Detect boundaries
        start_time = time.time()
        predicted_boundaries = detector.detect_boundaries(messages)
        detect_time = time.time() - start_time
        
        # Calculate metrics
        results = metrics_calc.calculate_exact_metrics(
            predicted_boundaries,
            gold_boundaries,
            len(messages)
        )
        
        print(f"Predicted: {predicted_boundaries}")
        print(f"F1: {results['f1']:.3f}, Time: {detect_time:.2f}s")
        
        total_precision += results['precision']
        total_recall += results['recall']
        total_f1 += results['f1']
    
    # Average metrics
    n = len(dialogues)
    print(f"\nAverage Results for {dataset_name}:")
    print(f"Precision: {total_precision/n:.3f}")
    print(f"Recall: {total_recall/n:.3f}")
    print(f"F1: {total_f1/n:.3f}")
    
    return total_f1/n


def main():
    print("Quick Test of Instruct Models on Real Datasets")
    print("="*60)
    
    # Test mistral:instruct with different thresholds
    thresholds = [0.5, 0.6, 0.7]
    
    best_results = {}
    
    for threshold in thresholds:
        print(f"\n\nTesting mistral:instruct with threshold {threshold}")
        print("-"*60)
        
        detector = OllamaInstructDetector(
            model_name="mistral:instruct",
            threshold=threshold,
            window_size=1,
            verbose=False
        )
        
        # Test on SuperDialseg
        superseg_f1 = test_on_dataset("SuperDialseg", superseg_path, detector)
        
        # Test on TIAGE
        tiage_f1 = test_on_dataset("TIAGE", tiage_path, detector)
        
        best_results[f"t{threshold}"] = {
            'SuperDialseg': superseg_f1,
            'TIAGE': tiage_f1
        }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for config, results in best_results.items():
        print(f"\n{config}:")
        print(f"  SuperDialseg F1: {results['SuperDialseg']:.3f}")
        print(f"  TIAGE F1: {results['TIAGE']:.3f}")
    
    # Compare with other methods
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER METHODS")
    print("="*60)
    print("Best F1 scores from previous evaluations:")
    print("SuperDialseg: Sentence-BERT=0.571, Sliding Window=0.560")
    print("TIAGE: Sentence-BERT=0.222, Sliding Window=0.219")
    print("\nInstruct model results above")


if __name__ == "__main__":
    main()