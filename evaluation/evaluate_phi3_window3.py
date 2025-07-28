#!/usr/bin/env python3
"""Evaluate phi3:mini with window_size=3 on SuperDialseg."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.metrics import SegmentationMetrics


def evaluate_superdialseg_phi3(threshold=0.7, window_size=3, num_dialogues=10):
    """Evaluate phi3:mini on SuperDialseg dataset."""
    print(f"\nEvaluating phi3:mini on SuperDialseg...")
    print(f"Window size: {window_size}, Threshold: {threshold}")
    
    # Load dataset
    loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
    conversations = loader.load_conversations(
        Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
        split='test'
    )[:num_dialogues]
    
    # Create detector with phi3:mini
    detector = OllamaInstructDetector(
        model_name="phi3:mini",  # This is the same as phi3:instruct
        threshold=threshold,
        window_size=window_size,
        verbose=False
    )
    
    metrics_calc = SegmentationMetrics()
    all_results = []
    total_time = 0
    
    for i, conv in enumerate(conversations):
        print(f"  Processing dialogue {i+1}/{len(conversations)}...", end='', flush=True)
        
        # Parse conversation
        messages, gold_boundaries = loader.parse_conversation(conv)
        
        # Detect boundaries
        start_time = time.time()
        predicted_boundaries = detector.detect_boundaries(messages)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Calculate metrics
        results = metrics_calc.calculate_exact_metrics(
            predicted_boundaries,
            gold_boundaries,
            len(messages)
        )
        
        all_results.append(results)
        print(f" F1={results['f1']:.3f} ({elapsed:.1f}s)")
    
    # Calculate averages
    avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
    avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
    avg_f1 = sum(r['f1'] for r in all_results) / len(all_results)
    avg_time = total_time / len(conversations)
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'total_time': total_time,
        'avg_time_per_dialogue': avg_time
    }


def main():
    print("Evaluating phi3:mini on SuperDialseg with Window Size 3")
    print("="*60)
    print("phi3:mini is 2.2 GB (same as phi3:instruct)")
    
    # Test different thresholds
    thresholds = [0.5, 0.6, 0.7]
    num_dialogues = 10
    
    results_table = []
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing threshold: {threshold}")
        print('='*60)
        
        try:
            results = evaluate_superdialseg_phi3(
                threshold=threshold,
                window_size=3,
                num_dialogues=num_dialogues
            )
            
            results_table.append({
                'threshold': threshold,
                'results': results
            })
            
            print(f"\nResults:")
            print(f"  Precision: {results['precision']:.3f}")
            print(f"  Recall: {results['recall']:.3f}")
            print(f"  F1: {results['f1']:.3f}")
            print(f"  Avg time/dialogue: {results['avg_time_per_dialogue']:.2f}s")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - phi3:mini on SuperDialseg (window=3)")
    print("="*60)
    print(f"\n{'Threshold':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Time/Dialog':<12}")
    print("-"*58)
    
    best_f1 = 0
    best_threshold = 0
    
    for result in results_table:
        threshold = result['threshold']
        r = result['results']
        
        print(f"{threshold:<10.1f} {r['precision']:<12.3f} {r['recall']:<12.3f} {r['f1']:<12.3f} {r['avg_time_per_dialogue']:<12.2f}s")
        
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_threshold = threshold
    
    print(f"\nBest: F1={best_f1:.3f} with threshold={best_threshold}")
    
    print("\n" + "="*60)
    print("COMPARISON (all using window_size=3)")
    print("="*60)
    print("Sentence-BERT: F1=0.571")
    print("Sliding Window: F1=0.560")
    print(f"phi3:mini: F1={best_f1:.3f}")
    print("\nNote: phi3 is 2.2GB (not really 'small' compared to qwen2:0.5b at 352MB)")


if __name__ == "__main__":
    main()