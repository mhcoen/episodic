#!/usr/bin/env python3
"""Evaluate small models with window_size=3 on SuperDialseg - Fixed version."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.metrics import SegmentationMetrics


def evaluate_superdialseg(model_name, threshold, window_size=3, num_dialogues=10):
    """Evaluate on SuperDialseg dataset."""
    print(f"\nEvaluating {model_name} on SuperDialseg...")
    print(f"Window size: {window_size}, Threshold: {threshold}")
    
    # Load dataset
    loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
    conversations = loader.load_conversations(
        Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
        split='test'
    )[:num_dialogues]
    
    # Create detector
    detector = OllamaInstructDetector(
        model_name=model_name,
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
    print("Evaluating Small Models on SuperDialseg with Window Size 3")
    print("="*60)
    
    # Test configurations
    models = [
        ("qwen2:0.5b", 0.5, 352),
        ("qwen2:0.5b", 0.3, 352),
        ("tinyllama:latest", 0.7, 637),
        ("qwen2:1.5b", 0.6, 934),
    ]
    
    # Number of dialogues to test
    num_dialogues = 5  # Start small
    
    results_table = []
    
    for model_name, threshold, size_mb in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({size_mb} MB)")
        print('='*60)
        
        try:
            results = evaluate_superdialseg(
                model_name, 
                threshold, 
                window_size=3,
                num_dialogues=num_dialogues
            )
            
            results_table.append({
                'model': model_name,
                'threshold': threshold,
                'size_mb': size_mb,
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
    print("SUMMARY - SuperDialseg with Window Size 3")
    print("="*60)
    print(f"\n{'Model':<25} {'Threshold':<10} {'F1 Score':<12} {'Time/Dialog':<12}")
    print("-"*59)
    
    for result in results_table:
        model = f"{result['model']} ({result['size_mb']} MB)"
        threshold = result['threshold']
        f1 = result['results']['f1']
        time_per = result['results']['avg_time_per_dialogue']
        
        print(f"{model:<25} {threshold:<10.1f} {f1:<12.3f} {time_per:<12.2f}s")
    
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER METHODS (window_size=3)")
    print("="*60)
    print("Sentence-BERT: F1=0.571")
    print("Sliding Window: F1=0.560")
    print("\nKey Question: Do small instruct models perform well with window_size=3?")


if __name__ == "__main__":
    main()