#!/usr/bin/env python3
"""Evaluate phi3:mini with robust parsing on SuperDialseg."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector_robust import RobustOllamaInstructDetector
from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.metrics import SegmentationMetrics


def evaluate_superdialseg_phi3_robust(threshold=0.7, window_size=3, num_dialogues=10):
    """Evaluate phi3:mini on SuperDialseg dataset with robust parsing."""
    print(f"\nEvaluating phi3:mini (robust) on SuperDialseg...")
    print(f"Window size: {window_size}, Threshold: {threshold}")
    
    # Load dataset
    loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
    conversations = loader.load_conversations(
        Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
        split='test'
    )[:num_dialogues]
    
    # Create detector with phi3:mini and robust parsing
    detector = RobustOllamaInstructDetector(
        model_name="phi3:mini",
        threshold=threshold,
        window_size=window_size,
        verbose=True  # Enable verbose to see what's happening
    )
    
    metrics_calc = SegmentationMetrics()
    all_results = []
    total_time = 0
    
    for i, conv in enumerate(conversations[:3]):  # Test on just 3 first
        print(f"\n{'='*60}")
        print(f"Processing dialogue {i+1}/3...")
        
        # Parse conversation
        messages, gold_boundaries = loader.parse_conversation(conv)
        print(f"Messages: {len(messages)}, Gold boundaries: {gold_boundaries}")
        
        # Detect boundaries
        start_time = time.time()
        predicted_boundaries = detector.detect_boundaries(messages)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        print(f"Predicted boundaries: {predicted_boundaries}")
        
        # Calculate metrics
        results = metrics_calc.calculate_exact_metrics(
            predicted_boundaries,
            gold_boundaries,
            len(messages)
        )
        
        all_results.append(results)
        print(f"F1={results['f1']:.3f}, Precision={results['precision']:.3f}, Recall={results['recall']:.3f} ({elapsed:.1f}s)")
    
    # Calculate averages
    if all_results:
        avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
        avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
        avg_f1 = sum(r['f1'] for r in all_results) / len(all_results)
        avg_time = total_time / len(all_results)
    else:
        avg_precision = avg_recall = avg_f1 = avg_time = 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'total_time': total_time,
        'avg_time_per_dialogue': avg_time
    }


def main():
    print("Testing Robust Ollama Detector with phi3:mini")
    print("="*60)
    
    # Test with threshold 0.7
    results = evaluate_superdialseg_phi3_robust(
        threshold=0.7,
        window_size=3,
        num_dialogues=3  # Just test on 3 dialogues
    )
    
    print(f"\n\n{'='*60}")
    print("RESULTS:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1: {results['f1']:.3f}")
    print(f"  Avg time/dialogue: {results['avg_time_per_dialogue']:.2f}s")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("The robust detector uses:")
    print("1. Keyword extraction instead of full text")
    print("2. Very simple prompts")
    print("3. Robust parsing that handles verbose responses")
    print("4. Fallback to semantic interpretation of responses")


if __name__ == "__main__":
    main()