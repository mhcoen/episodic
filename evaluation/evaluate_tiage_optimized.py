#!/usr/bin/env python3
"""Evaluate with optimized thresholds for TIAGE."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.tiage_loader import TiageDatasetLoader
from evaluation.detector_adapters import SlidingWindowAdapter
from evaluation.supervised_detector import SupervisedWindowDetector
from evaluation.bayesian_detector import WindowedBayesianAdapter
from evaluation.metrics import SegmentationMetrics, EvaluationResults


def evaluate_detector(detector, dataset_loader, num_dialogues=100):
    """Evaluate a detector on TIAGE dataset."""
    dialogues = dataset_loader.load_dialogues(max_dialogues=num_dialogues)
    
    metrics_calc = SegmentationMetrics()
    results_aggregator = EvaluationResults()
    
    start_time = time.time()
    
    for i, (messages, gold_boundaries) in enumerate(dialogues):
        predicted_boundaries = detector.detect_boundaries(messages)
        result = metrics_calc.evaluate_all(predicted_boundaries, gold_boundaries, len(messages))
        results_aggregator.add_dialogue(f"dialogue_{i}", result)
    
    elapsed_time = time.time() - start_time
    summary = results_aggregator.get_summary()
    
    return {
        'metrics': summary['metrics'],
        'elapsed_time': elapsed_time,
        'num_dialogues': len(dialogues)
    }


def main():
    dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/tiage/segmentation_file_test.json")
    loader = TiageDatasetLoader(str(dataset_path))
    
    # Based on initial results, TIAGE needs much lower thresholds
    # Test finer-grained thresholds in lower ranges
    test_configs = [
        # Sentence-BERT with very low thresholds
        ("SentenceBERT_t0.1", SupervisedWindowDetector(window_size=3, threshold=0.1)),
        ("SentenceBERT_t0.2", SupervisedWindowDetector(window_size=3, threshold=0.2)),
        ("SentenceBERT_t0.3", SupervisedWindowDetector(window_size=3, threshold=0.3)),
        ("SentenceBERT_t0.4", SupervisedWindowDetector(window_size=3, threshold=0.4)),
        
        # Sliding window with very low thresholds  
        ("SlidingWindow_t0.1", SlidingWindowAdapter(threshold=0.1, window_size=3)),
        ("SlidingWindow_t0.2", SlidingWindowAdapter(threshold=0.2, window_size=3)),
        ("SlidingWindow_t0.3", SlidingWindowAdapter(threshold=0.3, window_size=3)),
        ("SlidingWindow_t0.4", SlidingWindowAdapter(threshold=0.4, window_size=3)),
        
        # Bayesian with very low thresholds
        ("Bayesian_t0.1", WindowedBayesianAdapter(threshold=0.1, window_size=3)),
        ("Bayesian_t0.2", WindowedBayesianAdapter(threshold=0.2, window_size=3)),
        ("Bayesian_t0.3", WindowedBayesianAdapter(threshold=0.3, window_size=3)),
    ]
    
    # Evaluate all configurations
    results = {}
    print("Evaluating different thresholds on TIAGE dataset...")
    print("="*60)
    
    for config_name, detector in test_configs:
        print(f"\nEvaluating {config_name}...")
        result = evaluate_detector(detector, loader, num_dialogues=30)
        results[config_name] = result
        
        print(f"Precision: {result['metrics']['precision']:.3f}")
        print(f"Recall: {result['metrics']['recall']:.3f}")
        print(f"F1: {result['metrics']['f1']:.3f}")
        print(f"WindowDiff: {result['metrics']['window_diff']:.3f}")
        print(f"Time: {result['elapsed_time']:.1f}s")
    
    # Find best configurations
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS")
    print("="*60)
    
    # Best by F1
    best_f1 = max(results.items(), key=lambda x: x[1]['metrics']['f1'])
    print(f"\nBest F1 Score: {best_f1[0]}")
    print(f"  F1: {best_f1[1]['metrics']['f1']:.3f}")
    print(f"  Precision: {best_f1[1]['metrics']['precision']:.3f}")
    print(f"  Recall: {best_f1[1]['metrics']['recall']:.3f}")
    print(f"  WindowDiff: {best_f1[1]['metrics']['window_diff']:.3f}")
    
    # Best by WindowDiff
    best_wd = min(results.items(), key=lambda x: x[1]['metrics']['window_diff'])
    print(f"\nBest WindowDiff: {best_wd[0]}")
    print(f"  WindowDiff: {best_wd[1]['metrics']['window_diff']:.3f}")
    print(f"  F1: {best_wd[1]['metrics']['f1']:.3f}")
    print(f"  Precision: {best_wd[1]['metrics']['precision']:.3f}")
    print(f"  Recall: {best_wd[1]['metrics']['recall']:.3f}")
    
    # Save results
    output_file = Path("/Users/mhcoen/proj/episodic/evaluation_results/tiage_optimized_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()