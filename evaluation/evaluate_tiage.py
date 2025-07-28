#!/usr/bin/env python3
"""Evaluate topic detection on TIAGE dataset."""

import sys
import json
import time
from pathlib import Path

# Add episodic to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.tiage_loader import TiageDatasetLoader
from evaluation.detector_adapters import (
    SlidingWindowAdapter,
    HybridDetectorAdapter,
    KeywordDetectorAdapter
)
from evaluation.supervised_detector import SupervisedWindowDetector
from evaluation.bayesian_detector import WindowedBayesianAdapter
from evaluation.metrics import SegmentationMetrics, EvaluationResults


def evaluate_detector_on_tiage(detector, dataset_loader, num_dialogues=100, verbose=True):
    """Evaluate a detector on the TIAGE dataset."""
    dialogues = dataset_loader.load_dialogues(max_dialogues=num_dialogues)
    
    metrics_calc = SegmentationMetrics()
    results_aggregator = EvaluationResults()
    
    start_time = time.time()
    
    for i, (messages, gold_boundaries) in enumerate(dialogues):
        if verbose and i % 10 == 0:
            print(f"Processing dialogue {i+1}/{len(dialogues)}...")
        
        # Detect boundaries
        predicted_boundaries = detector.detect_boundaries(messages)
        
        # Calculate all metrics
        result = metrics_calc.evaluate_all(
            predicted_boundaries,
            gold_boundaries,
            len(messages)
        )
        
        results_aggregator.add_dialogue(f"dialogue_{i}", result)
    
    elapsed_time = time.time() - start_time
    
    # Get aggregated results
    summary = results_aggregator.get_summary()
    aggregates = results_aggregator.get_aggregate_metrics()
    
    return {
        'summary': {
            'num_dialogues': len(dialogues),
            'metrics': summary['metrics'],
            'detailed_aggregates': aggregates,
            'elapsed_time': elapsed_time,
            'detector_config': getattr(detector, 'config', {})
        },
        'dialogue_results': results_aggregator.dialogue_results
    }


def main():
    # Paths
    dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/tiage/segmentation_file_test.json")
    output_dir = Path("/Users/mhcoen/proj/episodic/evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset info
    loader = TiageDatasetLoader(str(dataset_path))
    dataset_info = loader.get_dataset_info()
    print(f"\nTIAGE Dataset Info:")
    print(f"Total dialogues: {dataset_info['num_dialogues']}")
    print(f"Avg turns per dialogue: {dataset_info['avg_turns_per_dialogue']:.1f}")
    print(f"Avg boundaries per dialogue: {dataset_info['avg_boundaries_per_dialogue']:.1f}")
    
    # Test configurations based on optimal thresholds from other datasets
    test_configs = [
        # Sentence-BERT with different thresholds
        ("supervised_all-MiniLM-L6-v2_w3_t0.3", 
         lambda: SupervisedWindowDetector(window_size=3, threshold=0.3)),
        ("supervised_all-MiniLM-L6-v2_w3_t0.5", 
         lambda: SupervisedWindowDetector(window_size=3, threshold=0.5)),
        
        # Sliding window with different thresholds
        ("sliding_window_w3_t0.3", 
         lambda: SlidingWindowAdapter(threshold=0.3, window_size=3)),
        ("sliding_window_w3_t0.7", 
         lambda: SlidingWindowAdapter(threshold=0.7, window_size=3)),
        
        # Bayesian with different thresholds
        ("bayesian_windowed_w3_t0.25",
         lambda: WindowedBayesianAdapter(threshold=0.25, window_size=3)),
        ("bayesian_windowed_w3_t0.3",
         lambda: WindowedBayesianAdapter(threshold=0.3, window_size=3)),
    ]
    
    # Run evaluations
    results = {}
    num_dialogues = min(300, dataset_info['num_dialogues'])  # Use up to 300 dialogues
    
    print(f"\nEvaluating on {num_dialogues} dialogues...\n")
    
    for config_name, detector_factory in test_configs:
        print(f"\nEvaluating {config_name}...")
        detector = detector_factory()
        
        try:
            result = evaluate_detector_on_tiage(
                detector, 
                loader,
                num_dialogues=num_dialogues,
                verbose=True
            )
            
            results[config_name] = result
            
            # Print summary
            summary = result['summary']['metrics']
            print(f"\nResults for {config_name}:")
            print(f"Precision: {summary['precision']:.3f}")
            print(f"Recall: {summary['recall']:.3f}")
            print(f"F1: {summary['f1']:.3f}")
            print(f"WindowDiff: {summary['window_diff']:.3f}")
            print(f"Pk: {summary['pk']:.3f}")
            print(f"Time: {result['summary']['elapsed_time']:.1f}s")
            
        except Exception as e:
            print(f"Error evaluating {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"tiage_evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Find best configurations
    print("\n" + "="*50)
    print("BEST CONFIGURATIONS FOR TIAGE DATASET")
    print("="*50)
    
    # Best by F1
    best_f1 = max(results.items(), 
                  key=lambda x: x[1]['summary']['metrics']['f1'])
    print(f"\nBest F1 Score: {best_f1[0]}")
    print(f"F1: {best_f1[1]['summary']['metrics']['f1']:.3f}")
    print(f"WindowDiff: {best_f1[1]['summary']['metrics']['window_diff']:.3f}")
    
    # Best by WindowDiff
    best_wd = min(results.items(), 
                  key=lambda x: x[1]['summary']['metrics']['window_diff'])
    print(f"\nBest WindowDiff: {best_wd[0]}")
    print(f"F1: {best_wd[1]['summary']['metrics']['f1']:.3f}")
    print(f"WindowDiff: {best_wd[1]['summary']['metrics']['window_diff']:.3f}")


if __name__ == "__main__":
    main()