#!/usr/bin/env python3
"""Evaluate topic detection on MP2D dataset."""

import sys
import json
import time
from pathlib import Path

# Add episodic to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.mp2d_loader import MP2DDatasetLoader
from evaluation.detector_adapters import (
    SlidingWindowAdapter,
    HybridDetectorAdapter,
    KeywordDetectorAdapter
)
from evaluation.supervised_detector import SupervisedWindowDetector
from evaluation.bayesian_detector import WindowedBayesianAdapter
from evaluation.metrics import SegmentationMetrics, EvaluationResults


def evaluate_detector_on_mp2d(detector, dataset_loader, num_dialogues=50, verbose=True):
    """Evaluate a detector on the MP2D dataset."""
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
    dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/MP2D_mini_sample.json")
    output_dir = Path("/Users/mhcoen/proj/episodic/evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset info
    loader = MP2DDatasetLoader(str(dataset_path))
    dataset_info = loader.get_dataset_info()
    print(f"\nMP2D Dataset Info:")
    print(f"Total dialogues: {dataset_info['num_dialogues']}")
    print(f"Avg turns per dialogue: {dataset_info['avg_turns_per_dialogue']:.1f}")
    print(f"Avg messages per dialogue: {dataset_info['avg_messages_per_dialogue']:.1f}")
    print(f"Avg boundaries per dialogue: {dataset_info['avg_boundaries_per_dialogue']:.1f}")
    print(f"Unique topics: {dataset_info['unique_topics']}")
    
    # Test configurations based on best performers from other datasets
    test_configs = [
        # Sentence-BERT with optimal thresholds from other datasets
        ("supervised_all-MiniLM-L6-v2_w3_t0.3", 
         lambda: SupervisedWindowDetector(window_size=3, threshold=0.3)),
        ("supervised_all-MiniLM-L6-v2_w3_t0.5", 
         lambda: SupervisedWindowDetector(window_size=3, threshold=0.5)),
        
        # Sliding window
        ("sliding_window_w3_t0.3", 
         lambda: SlidingWindowAdapter(threshold=0.3, window_size=3)),
        ("sliding_window_w3_t0.5", 
         lambda: SlidingWindowAdapter(threshold=0.5, window_size=3)),
        
        # Bayesian
        ("bayesian_windowed_w3_t0.25",
         lambda: WindowedBayesianAdapter(threshold=0.25, window_size=3)),
        
        # Keywords (for completeness)
        ("keywords_t0.5",
         lambda: KeywordDetectorAdapter(threshold=0.5)),
    ]
    
    # Run evaluations
    results = {}
    num_dialogues = min(50, dataset_info['num_dialogues'])  # Use up to 50 dialogues
    
    print(f"\nEvaluating on {num_dialogues} dialogues...\n")
    
    for config_name, detector_factory in test_configs:
        print(f"\nEvaluating {config_name}...")
        detector = detector_factory()
        
        try:
            result = evaluate_detector_on_mp2d(
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
    output_file = output_dir / f"mp2d_evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Find best configurations
    print("\n" + "="*50)
    print("BEST CONFIGURATIONS FOR MP2D DATASET")
    print("="*50)
    
    if results:
        # Best by F1
        best_f1 = max(results.items(), 
                      key=lambda x: x[1]['summary']['metrics']['f1'])
        print(f"\nBest F1 Score: {best_f1[0]}")
        print(f"F1: {best_f1[1]['summary']['metrics']['f1']:.3f}")
        print(f"Precision: {best_f1[1]['summary']['metrics']['precision']:.3f}")
        print(f"Recall: {best_f1[1]['summary']['metrics']['recall']:.3f}")
        print(f"WindowDiff: {best_f1[1]['summary']['metrics']['window_diff']:.3f}")
        
        # Best by WindowDiff
        best_wd = min(results.items(), 
                      key=lambda x: x[1]['summary']['metrics']['window_diff'])
        print(f"\nBest WindowDiff: {best_wd[0]}")
        print(f"WindowDiff: {best_wd[1]['summary']['metrics']['window_diff']:.3f}")
        print(f"F1: {best_wd[1]['summary']['metrics']['f1']:.3f}")
        
        # Compare with other datasets
        print("\n" + "="*50)
        print("CROSS-DATASET COMPARISON")
        print("="*50)
        print("\nBest F1 Scores by Dataset:")
        print(f"SuperDialseg: 0.571 (Sentence-BERT)")
        print(f"DialSeg711: 0.467 (Sentence-BERT)")
        print(f"TIAGE: 0.222 (Sentence-BERT)")
        print(f"MP2D: {best_f1[1]['summary']['metrics']['f1']:.3f} ({best_f1[0].split('_')[0]})")


if __name__ == "__main__":
    main()