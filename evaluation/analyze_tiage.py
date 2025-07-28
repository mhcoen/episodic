#!/usr/bin/env python3
"""Analyze TIAGE dataset characteristics and optimize thresholds."""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.tiage_loader import TiageDatasetLoader
from evaluation.detector_adapters import SlidingWindowAdapter
from evaluation.supervised_detector import SupervisedWindowDetector
from evaluation.metrics import SegmentationMetrics, EvaluationResults


def analyze_dataset_characteristics(dataset_path):
    """Analyze TIAGE dataset characteristics."""
    loader = TiageDatasetLoader(dataset_path)
    dialogues = loader.load_dialogues()
    
    print(f"\nAnalyzing {len(dialogues)} dialogues...")
    
    # Analyze boundary patterns
    boundary_positions = []
    dialogue_lengths = []
    boundaries_per_dialogue = []
    segment_lengths = []
    
    for messages, boundaries in dialogues:
        dialogue_lengths.append(len(messages))
        boundaries_per_dialogue.append(len(boundaries))
        
        # Relative positions of boundaries
        if boundaries:
            for b in boundaries:
                boundary_positions.append(b / len(messages))
            
            # Segment lengths
            prev = 0
            for b in boundaries:
                segment_lengths.append(b - prev)
                prev = b + 1
            segment_lengths.append(len(messages) - prev)
    
    print(f"\nDataset Statistics:")
    print(f"Total dialogues: {len(dialogues)}")
    print(f"Dialogue length: mean={np.mean(dialogue_lengths):.1f}, std={np.std(dialogue_lengths):.1f}")
    print(f"Boundaries per dialogue: mean={np.mean(boundaries_per_dialogue):.1f}, std={np.std(boundaries_per_dialogue):.1f}")
    print(f"Segment length: mean={np.mean(segment_lengths):.1f}, std={np.std(segment_lengths):.1f}")
    print(f"Boundary positions (relative): mean={np.mean(boundary_positions):.2f}, std={np.std(boundary_positions):.2f}")
    
    # Sample some dialogues to understand content
    print("\nSample dialogues:")
    for i in range(min(3, len(dialogues))):
        messages, boundaries = dialogues[i]
        print(f"\nDialogue {i+1}: {len(messages)} turns, boundaries at {boundaries}")
        for j, msg in enumerate(messages[:10]):  # First 10 messages
            marker = " <-- BOUNDARY" if j in boundaries else ""
            print(f"  {j}: [{msg['role']}] {msg['content'][:50]}...{marker}")


def optimize_thresholds(dataset_path, num_dialogues=50):
    """Find optimal thresholds for TIAGE dataset."""
    loader = TiageDatasetLoader(dataset_path)
    
    # Test threshold ranges
    threshold_ranges = {
        'sliding_window': np.arange(0.1, 0.9, 0.1),
        'sentence_bert': np.arange(0.1, 0.9, 0.1)
    }
    
    results = {}
    
    # Test sliding window
    print("\nOptimizing Sliding Window thresholds...")
    sw_results = []
    for threshold in threshold_ranges['sliding_window']:
        detector = SlidingWindowAdapter(threshold=threshold, window_size=3)
        metrics_calc = SegmentationMetrics()
        results_agg = EvaluationResults()
        
        dialogues = loader.load_dialogues(max_dialogues=num_dialogues)
        for i, (messages, gold_boundaries) in enumerate(dialogues):
            predicted = detector.detect_boundaries(messages)
            result = metrics_calc.evaluate_all(predicted, gold_boundaries, len(messages))
            results_agg.add_dialogue(f"d_{i}", result)
        
        summary = results_agg.get_summary()
        sw_results.append((threshold, summary['metrics']))
        print(f"  t={threshold:.1f}: F1={summary['metrics']['f1']:.3f}, WD={summary['metrics']['window_diff']:.3f}")
    
    # Test Sentence-BERT
    print("\nOptimizing Sentence-BERT thresholds...")
    sb_results = []
    for threshold in threshold_ranges['sentence_bert']:
        detector = SupervisedWindowDetector(window_size=3, threshold=threshold)
        metrics_calc = SegmentationMetrics()
        results_agg = EvaluationResults()
        
        dialogues = loader.load_dialogues(max_dialogues=num_dialogues)
        for i, (messages, gold_boundaries) in enumerate(dialogues):
            predicted = detector.detect_boundaries(messages)
            result = metrics_calc.evaluate_all(predicted, gold_boundaries, len(messages))
            results_agg.add_dialogue(f"d_{i}", result)
        
        summary = results_agg.get_summary()
        sb_results.append((threshold, summary['metrics']))
        print(f"  t={threshold:.1f}: F1={summary['metrics']['f1']:.3f}, WD={summary['metrics']['window_diff']:.3f}")
    
    # Find best thresholds
    best_sw = max(sw_results, key=lambda x: x[1]['f1'])
    best_sb = max(sb_results, key=lambda x: x[1]['f1'])
    
    print(f"\nBest Sliding Window threshold: {best_sw[0]:.1f} (F1={best_sw[1]['f1']:.3f})")
    print(f"Best Sentence-BERT threshold: {best_sb[0]:.1f} (F1={best_sb[1]['f1']:.3f})")
    
    return {
        'sliding_window': {'best_threshold': best_sw[0], 'metrics': best_sw[1]},
        'sentence_bert': {'best_threshold': best_sb[0], 'metrics': best_sb[1]},
        'all_results': {'sliding_window': sw_results, 'sentence_bert': sb_results}
    }


def main():
    dataset_path = "/Users/mhcoen/proj/episodic/datasets/tiage/segmentation_file_test.json"
    
    # Analyze dataset
    analyze_dataset_characteristics(dataset_path)
    
    # Optimize thresholds
    print("\n" + "="*50)
    print("THRESHOLD OPTIMIZATION")
    print("="*50)
    optimization_results = optimize_thresholds(dataset_path)
    
    # Save results
    output_path = Path("/Users/mhcoen/proj/episodic/evaluation_results/tiage_optimization.json")
    with open(output_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print(f"\nOptimization results saved to: {output_path}")


if __name__ == "__main__":
    main()