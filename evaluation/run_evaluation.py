#!/usr/bin/env python3
"""
Run topic detection evaluation on SuperDialseg dataset.

This script evaluates different topic detection strategies from Episodic
on the SuperDialseg benchmark dataset.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superdialseg_loader import SuperDialsegLoader
from metrics import SegmentationMetrics, EvaluationResults
from detector_adapters import create_detector, BaseDetectorAdapter


def evaluate_detector_on_dataset(
    detector: BaseDetectorAdapter,
    dataset_path: Path,
    split: str = 'test',
    max_dialogues: int = None,
    verbose: bool = False
) -> EvaluationResults:
    """
    Evaluate a detector on a dataset split.
    
    Args:
        detector: The detector adapter to evaluate
        dataset_path: Path to the dataset
        split: Dataset split to use ('train', 'validation', 'test')
        max_dialogues: Maximum number of dialogues to evaluate (None for all)
        verbose: Print progress information
        
    Returns:
        EvaluationResults object with metrics
    """
    loader = SuperDialsegLoader()
    metrics_calc = SegmentationMetrics()
    results = EvaluationResults()
    
    # Load conversations
    if verbose:
        print(f"\nLoading {split} split from {dataset_path}...")
    
    try:
        conversations = loader.load_conversations(dataset_path, split)
    except FileNotFoundError as e:
        print(f"Error: Could not load {split} split: {e}")
        return results
    
    if max_dialogues:
        conversations = conversations[:max_dialogues]
    
    if verbose:
        print(f"Evaluating {len(conversations)} dialogues...")
    
    # Process each conversation
    for i, conv_data in enumerate(conversations):
        if verbose and i % 10 == 0:
            print(f"  Processing dialogue {i+1}/{len(conversations)}...")
        
        try:
            # Parse conversation
            messages, gold_boundaries = loader.parse_conversation(conv_data)
            
            # Skip very short conversations
            if len(messages) < 4:
                continue
            
            # Detect boundaries
            predicted_boundaries = detector.detect_boundaries(messages)
            
            # Debug output for first few dialogues
            if verbose and i < 3:
                print(f"    Messages: {len(messages)}")
                print(f"    Gold boundaries: {gold_boundaries[:5]}...")
                print(f"    Predicted boundaries: {predicted_boundaries[:5]}...")
            
            # Calculate metrics
            dialogue_metrics = metrics_calc.evaluate_all(
                predicted_boundaries,
                gold_boundaries,
                len(messages)
            )
            
            # Store results
            dialogue_id = conv_data.get('id', f'dialogue_{i}')
            results.add_dialogue(dialogue_id, dialogue_metrics)
            
            # Reset detector state
            detector.reset()
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Error processing dialogue {i}: {e}")
            continue
    
    return results


def run_evaluation(args):
    """Run the evaluation based on command line arguments."""
    # Check if dataset exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"\nError: Dataset not found at {dataset_path}")
        print(f"Available datasets in the project:")
        datasets_dir = Path("/Users/mhcoen/proj/episodic/datasets")
        if datasets_dir.exists():
            for item in datasets_dir.iterdir():
                if item.is_dir():
                    print(f"  {item}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure detectors to evaluate
    detectors_config = []
    
    if args.detector == 'all':
        # Test all detector types
        detectors_config = [
            ('sliding_window_w3', {'window_size': 3, 'threshold': 0.9}),
            ('sliding_window_w5', {'window_size': 5, 'threshold': 0.9}),
            ('keywords', {'threshold': 0.5}),
            ('hybrid', {'threshold': 0.6}),
            ('bayesian', {'variant': 'windowed'}),
        ]
        
        # Only add LLM if not skipping
        if not args.skip_llm:
            detectors_config.append(('llm', {}))
            
    else:
        # Single detector
        kwargs = {}
        if args.threshold is not None:
            kwargs['threshold'] = args.threshold
        if args.window_size is not None:
            kwargs['window_size'] = args.window_size
        if args.model is not None:
            kwargs['model'] = args.model
            
        detectors_config = [(args.detector, kwargs)]
    
    # Results storage
    all_results = {}
    
    # Evaluate each detector
    for detector_name, kwargs in detectors_config:
        print(f"\n{'='*60}")
        print(f"Evaluating: {detector_name}")
        print(f"{'='*60}")
        
        # Create detector
        # Extract base detector type (e.g., 'sliding_window' from 'sliding_window_w3')
        if detector_name.startswith('sliding_window'):
            detector_type = 'sliding_window'
        else:
            detector_type = detector_name
        detector = create_detector(detector_type, **kwargs)
        
        # Run evaluation
        start_time = time.time()
        results = evaluate_detector_on_dataset(
            detector,
            dataset_path,
            split=args.split,
            max_dialogues=args.max_dialogues,
            verbose=args.verbose
        )
        elapsed_time = time.time() - start_time
        
        # Get summary
        summary = results.get_summary()
        summary['elapsed_time'] = elapsed_time
        summary['detector_config'] = kwargs
        
        # Print results
        results.print_summary()
        print(f"\nElapsed time: {elapsed_time:.2f} seconds")
        
        # Store results
        all_results[detector.name] = {
            'summary': summary,
            'dialogue_results': results.dialogue_results if args.save_detailed else []
        }
    
    # Save results to file
    output_file = output_dir / f"evaluation_results_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Print comparison table if multiple detectors
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        print(f"{'Detector':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'WindowDiff':<12} {'Pk':<10}")
        print("-"*80)
        
        for name, result in all_results.items():
            metrics = result['summary']['metrics']
            print(f"{name:<25} "
                  f"{metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} "
                  f"{metrics['window_diff']:<12.3f} "
                  f"{metrics['pk']:<10.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Episodic topic detection on SuperDialseg dataset"
    )
    
    parser.add_argument(
        'dataset_path',
        help='Path to SuperDialseg dataset directory'
    )
    
    parser.add_argument(
        '--detector',
        choices=['sliding_window', 'hybrid', 'keywords', 'llm', 'combined', 'bayesian', 'supervised', 'all'],
        default='all',
        help='Detector type to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--split',
        choices=['train', 'validation', 'test'],
        default='test',
        help='Dataset split to use (default: test)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='evaluation_results',
        help='Directory to save results (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--max-dialogues',
        type=int,
        help='Maximum number of dialogues to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='Detection threshold to use'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        help='Window size for sliding window detector'
    )
    
    parser.add_argument(
        '--model',
        help='LLM model to use for LLM detector'
    )
    
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM-based detection (faster evaluation)'
    )
    
    parser.add_argument(
        '--save-detailed',
        action='store_true',
        help='Save per-dialogue results (creates larger output file)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()