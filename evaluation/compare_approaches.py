#!/usr/bin/env python3
"""
Compare all topic detection approaches on both datasets.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

# Define all approaches to test
approaches = [
    # Original sliding window
    ('sliding_window', {'window_size': 3, 'threshold': 0.3}, 'SW-0.3'),
    ('sliding_window', {'window_size': 3, 'threshold': 0.5}, 'SW-0.5'),
    
    # Keywords
    ('keywords', {'threshold': 0.3}, 'Keywords'),
    
    # Bayesian
    ('bayesian', {'threshold': 0.25, 'variant': 'windowed'}, 'Bayesian-W'),
    
    # Supervised (if transformers available)
    ('supervised', {'threshold': 0.5, 'variant': 'window'}, 'Supervised'),
]

datasets = {
    'superseg': '/Users/mhcoen/proj/episodic/datasets/superseg',
    'dialseg711': '/Users/mhcoen/proj/episodic/datasets/dialseg711',
}

def run_single_evaluation(dataset_name, dataset_path, detector_type, config, max_dialogues=50):
    """Run evaluation and extract metrics."""
    cmd = [
        'python', 'evaluation/run_evaluation.py',
        dataset_path,
        '--detector', detector_type,
        '--max-dialogues', str(max_dialogues),
        '--output-dir', f'evaluation_results/comparison_{dataset_name}'
    ]
    
    # Add config parameters
    for key, value in config.items():
        if key == 'variant':
            continue  # Handle internally
        cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    print(f"  Running {detector_type} on {dataset_name}...", end='', flush=True)
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f" ERROR")
        print(f"    {result.stderr}")
        return None
    
    print(f" done ({elapsed:.1f}s)")
    
    # Parse metrics
    metrics = {'time': elapsed}
    for line in result.stdout.split('\n'):
        if 'Precision:' in line and 'Main Metrics' in result.stdout:
            metrics['precision'] = float(line.split(':')[1].strip())
        elif 'Recall:' in line and 'Recall:' == line.strip().split()[0]:
            metrics['recall'] = float(line.split(':')[1].strip())
        elif 'F1:' in line and 'F1:' == line.strip().split()[0]:
            metrics['f1'] = float(line.split(':')[1].strip())
        elif 'F1 (w=3):' in line:
            metrics['f1_w3'] = float(line.split(':')[1].strip())
        elif 'WindowDiff:' in line:
            metrics['window_diff'] = float(line.split(':')[1].strip())
    
    return metrics

def main():
    print("="*80)
    print("COMPREHENSIVE TOPIC DETECTION COMPARISON")
    print("="*80)
    print(f"Testing {len(approaches)} approaches on {len(datasets)} datasets\n")
    
    results = {}
    
    # Run all evaluations
    for dataset_name, dataset_path in datasets.items():
        print(f"\nEvaluating on {dataset_name}:")
        results[dataset_name] = {}
        
        for detector_type, config, label in approaches:
            metrics = run_single_evaluation(
                dataset_name, dataset_path, detector_type, config
            )
            if metrics:
                results[dataset_name][label] = metrics
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"evaluation_results/approach_comparison_{timestamp}.json"
    Path("evaluation_results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"{'Approach':<15} {'Dataset':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'F1(w=3)':<10} {'WinDiff':<10} {'Time(s)':<10}")
    print("-"*100)
    
    # Results for each approach
    for detector_type, config, label in approaches:
        for dataset_name in datasets:
            if label in results.get(dataset_name, {}):
                m = results[dataset_name][label]
                print(f"{label:<15} {dataset_name:<12} "
                      f"{m.get('precision', 0):<10.3f} "
                      f"{m.get('recall', 0):<10.3f} "
                      f"{m.get('f1', 0):<10.3f} "
                      f"{m.get('f1_w3', 0):<10.3f} "
                      f"{m.get('window_diff', 0):<10.3f} "
                      f"{m.get('time', 0):<10.1f}")
    
    # Best performers
    print("\n" + "="*100)
    print("BEST PERFORMERS")
    print("="*100)
    
    for dataset_name in datasets:
        if dataset_name in results and results[dataset_name]:
            print(f"\n{dataset_name}:")
            
            # Best F1
            best_f1 = max(results[dataset_name].items(), 
                         key=lambda x: x[1].get('f1', 0))
            print(f"  Best F1: {best_f1[0]} (F1={best_f1[1]['f1']:.3f})")
            
            # Best F1 w=3
            best_f1_w3 = max(results[dataset_name].items(), 
                           key=lambda x: x[1].get('f1_w3', 0))
            print(f"  Best F1(w=3): {best_f1_w3[0]} (F1={best_f1_w3[1]['f1_w3']:.3f})")
            
            # Fastest
            fastest = min(results[dataset_name].items(), 
                        key=lambda x: x[1].get('time', float('inf')))
            print(f"  Fastest: {fastest[0]} ({fastest[1]['time']:.1f}s)")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()