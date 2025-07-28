#!/usr/bin/env python3
"""
Compare evaluation results across different datasets.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

# Test configurations
configs = [
    {'threshold': 0.3, 'window_size': 3},
    {'threshold': 0.5, 'window_size': 3},
    {'threshold': 0.7, 'window_size': 3},
    {'threshold': 0.9, 'window_size': 3},
]

datasets = [
    ('superseg', '/Users/mhcoen/proj/episodic/datasets/superseg'),
    ('dialseg711', '/Users/mhcoen/proj/episodic/datasets/dialseg711'),
]

def run_evaluation(dataset_name, dataset_path, config, max_dialogues=100):
    """Run evaluation and return metrics."""
    cmd = [
        'python', 'evaluation/run_evaluation.py',
        dataset_path,
        '--detector', 'sliding_window',
        '--threshold', str(config['threshold']),
        '--window-size', str(config['window_size']),
        '--max-dialogues', str(max_dialogues),
        '--output-dir', f'evaluation_results/{dataset_name}_comparison'
    ]
    
    print(f"Running {dataset_name} with threshold={config['threshold']}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Parse metrics from output
    metrics = {}
    for line in result.stdout.split('\n'):
        if 'Precision:' in line and 'Main Metrics' in result.stdout:
            metrics['precision'] = float(line.split(':')[1].strip())
        elif 'Recall:' in line and 'Recall:' == line.strip().split()[0]:
            metrics['recall'] = float(line.split(':')[1].strip())
        elif 'F1:' in line and 'F1:' == line.strip().split()[0]:
            metrics['f1'] = float(line.split(':')[1].strip())
        elif 'WindowDiff:' in line:
            metrics['window_diff'] = float(line.split(':')[1].strip())
        elif 'F1 (w=3):' in line:
            metrics['f1_w3'] = float(line.split(':')[1].strip())
    
    return metrics

def main():
    results = {}
    
    # Run evaluations
    for dataset_name, dataset_path in datasets:
        results[dataset_name] = {}
        for config in configs:
            key = f"t{config['threshold']}"
            metrics = run_evaluation(dataset_name, dataset_path, config)
            if metrics:
                results[dataset_name][key] = metrics
    
    # Save results
    output_file = f"evaluation_results/dataset_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*80)
    print("DATASET COMPARISON - Sliding Window (w=3)")
    print("="*80)
    
    # Print header
    print(f"{'Dataset':<15} {'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'F1(w=3)':<10} {'WindowDiff':<10}")
    print("-"*80)
    
    # Print results for each dataset
    for dataset_name in datasets:
        dataset_results = results.get(dataset_name[0], {})
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            key = f"t{threshold}"
            if key in dataset_results:
                m = dataset_results[key]
                print(f"{dataset_name[0]:<15} {threshold:<10.1f} "
                      f"{m.get('precision', 0):<10.3f} "
                      f"{m.get('recall', 0):<10.3f} "
                      f"{m.get('f1', 0):<10.3f} "
                      f"{m.get('f1_w3', 0):<10.3f} "
                      f"{m.get('window_diff', 0):<10.3f}")
    
    # Find best threshold for each dataset
    print("\n" + "="*80)
    print("OPTIMAL THRESHOLDS")
    print("="*80)
    
    for dataset_name in datasets:
        dataset_results = results.get(dataset_name[0], {})
        if dataset_results:
            # Find best by F1 score
            best_f1 = max(dataset_results.items(), 
                         key=lambda x: x[1].get('f1', 0))
            # Find best by WindowDiff (lower is better)
            best_wd = min(dataset_results.items(), 
                         key=lambda x: x[1].get('window_diff', 1.0))
            
            print(f"\n{dataset_name[0]}:")
            print(f"  Best F1: threshold={best_f1[0][1:]} (F1={best_f1[1]['f1']:.3f})")
            print(f"  Best WindowDiff: threshold={best_wd[0][1:]} (WD={best_wd[1]['window_diff']:.3f})")

if __name__ == "__main__":
    main()