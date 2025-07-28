#!/usr/bin/env python3
"""
Run evaluation with optimized thresholds for each detector.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Define optimized configurations based on initial testing
detector_configs = [
    # Sliding window with different thresholds
    ('sliding_window', {'window_size': 3, 'threshold': 0.3}, 'sw_w3_t0.3'),
    ('sliding_window', {'window_size': 3, 'threshold': 0.4}, 'sw_w3_t0.4'),
    ('sliding_window', {'window_size': 3, 'threshold': 0.5}, 'sw_w3_t0.5'),
    ('sliding_window', {'window_size': 5, 'threshold': 0.3}, 'sw_w5_t0.3'),
    ('sliding_window', {'window_size': 5, 'threshold': 0.4}, 'sw_w5_t0.4'),
    
    # Keywords with different thresholds
    ('keywords', {'threshold': 0.3}, 'kw_t0.3'),
    ('keywords', {'threshold': 0.4}, 'kw_t0.4'),
    ('keywords', {'threshold': 0.5}, 'kw_t0.5'),
    
    # Hybrid with different thresholds
    ('hybrid', {'threshold': 0.4}, 'hybrid_t0.4'),
    ('hybrid', {'threshold': 0.5}, 'hybrid_t0.5'),
    ('hybrid', {'threshold': 0.6}, 'hybrid_t0.6'),
]

def run_single_evaluation(detector_type, config, name, dataset_path, max_dialogues=100):
    """Run evaluation for a single configuration."""
    cmd = [
        'python', 'evaluation/run_evaluation.py',
        str(dataset_path),
        '--detector', detector_type,
        '--max-dialogues', str(max_dialogues),
        '--output-dir', f'evaluation_results/{name}'
    ]
    
    # Add configuration parameters
    if 'threshold' in config:
        cmd.extend(['--threshold', str(config['threshold'])])
    if 'window_size' in config:
        cmd.extend(['--window-size', str(config['window_size'])])
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {name}:")
        print(result.stderr)
        return None
    
    # Parse results from output
    output_lines = result.stdout.split('\n')
    metrics = {}
    
    for line in output_lines:
        if 'Precision:' in line:
            metrics['precision'] = float(line.split(':')[1].strip())
        elif 'Recall:' in line and 'Recall:' == line.strip().split()[0]:
            metrics['recall'] = float(line.split(':')[1].strip())
        elif 'F1:' in line and 'F1:' == line.strip().split()[0]:
            metrics['f1'] = float(line.split(':')[1].strip())
        elif 'WindowDiff:' in line:
            metrics['window_diff'] = float(line.split(':')[1].strip())
        elif 'Pk:' in line and 'Pk:' == line.strip().split()[0]:
            metrics['pk'] = float(line.split(':')[1].strip())
        elif 'F1 (w=3):' in line:
            metrics['f1_w3'] = float(line.split(':')[1].strip())
    
    return metrics

def main():
    # Dataset path
    dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/superseg")
    
    # Results storage
    all_results = {}
    
    # Run each configuration
    for detector_type, config, name in detector_configs:
        metrics = run_single_evaluation(detector_type, config, name, dataset_path)
        if metrics:
            all_results[name] = {
                'detector': detector_type,
                'config': config,
                'metrics': metrics
            }
    
    # Save all results
    output_file = f"evaluation_results/optimized_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("evaluation_results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*100)
    print("OPTIMIZED EVALUATION RESULTS")
    print("="*100)
    print(f"{'Configuration':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'F1(w=3)':<10} {'WindowDiff':<12} {'Pk':<10}")
    print("-"*100)
    
    # Sort by F1 score
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['metrics'].get('f1', 0), 
                          reverse=True)
    
    for name, result in sorted_results:
        metrics = result['metrics']
        print(f"{name:<20} "
              f"{metrics.get('precision', 0):<10.3f} "
              f"{metrics.get('recall', 0):<10.3f} "
              f"{metrics.get('f1', 0):<10.3f} "
              f"{metrics.get('f1_w3', 0):<10.3f} "
              f"{metrics.get('window_diff', 0):<12.3f} "
              f"{metrics.get('pk', 0):<10.3f}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()