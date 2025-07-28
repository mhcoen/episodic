#!/usr/bin/env python3
"""Evaluate small models with window_size=3 for fair comparison."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.tiage_loader import TiageDatasetLoader
from evaluation.metrics import SegmentationMetrics, EvaluationResults


def evaluate_on_dataset(detector, dataset_loader, dataset_name, num_dialogues=10):
    """Evaluate detector on a dataset."""
    print(f"\nEvaluating on {dataset_name} ({num_dialogues} dialogues)...")
    
    dialogues = dataset_loader.load_dialogues(max_dialogues=num_dialogues)
    metrics_calc = SegmentationMetrics()
    results_aggregator = EvaluationResults()
    
    start_time = time.time()
    
    for i, (messages, gold_boundaries) in enumerate(dialogues):
        if i % 5 == 0:
            print(f"  Processing dialogue {i+1}/{len(dialogues)}...")
        
        # Detect boundaries
        predicted_boundaries = detector.detect_boundaries(messages)
        
        # Calculate metrics
        result = metrics_calc.evaluate_all(
            predicted_boundaries,
            gold_boundaries,
            len(messages)
        )
        
        results_aggregator.add_dialogue(f"dialogue_{i}", result)
    
    elapsed_time = time.time() - start_time
    
    # Get summary
    summary = results_aggregator.get_summary()
    
    print(f"\nResults for {dataset_name}:")
    print(f"Precision: {summary['metrics']['precision']:.3f}")
    print(f"Recall: {summary['metrics']['recall']:.3f}")
    print(f"F1: {summary['metrics']['f1']:.3f}")
    print(f"WindowDiff: {summary['metrics']['window_diff']:.3f}")
    print(f"Time: {elapsed_time:.1f}s ({elapsed_time/len(dialogues):.2f}s per dialogue)")
    
    return {
        'dataset': dataset_name,
        'num_dialogues': len(dialogues),
        'metrics': summary['metrics'],
        'time': elapsed_time,
        'time_per_dialogue': elapsed_time/len(dialogues)
    }


def main():
    print("Evaluating Small Models with Window Size 3 (Fair Comparison)")
    print("="*60)
    
    # Models to test with window_size=3
    models = [
        ("qwen2:0.5b", 0.5, 352),
        ("qwen2:0.5b", 0.3, 352),  # Try lower threshold
        ("tinyllama:latest", 0.7, 637),
        ("tinyllama:latest", 0.5, 637),  # Try lower threshold
        ("qwen2:1.5b", 0.6, 934),
        ("qwen2:1.5b", 0.4, 934),  # Try lower threshold
    ]
    
    # Datasets
    datasets = [
        ("SuperDialseg", SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")),
        ("TIAGE", TiageDatasetLoader("/Users/mhcoen/proj/episodic/datasets/tiage/segmentation_file_test.json")),
    ]
    
    # Number of dialogues to test
    num_dialogues = 10
    
    all_results = []
    
    for model_name, threshold, size_mb in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name} (Size: {size_mb} MB)")
        print(f"Window Size: 3, Threshold: {threshold}")
        print('='*60)
        
        # Create detector with window_size=3
        detector = OllamaInstructDetector(
            model_name=model_name,
            threshold=threshold,
            window_size=3,  # Now using same window size as other methods
            verbose=False
        )
        
        model_results = {
            'model': model_name,
            'size_mb': size_mb,
            'threshold': threshold,
            'window_size': 3,
            'results': {}
        }
        
        for dataset_name, loader in datasets:
            try:
                result = evaluate_on_dataset(
                    detector, 
                    loader, 
                    dataset_name, 
                    num_dialogues
                )
                model_results['results'][dataset_name] = result
            except Exception as e:
                print(f"Error on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        
        all_results.append(model_results)
    
    # Save results
    output_file = Path("evaluation_results/small_models_window3_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Small Models with Window Size 3")
    print("="*60)
    print(f"\n{'Model':<25} {'Threshold':<10} {'SuperDialseg F1':<18} {'TIAGE F1':<12} {'Avg Time/Dialog':<15}")
    print("-"*80)
    
    best_super_f1 = 0
    best_tiage_f1 = 0
    best_super_config = ""
    best_tiage_config = ""
    
    for result in all_results:
        model = result['model']
        threshold = result['threshold']
        
        if 'SuperDialseg' in result['results']:
            super_f1 = result['results']['SuperDialseg']['metrics']['f1']
            super_time = result['results']['SuperDialseg']['time_per_dialogue']
            if super_f1 > best_super_f1:
                best_super_f1 = super_f1
                best_super_config = f"{model} (t={threshold})"
        else:
            super_f1 = 0.0
            super_time = 0.0
            
        if 'TIAGE' in result['results']:
            tiage_f1 = result['results']['TIAGE']['metrics']['f1']
            tiage_time = result['results']['TIAGE']['time_per_dialogue']
            if tiage_f1 > best_tiage_f1:
                best_tiage_f1 = tiage_f1
                best_tiage_config = f"{model} (t={threshold})"
        else:
            tiage_f1 = 0.0
            tiage_time = 0.0
        
        avg_time = (super_time + tiage_time) / 2
        
        config_str = f"{model} (t={threshold})"
        print(f"{config_str:<25} {threshold:<10.1f} {super_f1:<18.3f} {tiage_f1:<12.3f} {avg_time:<15.2f}s")
    
    # Compare with other methods using window_size=3
    print("\n" + "="*60)
    print("FAIR COMPARISON WITH OTHER METHODS (all using window_size=3)")
    print("="*60)
    print("Previous Best Results with window_size=3:")
    print("  SuperDialseg: Sentence-BERT F1=0.571 (window_size=3)")
    print("  SuperDialseg: Sliding Window F1=0.560 (window_size=3)")
    print("  TIAGE: Sentence-BERT F1=0.222 (window_size=3)")
    print("  TIAGE: Sliding Window F1=0.219 (window_size=3)")
    print(f"\nBest Small Model Results with window_size=3:")
    print(f"  SuperDialseg: {best_super_config} F1={best_super_f1:.3f}")
    print(f"  TIAGE: {best_tiage_config} F1={best_tiage_f1:.3f}")
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print("Now comparing apples to apples with same window size (3)")
    print("Results show whether small instruct models are truly competitive")


if __name__ == "__main__":
    main()