#!/usr/bin/env python3
"""Evaluate Ollama instruct models on segmentation datasets."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.tiage_loader import TiageDatasetLoader
from evaluation.metrics import SegmentationMetrics, EvaluationResults


def evaluate_on_dataset(detector, dataset_loader, dataset_name, num_dialogues=20):
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
    print("Evaluating Ollama Instruct Models on Dialogue Datasets")
    print("="*60)
    
    # Models to test
    models = [
        ("mistral:instruct", 0.7),
        ("mistral:instruct", 0.6),
        ("mistral:instruct", 0.5),
        # Add more models as needed
        # ("llama3:instruct", 0.7),
        # ("phi3:instruct", 0.7),
    ]
    
    # Datasets
    datasets = [
        ("SuperDialseg", SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")),
        ("TIAGE", TiageDatasetLoader("/Users/mhcoen/proj/episodic/datasets/tiage/segmentation_file_test.json")),
    ]
    
    # Number of dialogues to test (use fewer for faster testing)
    num_dialogues = 10  # Reduced for faster testing
    
    all_results = []
    
    for model_name, threshold in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name} with threshold {threshold}")
        print('='*60)
        
        detector = OllamaInstructDetector(
            model_name=model_name,
            threshold=threshold,
            window_size=1,  # Simple pairwise comparison
            verbose=False
        )
        
        model_results = {
            'model': model_name,
            'threshold': threshold,
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
        
        all_results.append(model_results)
    
    # Save results
    output_file = Path("evaluation_results/ollama_instruct_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Best F1 Scores")
    print("="*60)
    
    for dataset_name, _ in datasets:
        best_f1 = 0
        best_config = None
        
        for result in all_results:
            if dataset_name in result['results']:
                f1 = result['results'][dataset_name]['metrics']['f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = f"{result['model']} (t={result['threshold']})"
        
        if best_config:
            print(f"{dataset_name}: {best_config} - F1={best_f1:.3f}")
    
    # Compare with other methods
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER METHODS")
    print("="*60)
    print("SuperDialseg Best F1 Scores:")
    print("  Sentence-BERT: 0.571")
    print("  Sliding Window: 0.560")
    print("  Ollama Instruct: Check results above")
    print("\nTIAGE Best F1 Scores:")
    print("  Sentence-BERT: 0.222")
    print("  Sliding Window: 0.219")
    print("  Ollama Instruct: Check results above")


if __name__ == "__main__":
    main()