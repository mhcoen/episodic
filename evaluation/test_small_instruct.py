#!/usr/bin/env python3
"""Test small instruct models for speed and accuracy."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.metrics import SegmentationMetrics

# Test dialogue
test_messages = [
    {"role": "user", "content": "How's the weather today?"},
    {"role": "assistant", "content": "It's sunny and 75 degrees."},
    {"role": "user", "content": "Great day for a picnic!"},
    # Topic change
    {"role": "user", "content": "Can you help with Python code?"},
    {"role": "assistant", "content": "Sure, I'd be happy to help with Python."},
    {"role": "user", "content": "I have a sorting error."},
    # Topic change  
    {"role": "user", "content": "What's a good pasta recipe?"},
    {"role": "assistant", "content": "I recommend a simple carbonara."},
]

expected_boundaries = [2, 5]


def test_model(model_name, threshold=0.7):
    """Test a model for speed and accuracy."""
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print('='*50)
    
    detector = OllamaInstructDetector(
        model_name=model_name,
        threshold=threshold,
        window_size=1,
        verbose=False
    )
    
    # Time the detection
    start_time = time.time()
    predicted = detector.detect_boundaries(test_messages)
    total_time = time.time() - start_time
    
    # Calculate metrics
    metrics = SegmentationMetrics()
    results = metrics.calculate_exact_metrics(
        predicted,
        expected_boundaries,
        len(test_messages)
    )
    
    print(f"Time: {total_time:.2f}s ({total_time/(len(test_messages)-1):.2f}s per transition)")
    print(f"Predicted: {predicted} (expected: {expected_boundaries})")
    print(f"F1 Score: {results['f1']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    
    return total_time, results['f1']


def main():
    print("Testing Small Instruct Models")
    print("="*50)
    
    models_to_test = [
        ("gemma:2b-instruct", 0.7),
        ("gemma:2b-instruct", 0.6),
        ("phi3:instruct", 0.7),
    ]
    
    # Also test if these are available
    optional_models = [
        ("tinyllama:instruct", 0.7),
        ("qwen:0.5b-instruct", 0.7),
    ]
    
    results = {}
    
    # Test available models
    for model, threshold in models_to_test:
        try:
            time_taken, f1 = test_model(model, threshold)
            results[f"{model}_t{threshold}"] = (time_taken, f1)
        except Exception as e:
            print(f"Error testing {model}: {e}")
    
    # Try optional models
    print("\n" + "="*50)
    print("Checking for smaller models...")
    print("="*50)
    
    for model, threshold in optional_models:
        print(f"\nTrying {model}...")
        try:
            time_taken, f1 = test_model(model, threshold)
            results[f"{model}_t{threshold}"] = (time_taken, f1)
        except:
            print(f"{model} not installed. Install with: ollama pull {model}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY - Speed vs Accuracy")
    print("="*50)
    print(f"{'Model':<30} {'Time (s)':<10} {'F1 Score':<10}")
    print("-"*50)
    
    for config, (time_taken, f1) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"{config:<30} {time_taken:<10.2f} {f1:<10.3f}")
    
    print("\nFor comparison:")
    print("- Sentence-BERT: ~0.01s per transition")
    print("- Sliding Window: ~0.001s per transition")


if __name__ == "__main__":
    main()