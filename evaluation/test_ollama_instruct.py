#!/usr/bin/env python3
"""Test Ollama instruct models for topic detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector, FastOllamaDetector
from evaluation.metrics import SegmentationMetrics

# Test dialogue with clear topic changes
test_dialogue = [
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm, about 75 degrees Fahrenheit."},
    {"role": "user", "content": "Perfect for a walk in the park!"},
    {"role": "assistant", "content": "Yes, it's ideal weather for outdoor activities."},
    # Topic change here (boundary at index 3)
    {"role": "user", "content": "Can you help me debug this Python code?"},
    {"role": "assistant", "content": "Of course! Please share the code and the error you're encountering."},
    {"role": "user", "content": "I'm getting a KeyError when accessing my dictionary."},
    {"role": "assistant", "content": "KeyErrors occur when you try to access a key that doesn't exist in the dictionary."},
    # Topic change here (boundary at index 7)
    {"role": "user", "content": "What's a good recipe for chocolate cake?"},
    {"role": "assistant", "content": "Here's a simple chocolate cake recipe that's always a hit!"},
]

expected_boundaries = [3, 7]


def test_model(model_name, threshold=0.7, window_size=1, verbose=True):
    """Test a specific model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} (threshold={threshold}, window={window_size})")
    print('='*60)
    
    try:
        detector = OllamaInstructDetector(
            model_name=model_name,
            threshold=threshold,
            window_size=window_size,
            verbose=verbose
        )
        
        predicted = detector.detect_boundaries(test_dialogue)
        
        # Calculate metrics
        metrics = SegmentationMetrics()
        results = metrics.calculate_exact_metrics(
            predicted,
            expected_boundaries,
            len(test_dialogue)
        )
        
        print(f"\nResults:")
        print(f"Expected boundaries: {expected_boundaries}")
        print(f"Predicted boundaries: {predicted}")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("Testing Ollama Instruct Models for Topic Detection")
    print("="*60)
    
    # Test configurations
    test_configs = [
        # Model name, threshold, window_size
        ("mistral:instruct", 0.7, 1),
        ("mistral:instruct", 0.6, 1),
        ("llama3:instruct", 0.7, 1),
        ("phi3:instruct", 0.7, 1),  # Faster model
        # Test with context window
        ("mistral:instruct", 0.7, 3),
    ]
    
    results = {}
    
    print("\nDetected models: mistral:instruct, llama3:instruct")
    print("Starting tests...")
    
    for model, threshold, window in test_configs:
        config_name = f"{model}_t{threshold}_w{window}"
        result = test_model(model, threshold, window, verbose=False)
        if result:
            results[config_name] = result
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Best F1
        best_config = max(results.items(), key=lambda x: x[1]['f1'])
        print(f"\nBest F1 Score: {best_config[0]}")
        print(f"F1: {best_config[1]['f1']:.3f}")
        print(f"Precision: {best_config[1]['precision']:.3f}")
        print(f"Recall: {best_config[1]['recall']:.3f}")
        
        # All results
        print("\nAll Results:")
        for config, result in sorted(results.items()):
            print(f"{config:30} F1={result['f1']:.3f} P={result['precision']:.3f} R={result['recall']:.3f}")


if __name__ == "__main__":
    main()