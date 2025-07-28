#!/usr/bin/env python3
"""Test instruct LLM detector on sample dialogues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.instruct_llm_detector import FastInstructDetector
from evaluation.metrics import SegmentationMetrics

# Sample dialogue with clear topic changes
test_messages = [
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm, about 75 degrees."},
    {"role": "user", "content": "Perfect for a walk in the park!"},
    {"role": "assistant", "content": "Yes, great weather for outdoor activities."},
    # Topic change here (index 3)
    {"role": "user", "content": "Can you help me debug this Python code?"},
    {"role": "assistant", "content": "Of course! Please share the code and error."},
    {"role": "user", "content": "I'm getting a KeyError in my dictionary."},
    {"role": "assistant", "content": "KeyErrors occur when accessing non-existent keys."},
    # Topic change here (index 7)
    {"role": "user", "content": "What's a good recipe for chocolate cake?"},
    {"role": "assistant", "content": "Here's a simple chocolate cake recipe..."},
]

expected_boundaries = [3, 7]


def test_detector(detector_name, detector):
    """Test a detector and print results."""
    print(f"\nTesting {detector_name}:")
    print("-" * 50)
    
    # Detect boundaries
    predicted = detector.detect_boundaries(test_messages)
    
    # Calculate metrics
    metrics = SegmentationMetrics()
    results = metrics.calculate_exact_metrics(
        predicted, 
        expected_boundaries,
        len(test_messages)
    )
    
    print(f"Expected boundaries: {expected_boundaries}")
    print(f"Predicted boundaries: {predicted}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1: {results['f1']:.3f}")
    
    return results


def main():
    print("Testing Instruct LLM Detectors")
    print("=" * 50)
    
    # Test different configurations
    detectors = [
        # Fast T5-based detector
        ("Flan-T5-Small (t=0.5)", 
         lambda: FastInstructDetector(
             model_name="google/flan-t5-small",
             threshold=0.5
         )),
        
        ("Flan-T5-Small (t=0.7)", 
         lambda: FastInstructDetector(
             model_name="google/flan-t5-small",
             threshold=0.7
         )),
        
        # You can add more models here
        # ("Flan-T5-Base (t=0.5)", 
        #  lambda: FastInstructDetector(
        #      model_name="google/flan-t5-base",
        #      threshold=0.5
        #  )),
    ]
    
    results = {}
    for name, detector_factory in detectors:
        try:
            detector = detector_factory()
            result = test_detector(name, detector)
            results[name] = result
        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"Best F1: {best_f1[0]} = {best_f1[1]['f1']:.3f}")


if __name__ == "__main__":
    main()