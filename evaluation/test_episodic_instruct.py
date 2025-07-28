#!/usr/bin/env python3
"""Test Episodic instruct detector with actual LLM calls."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up minimal Episodic environment
os.environ['EPISODIC_CONFIG_PATH'] = str(Path.home() / '.episodic' / 'config.json')

from evaluation.episodic_instruct_detector import EpisodicInstructDetector
from evaluation.metrics import SegmentationMetrics

# Test dialogue with clear topic boundaries
test_dialogue = [
    {"role": "user", "content": "What's the weather forecast for tomorrow?"},
    {"role": "assistant", "content": "Tomorrow will be partly cloudy with a high of 72°F and a low of 58°F. There's a 20% chance of rain in the afternoon."},
    {"role": "user", "content": "Should I bring an umbrella?"},
    {"role": "assistant", "content": "Given the low 20% chance of rain, an umbrella isn't necessary, but you might want to bring a light jacket for the cooler evening."},
    # Topic change here (boundary at index 3)
    {"role": "user", "content": "Can you explain how neural networks work?"},
    {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers."},
    {"role": "user", "content": "What's the difference between CNN and RNN?"},
    {"role": "assistant", "content": "CNNs (Convolutional Neural Networks) are designed for spatial data like images, while RNNs (Recurrent Neural Networks) are designed for sequential data like text or time series."},
    # Topic change here (boundary at index 7)
    {"role": "user", "content": "I need help planning a trip to Japan."},
    {"role": "assistant", "content": "I'd be happy to help! When are you planning to visit Japan, and what are your main interests - culture, food, nature, or technology?"},
]

expected_boundaries = [3, 7]


def test_simple_prompt():
    """Test with simple pairwise prompts."""
    print("\n=== Testing Simple Pairwise Prompt ===")
    
    # Test different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        detector = EpisodicInstructDetector(
            threshold=threshold,
            use_simple_prompt=True
        )
        
        predicted = detector.detect_boundaries(test_dialogue)
        
        metrics = SegmentationMetrics()
        results = metrics.calculate_exact_metrics(
            predicted,
            expected_boundaries,
            len(test_dialogue)
        )
        
        print(f"Expected: {expected_boundaries}")
        print(f"Predicted: {predicted}")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1: {results['f1']:.3f}")


def test_window_prompt():
    """Test with window-based context."""
    print("\n=== Testing Window-based Context ===")
    
    detector = EpisodicInstructDetector(
        threshold=0.7,
        window_size=3,
        use_simple_prompt=False
    )
    
    predicted = detector.detect_boundaries(test_dialogue)
    
    metrics = SegmentationMetrics()
    results = metrics.calculate_exact_metrics(
        predicted,
        expected_boundaries,
        len(test_dialogue)
    )
    
    print(f"Expected: {expected_boundaries}")
    print(f"Predicted: {predicted}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1: {results['f1']:.3f}")


def main():
    print("Testing Episodic Instruct LLM Detector")
    print("=" * 50)
    
    try:
        # Check if we can initialize the LLM
        from episodic.llm import LLM
        llm = LLM()
        print(f"Using model: {llm.model}")
        print(f"Provider: {llm.provider}")
        
        # Run tests
        test_simple_prompt()
        test_window_prompt()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Episodic is configured with a valid LLM provider.")
        print("You may need to run: episodic --set")


if __name__ == "__main__":
    main()