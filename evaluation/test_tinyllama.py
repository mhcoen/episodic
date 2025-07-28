#!/usr/bin/env python3
"""Test TinyLlama model for topic detection."""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def test_tinyllama(threshold=0.7):
    """Test TinyLlama model."""
    print("\n" + "="*50)
    print("Testing tinyllama:latest")
    print("="*50)
    
    predicted = []
    drift_scores = []
    
    # Test each transition
    for i in range(1, len(test_messages)):
        prev = test_messages[i-1]
        curr = test_messages[i]
        
        # Create drift prompt - using instruct-style prompting
        prompt = f"""Analyze topic drift between two messages.

Message 1 ({prev['role']}): {prev['content']}
Message 2 ({curr['role']}): {curr['content']}

Rate drift from 0.0 (same topic) to 1.0 (completely different).
Respond with ONLY a decimal number.

Score:"""
        
        # Call model
        start_time = time.time()
        try:
            result = subprocess.run(
                ['ollama', 'run', 'tinyllama:latest', prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                # Extract number from response
                try:
                    # Try to find a decimal number in the response
                    import re
                    numbers = re.findall(r'\d*\.?\d+', response)
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                    else:
                        score = 0.5  # Default if can't parse
                except:
                    score = 0.5
                    
                elapsed = time.time() - start_time
                drift_scores.append(score)
                
                print(f"  {i-1}→{i}: score={score:.3f} (time={elapsed:.2f}s)")
                
                if score >= threshold:
                    predicted.append(i-1)
            else:
                print(f"  {i-1}→{i}: ERROR - {result.stderr}")
                drift_scores.append(0.5)
                
        except subprocess.TimeoutExpired:
            print(f"  {i-1}→{i}: TIMEOUT")
            drift_scores.append(0.5)
        except Exception as e:
            print(f"  {i-1}→{i}: ERROR - {e}")
            drift_scores.append(0.5)
    
    # Calculate metrics
    metrics = SegmentationMetrics()
    results = metrics.calculate_exact_metrics(
        predicted,
        expected_boundaries,
        len(test_messages)
    )
    
    print(f"\nResults:")
    print(f"Predicted: {predicted} (expected: {expected_boundaries})")
    print(f"F1 Score: {results['f1']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"Drift scores: {[f'{s:.2f}' for s in drift_scores]}")
    
    return results['f1'], drift_scores


def main():
    print("Testing TinyLlama for Topic Detection")
    print("="*50)
    
    # Check model size
    result = subprocess.run(
        ['ollama', 'list'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'tinyllama' in line.lower():
                print(f"Model info: {line}")
                break
    
    # Test with different thresholds
    thresholds = [0.7, 0.6, 0.5]
    
    for threshold in thresholds:
        f1, scores = test_tinyllama(threshold)
        print(f"\nThreshold {threshold}: F1={f1:.3f}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("TinyLlama is a 1.1B parameter model (637 MB)")
    print("It's the smallest available instruct-capable model")
    print("\nSize comparison:")
    print("- TinyLlama: 637 MB")
    print("- qwen2:1.5b: 934 MB")
    print("- gemma:2b-instruct: 1.6 GB")
    print("- phi3:instruct: 2.2 GB")
    print("- mistral:instruct: 4.1 GB")
    print("\nTinyLlama is ~68% the size of qwen2:1.5b")
    print("and only ~15% the size of mistral:instruct!")


if __name__ == "__main__":
    main()