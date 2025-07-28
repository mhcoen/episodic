#!/usr/bin/env python3
"""Test the smallest available models for topic detection."""

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


def test_model_as_instruct(model_name, threshold=0.7):
    """Test any model in instruct mode."""
    print(f"\n{'='*50}")
    print(f"Testing {model_name} as instruct model")
    print('='*50)
    
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
                ['ollama', 'run', model_name, prompt],
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
    print("Testing Smallest Models for Topic Detection")
    print("="*50)
    
    # Models to test (from smallest to larger)
    models_to_test = [
        ("qwen2:1.5b", 0.7, 934),  # 934 MB
        ("qwen2:1.5b", 0.6, 934),
        ("gemma:2b-instruct", 0.7, 1600),  # 1.6 GB
        ("phi3:mini", 0.7, 2200),  # 2.2 GB (same as phi3:instruct)
    ]
    
    results = {}
    
    for model, threshold, size_mb in models_to_test:
        print(f"\nModel size: {size_mb/1000:.1f} GB")
        try:
            f1, scores = test_model_as_instruct(model, threshold)
            results[f"{model}_t{threshold}"] = (f1, size_mb)
        except Exception as e:
            print(f"Error testing {model}: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY - Size vs Accuracy")
    print("="*50)
    print(f"{'Model':<30} {'Size (GB)':<10} {'F1 Score':<10}")
    print("-"*50)
    
    for config, (f1, size_mb) in sorted(results.items(), key=lambda x: x[1][1]):
        print(f"{config:<30} {size_mb/1000:<10.1f} {f1:<10.3f}")
    
    print("\nNotes:")
    print("- qwen2:1.5b (934 MB) is the smallest readily available model")
    print("- It's ~42% the size of phi3:instruct (2.2 GB)")
    print("- For comparison: mistral:instruct is 4.1 GB")
    
    # Try to find even smaller models
    print("\nChecking for other small models...")
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            small_models = []
            for line in lines:
                if any(term in line.lower() for term in ['0.5b', '1b', 'tiny', 'nano', 'micro']):
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[0]
                        size = parts[2]
                        small_models.append((name, size))
            
            if small_models:
                print("\nOther potentially small models found:")
                for name, size in small_models:
                    print(f"  {name}: {size}")
            else:
                print("\nNo smaller models found in local Ollama repository.")
                print("You could try pulling models like:")
                print("  - tinyllama (if it exists)")
                print("  - qwen:0.5b (if available)")
                print("  - Any quantized versions (q2, q3, q4)")
    except Exception as e:
        print(f"Error checking models: {e}")


if __name__ == "__main__":
    main()