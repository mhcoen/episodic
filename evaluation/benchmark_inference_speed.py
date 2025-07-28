#!/usr/bin/env python3
"""Benchmark inference speed of different topic detection methods."""

import time
import torch
import numpy as np
from pathlib import Path
import subprocess
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.sliding_window_detector import SlidingWindowDetector
from evaluation.sentence_bert_detector import SentenceBERTDetector


class TopicDetectionModel(nn.Module):
    """Fine-tuned model class."""
    def __init__(self, model_name="microsoft/xtremedistil-l6-h256-uncased", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def create_test_messages(n_messages=100):
    """Create test messages for benchmarking."""
    messages = []
    topics = [
        "The weather is really nice today. Perfect for outdoor activities.",
        "I need help with Python programming. Debugging some code.",
        "Looking into social security benefits for retirement planning.",
        "Discussing the latest movie releases and entertainment news.",
        "Working on machine learning models for text classification."
    ]
    
    for i in range(n_messages):
        topic_idx = i // 20  # Change topic every 20 messages
        messages.append({
            'role': 'user' if i % 2 == 0 else 'assistant',
            'content': topics[topic_idx % len(topics)] + f" Message {i}."
        })
    
    return messages


def benchmark_finetuned_model(messages, device='cpu'):
    """Benchmark fine-tuned XtremDistil model."""
    print("\n1. Fine-tuned XtremDistil (13M params)")
    print("-" * 40)
    
    # Load model
    model_path = Path("evaluation/finetuned_models/topic_detector_full.pt")
    if not model_path.exists():
        print("  Model not found. Using placeholder timing.")
        return 0.1
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
    model = TopicDetectionModel()
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Benchmark
    start_time = time.time()
    boundaries = []
    
    with torch.no_grad():
        for i in range(1, len(messages)):
            text = f"Determine if there is a topic change between these two messages. Output only '1' for topic change or '0' for same topic.\n\nMessage 1: {messages[i-1]['content']}\nMessage 2: {messages[i]['content']}"
            
            encoding = tokenizer(text, truncation=True, padding='max_length', 
                               max_length=256, return_tensors='pt')
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()
            
            if pred == 1:
                boundaries.append(i)
    
    inference_time = time.time() - start_time
    
    print(f"  Total time: {inference_time:.3f}s")
    print(f"  Time per comparison: {inference_time/(len(messages)-1)*1000:.2f}ms")
    print(f"  Comparisons per second: {(len(messages)-1)/inference_time:.1f}")
    print(f"  Boundaries detected: {len(boundaries)}")
    
    return inference_time


def benchmark_sentence_bert(messages):
    """Benchmark Sentence-BERT detector."""
    print("\n2. Sentence-BERT (384-dim embeddings)")
    print("-" * 40)
    
    detector = SentenceBERTDetector(threshold=0.25, window_size=3)
    
    start_time = time.time()
    boundaries = detector.detect_boundaries(messages)
    inference_time = time.time() - start_time
    
    print(f"  Total time: {inference_time:.3f}s")
    print(f"  Time per message: {inference_time/len(messages)*1000:.2f}ms")
    print(f"  Messages per second: {len(messages)/inference_time:.1f}")
    print(f"  Boundaries detected: {len(boundaries)}")
    
    return inference_time


def benchmark_sliding_window(messages):
    """Benchmark sliding window detector."""
    print("\n3. Sliding Window (TF-IDF)")
    print("-" * 40)
    
    detector = SlidingWindowDetector(threshold=0.4, window_size=3)
    
    start_time = time.time()
    boundaries = detector.detect_boundaries(messages)
    inference_time = time.time() - start_time
    
    print(f"  Total time: {inference_time:.3f}s")
    print(f"  Time per message: {inference_time/len(messages)*1000:.2f}ms")
    print(f"  Messages per second: {len(messages)/inference_time:.1f}")
    print(f"  Boundaries detected: {len(boundaries)}")
    
    return inference_time


def benchmark_ollama_model(messages, model_name="mistral:instruct"):
    """Benchmark Ollama instruct model."""
    print(f"\n4. Ollama {model_name}")
    print("-" * 40)
    
    # Test if model is available
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if model_name.split(':')[0] not in result.stdout:
            print(f"  Model {model_name} not available")
            return float('inf')
    except:
        print("  Ollama not available")
        return float('inf')
    
    # Benchmark a subset (too slow for all messages)
    n_test = min(10, len(messages) - 1)
    
    start_time = time.time()
    boundaries = []
    
    for i in range(1, n_test + 1):
        prompt = f"""Rate topic change from 0.0 to 1.0.

Message 1: {messages[i-1]['content']}
Message 2: {messages[i]['content']}

Output ONLY a number like 0.7
Nothing else. Just the number."""
        
        result = subprocess.run(
            ['ollama', 'run', model_name, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        try:
            score = float(result.stdout.strip())
            if score > 0.7:
                boundaries.append(i)
        except:
            pass
    
    inference_time = time.time() - start_time
    avg_time_per_comparison = inference_time / n_test
    
    # Extrapolate to full dataset
    estimated_total_time = avg_time_per_comparison * (len(messages) - 1)
    
    print(f"  Time for {n_test} comparisons: {inference_time:.3f}s")
    print(f"  Time per comparison: {avg_time_per_comparison*1000:.2f}ms")
    print(f"  Comparisons per second: {1/avg_time_per_comparison:.2f}")
    print(f"  Estimated total time: {estimated_total_time:.1f}s")
    
    return estimated_total_time


def main():
    print("Topic Detection Method Speed Benchmark")
    print("="*60)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create test data
    n_messages = 100
    print(f"\nTest dataset: {n_messages} messages")
    messages = create_test_messages(n_messages)
    
    # Run benchmarks
    results = {}
    
    results['finetuned'] = benchmark_finetuned_model(messages, device)
    results['sentence_bert'] = benchmark_sentence_bert(messages)
    results['sliding_window'] = benchmark_sliding_window(messages)
    
    # Optional: benchmark small ollama model
    # results['ollama_tiny'] = benchmark_ollama_model(messages, "qwen2:0.5b")
    
    # Summary
    print("\n" + "="*60)
    print("SPEED COMPARISON SUMMARY")
    print("="*60)
    
    # Sort by speed
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print(f"\nRanking (fastest to slowest):")
    baseline_time = sorted_results[0][1]
    
    for i, (method, time_taken) in enumerate(sorted_results, 1):
        speedup = baseline_time / time_taken if time_taken > 0 else 0
        print(f"{i}. {method}: {time_taken:.3f}s")
        if i > 1:
            print(f"   → {speedup:.1f}x slower than fastest")
    
    # Throughput comparison
    print(f"\nThroughput (messages/second):")
    for method, time_taken in sorted_results:
        if time_taken > 0:
            throughput = n_messages / time_taken
            print(f"  {method}: {throughput:.1f} msg/s")
    
    # Practical implications
    print("\n" + "="*60)
    print("PRACTICAL IMPLICATIONS")
    print("="*60)
    print("For a 1-hour conversation (~600 messages):")
    for method, time_taken in sorted_results:
        if time_taken > 0:
            estimated_time = (time_taken / n_messages) * 600
            print(f"  {method}: {estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()