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
    
    # Warm up
    for i in range(3):
        text = f"Test message {i}"
        encoding = tokenizer(text, truncation=True, padding='max_length', 
                           max_length=256, return_tensors='pt')
        with torch.no_grad():
            _ = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
    
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
    """Benchmark Sentence-BERT embeddings + cosine similarity."""
    print("\n2. Sentence-BERT (all-MiniLM-L6-v2)")
    print("-" * 40)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract just the content
    texts = [msg['content'] for msg in messages]
    
    # Benchmark embedding computation
    start_time = time.time()
    
    # Encode all messages at once (more efficient)
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Compute cosine similarities
    boundaries = []
    for i in range(1, len(messages)):
        # Simple pairwise similarity
        similarity = torch.cosine_similarity(embeddings[i-1:i], embeddings[i:i+1])
        
        # Topic change if similarity is low
        if similarity.item() < 0.75:  # threshold
            boundaries.append(i)
    
    inference_time = time.time() - start_time
    
    print(f"  Total time: {inference_time:.3f}s")
    print(f"  Time per message: {inference_time/len(messages)*1000:.2f}ms")
    print(f"  Messages per second: {len(messages)/inference_time:.1f}")
    print(f"  Boundaries detected: {len(boundaries)}")
    
    return inference_time


def benchmark_tfidf(messages):
    """Benchmark TF-IDF + cosine similarity."""
    print("\n3. TF-IDF Vectorizer")
    print("-" * 40)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Extract texts
    texts = [msg['content'] for msg in messages]
    
    start_time = time.time()
    
    # Vectorize all texts
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Compute similarities
    boundaries = []
    for i in range(1, len(messages)):
        similarity = cosine_similarity(tfidf_matrix[i-1:i], tfidf_matrix[i:i+1])[0, 0]
        
        if similarity < 0.3:  # threshold
            boundaries.append(i)
    
    inference_time = time.time() - start_time
    
    print(f"  Total time: {inference_time:.3f}s")
    print(f"  Time per message: {inference_time/len(messages)*1000:.2f}ms")
    print(f"  Messages per second: {len(messages)/inference_time:.1f}")
    print(f"  Boundaries detected: {len(boundaries)}")
    
    return inference_time


def benchmark_ollama_model(messages, model_name="qwen2:0.5b"):
    """Benchmark Ollama instruct model."""
    print(f"\n4. Ollama {model_name} (LLM-based)")
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
        prompt = f"""Rate topic change 0-1:
Before: {messages[i-1]['content'][:50]}
After: {messages[i]['content'][:50]}
Number:"""
        
        result = subprocess.run(
            ['ollama', 'run', model_name, prompt],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        try:
            # Extract any number from response
            import re
            numbers = re.findall(r'\d*\.?\d+', result.stdout)
            if numbers:
                score = float(numbers[0])
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
    print(f"\nTest dataset: {n_messages} messages (5 topics, 20 messages each)")
    messages = create_test_messages(n_messages)
    
    # Run benchmarks
    results = {}
    
    try:
        results['TF-IDF'] = benchmark_tfidf(messages)
    except Exception as e:
        print(f"  Error: {e}")
        results['TF-IDF'] = float('inf')
    
    try:
        results['Fine-tuned XtremDistil'] = benchmark_finetuned_model(messages, device)
    except Exception as e:
        print(f"  Error: {e}")
        results['Fine-tuned XtremDistil'] = float('inf')
    
    try:
        results['Sentence-BERT'] = benchmark_sentence_bert(messages)
    except Exception as e:
        print(f"  Error: {e}")
        results['Sentence-BERT'] = float('inf')
    
    try:
        results['Ollama qwen2:0.5b'] = benchmark_ollama_model(messages, "qwen2:0.5b")
    except Exception as e:
        print(f"  Error: {e}")
        results['Ollama qwen2:0.5b'] = float('inf')
    
    # Summary
    print("\n" + "="*60)
    print("SPEED COMPARISON SUMMARY")
    print("="*60)
    
    # Sort by speed
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print(f"\nRanking (fastest to slowest):")
    baseline_time = sorted_results[0][1] if sorted_results[0][1] < float('inf') else 1.0
    
    for i, (method, time_taken) in enumerate(sorted_results, 1):
        if time_taken < float('inf'):
            speedup = time_taken / baseline_time
            print(f"{i}. {method}: {time_taken:.3f}s", end="")
            if i > 1:
                print(f" ({speedup:.1f}x slower)")
            else:
                print(" (baseline)")
        else:
            print(f"{i}. {method}: Not available")
    
    # Throughput comparison
    print(f"\nThroughput (messages/second):")
    for method, time_taken in sorted_results:
        if time_taken < float('inf'):
            throughput = n_messages / time_taken
            print(f"  {method}: {throughput:.1f} msg/s")
    
    # Latency comparison
    print(f"\nLatency (ms per comparison):")
    for method, time_taken in sorted_results:
        if time_taken < float('inf'):
            latency = (time_taken / (n_messages - 1)) * 1000
            print(f"  {method}: {latency:.1f} ms")
    
    # Practical implications
    print("\n" + "="*60)
    print("PRACTICAL IMPLICATIONS")
    print("="*60)
    print("For a 1-hour conversation (~600 messages):")
    for method, time_taken in sorted_results:
        if time_taken < float('inf'):
            estimated_time = (time_taken / n_messages) * 600
            if estimated_time < 1:
                print(f"  {method}: {estimated_time:.3f}s (real-time)")
            else:
                print(f"  {method}: {estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")
    
    print("\nFor real-time processing (30 messages/minute):")
    for method, time_taken in sorted_results:
        if time_taken < float('inf'):
            time_per_msg = time_taken / n_messages
            if time_per_msg < 2.0:  # 2 seconds per message for real-time
                print(f"  {method}: ✓ Can process in real-time")
            else:
                print(f"  {method}: ✗ Too slow for real-time")


if __name__ == "__main__":
    main()