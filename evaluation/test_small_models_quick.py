#!/usr/bin/env python3
"""Quick test of small models on a few dialogues."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ollama_instruct_detector import OllamaInstructDetector
from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.metrics import SegmentationMetrics


def test_on_sample_dialogues():
    """Test on just a few dialogues to verify it works."""
    
    # Load SuperDialseg
    loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
    # Load and convert to dialogue format
    conversations = loader.load_conversations(
        Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
        split='test'
    )[:3]
    
    dialogues = []
    for conv in conversations:
        messages = []
        boundaries = []
        prev_topic = None
        
        for i, turn in enumerate(conv['turns']):
            messages.append({
                'role': turn['role'],
                'content': turn['utterance']
            })
            
            if 'topic_id' in turn:
                if prev_topic is not None and turn['topic_id'] != prev_topic:
                    boundaries.append(i - 1)
                prev_topic = turn['topic_id']
        
        dialogues.append((messages, boundaries))
    
    models = [
        ("qwen2:0.5b", 0.5),
        ("tinyllama:latest", 0.7),
    ]
    
    for model_name, threshold in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name} with threshold {threshold}")
        print('='*60)
        
        detector = OllamaInstructDetector(
            model_name=model_name,
            threshold=threshold,
            window_size=1,
            verbose=True  # Show what's happening
        )
        
        metrics_calc = SegmentationMetrics()
        
        for i, (messages, gold_boundaries) in enumerate(dialogues):
            print(f"\nDialogue {i+1}: {len(messages)} messages")
            print(f"Gold boundaries: {gold_boundaries}")
            
            # Only test first few transitions
            test_messages = messages[:10]
            test_gold = [b for b in gold_boundaries if b < 9]
            
            start_time = time.time()
            predicted = detector.detect_boundaries(test_messages)
            elapsed = time.time() - start_time
            
            print(f"Predicted: {predicted}")
            print(f"Time: {elapsed:.2f}s")
            
            # Calculate metrics
            results = metrics_calc.calculate_exact_metrics(
                predicted,
                test_gold,
                len(test_messages)
            )
            
            print(f"F1: {results['f1']:.3f}")
            
            # Stop after first dialogue for quick test
            break


if __name__ == "__main__":
    test_on_sample_dialogues()