#!/usr/bin/env python3
"""Analyze available datasets for fine-tuning small models."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.tiage_loader import TiageDatasetLoader


def analyze_datasets():
    """Analyze available datasets for fine-tuning."""
    print("Analyzing Available Datasets for Fine-tuning")
    print("="*60)
    
    # 1. SuperDialseg Dataset
    print("\n1. SuperDialseg Dataset:")
    print("-"*40)
    try:
        loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
        
        # Count conversations and boundaries
        splits = ['train', 'dev', 'test']
        total_boundaries = 0
        total_messages = 0
        
        for split in splits:
            conversations = loader.load_conversations(
                Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
                split=split
            )
            
            split_boundaries = 0
            split_messages = 0
            
            for conv in conversations:
                messages, boundaries = loader.parse_conversation(conv)
                split_boundaries += len(boundaries)
                split_messages += len(messages)
            
            total_boundaries += split_boundaries
            total_messages += split_messages
            
            print(f"  {split}: {len(conversations)} conversations, "
                  f"{split_boundaries} boundaries, {split_messages} messages")
        
        print(f"  Total: {total_boundaries} boundaries, {total_messages} messages")
        
    except Exception as e:
        print(f"  Error loading SuperDialseg: {e}")
    
    # 2. TIAGE Dataset
    print("\n2. TIAGE Dataset:")
    print("-"*40)
    try:
        loader = TiageDatasetLoader()
        datasets = ['wikipedia', 'nytimes', 'cnn', 'podcast']
        
        total_boundaries = 0
        total_messages = 0
        
        for dataset_name in datasets:
            conversations = loader.load_dataset(dataset_name)
            dataset_boundaries = 0
            dataset_messages = 0
            
            for conv in conversations:
                messages, boundaries = loader.parse_conversation(conv)
                dataset_boundaries += len(boundaries)
                dataset_messages += len(messages)
            
            total_boundaries += dataset_boundaries
            total_messages += dataset_messages
            
            print(f"  {dataset_name}: {len(conversations)} conversations, "
                  f"{dataset_boundaries} boundaries, {dataset_messages} messages")
        
        print(f"  Total: {total_boundaries} boundaries, {total_messages} messages")
        
    except Exception as e:
        print(f"  Error loading TIAGE: {e}")
    
    # 3. Training Data Format Options
    print("\n3. Training Data Format Options:")
    print("-"*40)
    print("  a) Classification format (boundary/no-boundary):")
    print("     Input: Two messages or windows")
    print("     Output: 0 or 1")
    print("  b) Regression format (drift score):")
    print("     Input: Two messages or windows")
    print("     Output: 0.0 to 1.0")
    print("  c) Sequence labeling format:")
    print("     Input: Full conversation")
    print("     Output: Boundary positions")
    
    # 4. Fine-tuning Approaches
    print("\n4. Fine-tuning Approaches for Small Models:")
    print("-"*40)
    print("  a) LoRA (Low-Rank Adaptation):")
    print("     - Adds small trainable matrices")
    print("     - Minimal memory overhead")
    print("     - Works well for task-specific adaptation")
    print("  b) QLoRA (Quantized LoRA):")
    print("     - 4-bit quantization + LoRA")
    print("     - Even smaller memory footprint")
    print("     - Good for very small models")
    print("  c) Adapter layers:")
    print("     - Add small bottleneck layers")
    print("     - Keep base model frozen")
    print("  d) Prompt tuning:")
    print("     - Learn soft prompts")
    print("     - No model weight changes")
    
    # 5. Recommended Approach
    print("\n5. Recommended Approach:")
    print("-"*40)
    print("  Model: TinyLlama (637 MB) or qwen2:0.5b (352 MB)")
    print("  Method: QLoRA for minimal memory usage")
    print("  Task: Binary classification (boundary/no-boundary)")
    print("  Window size: Start with 1, then try 3")
    print("  Training examples: ~10,000+ from combined datasets")


def create_training_examples():
    """Create training examples from datasets."""
    print("\n\nCreating Sample Training Examples")
    print("="*60)
    
    # Load a few examples from SuperDialseg
    loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
    conversations = loader.load_conversations(
        Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
        split='train'
    )[:2]  # Just 2 conversations for examples
    
    examples = []
    
    for conv in conversations:
        messages, boundaries = loader.parse_conversation(conv)
        
        # Create examples for each position
        for i in range(1, len(messages)):
            is_boundary = i in boundaries
            
            # Window size 1 example
            example = {
                "instruction": "Determine if there is a topic change between these messages. Output 1 for topic change, 0 for same topic.",
                "input": f"Message 1: {messages[i-1]['content']}\nMessage 2: {messages[i]['content']}",
                "output": "1" if is_boundary else "0"
            }
            examples.append(example)
            
            if len(examples) >= 5:
                break
        
        if len(examples) >= 5:
            break
    
    print("\nSample Training Examples (Window Size 1):")
    print("-"*40)
    for i, ex in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"Input: {ex['input'][:100]}...")
        print(f"Output: {ex['output']}")
        print(f"Is boundary: {'Yes' if ex['output'] == '1' else 'No'}")
    
    # Window size 3 examples
    print("\n\nSample Training Examples (Window Size 3):")
    print("-"*40)
    
    window_examples = []
    window_size = 3
    
    for conv in conversations:
        messages, boundaries = loader.parse_conversation(conv)
        
        for i in range(window_size, len(messages)):
            is_boundary = i in boundaries
            
            # Get windows
            window_a = messages[max(0, i-window_size):i]
            window_b = messages[i:min(len(messages), i+window_size)]
            
            window_a_text = " ".join([m['content'] for m in window_a])
            window_b_text = " ".join([m['content'] for m in window_b])
            
            example = {
                "instruction": "Compare the topics between Window A and Window B. Output 1 if topics are different, 0 if same.",
                "input": f"Window A: {window_a_text}\nWindow B: {window_b_text}",
                "output": "1" if is_boundary else "0"
            }
            window_examples.append(example)
            
            if len(window_examples) >= 3:
                break
        
        if len(window_examples) >= 3:
            break
    
    for i, ex in enumerate(window_examples):
        print(f"\nExample {i+1}:")
        print(f"Input: {ex['input'][:150]}...")
        print(f"Output: {ex['output']}")
    
    return examples, window_examples


if __name__ == "__main__":
    analyze_datasets()
    examples_w1, examples_w3 = create_training_examples()
    
    print("\n\nNext Steps:")
    print("="*60)
    print("1. Create full training dataset in JSON format")
    print("2. Use Hugging Face transformers with PEFT for fine-tuning")
    print("3. Start with TinyLlama + QLoRA")
    print("4. Train on window_size=1 first (simpler task)")
    print("5. Evaluate on held-out test sets")