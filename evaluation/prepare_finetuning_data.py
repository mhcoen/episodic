#!/usr/bin/env python3
"""Prepare training data for fine-tuning small models on topic detection."""

import json
import random
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.tiage_loader import TiageDatasetLoader


def create_window_size_1_examples(messages: List[Dict[str, Any]], 
                                  boundaries: List[int]) -> List[Dict[str, Any]]:
    """Create training examples for window size 1 (pairwise comparison)."""
    examples = []
    
    for i in range(1, len(messages)):
        is_boundary = i in boundaries
        
        # Create example
        example = {
            "instruction": "Determine if there is a topic change between these two messages. Output only '1' for topic change or '0' for same topic.",
            "input": f"Message 1: {messages[i-1]['content']}\nMessage 2: {messages[i]['content']}",
            "output": "1" if is_boundary else "0",
            "metadata": {
                "is_boundary": is_boundary,
                "position": i
            }
        }
        examples.append(example)
    
    return examples


def create_window_size_3_examples(messages: List[Dict[str, Any]], 
                                  boundaries: List[int],
                                  window_size: int = 3) -> List[Dict[str, Any]]:
    """Create training examples for window size 3."""
    examples = []
    
    for i in range(window_size, len(messages)):
        is_boundary = i in boundaries
        
        # Get windows
        window_a = messages[max(0, i-window_size):i]
        window_b = messages[i:min(len(messages), i+window_size)]
        
        # Format windows - just concatenate for simplicity
        window_a_text = " ".join([m['content'] for m in window_a])
        window_b_text = " ".join([m['content'] for m in window_b])
        
        # Truncate if too long
        max_len = 500
        if len(window_a_text) > max_len:
            window_a_text = window_a_text[:max_len] + "..."
        if len(window_b_text) > max_len:
            window_b_text = window_b_text[:max_len] + "..."
        
        example = {
            "instruction": "Compare the topics between Window A (before) and Window B (after). Output only '1' if the topics are different or '0' if they are the same.",
            "input": f"Window A: {window_a_text}\n\nWindow B: {window_b_text}",
            "output": "1" if is_boundary else "0",
            "metadata": {
                "is_boundary": is_boundary,
                "position": i,
                "window_size": window_size
            }
        }
        examples.append(example)
    
    return examples


def prepare_superdialseg_data(split: str = 'train', 
                              window_size: int = 1) -> List[Dict[str, Any]]:
    """Prepare training data from SuperDialseg dataset."""
    print(f"Loading SuperDialseg {split} split...")
    
    loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
    
    # Map split names
    split_map = {'dev': 'validation', 'valid': 'validation'}
    actual_split = split_map.get(split, split)
    
    conversations = loader.load_conversations(
        Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
        split=actual_split
    )
    
    all_examples = []
    
    for i, conv in enumerate(conversations):
        if i % 100 == 0:
            print(f"  Processing conversation {i}/{len(conversations)}...")
        
        messages, boundaries = loader.parse_conversation(conv)
        
        if window_size == 1:
            examples = create_window_size_1_examples(messages, boundaries)
        else:
            examples = create_window_size_3_examples(messages, boundaries, window_size)
        
        # Add dataset source
        for ex in examples:
            ex['metadata']['dataset'] = 'superdialseg'
            ex['metadata']['conversation_id'] = i
        
        all_examples.extend(examples)
    
    return all_examples


def prepare_tiage_data(dataset_name: str = 'wikipedia', 
                       window_size: int = 1) -> List[Dict[str, Any]]:
    """Prepare training data from TIAGE dataset."""
    print(f"Loading TIAGE {dataset_name} dataset...")
    
    dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/tiage")
    loader = TiageDatasetLoader(dataset_path)
    
    conversations = loader.load_dataset(dataset_name)
    
    all_examples = []
    
    for i, conv in enumerate(conversations):
        if i % 10 == 0:
            print(f"  Processing conversation {i}/{len(conversations)}...")
        
        messages, boundaries = loader.parse_conversation(conv)
        
        if window_size == 1:
            examples = create_window_size_1_examples(messages, boundaries)
        else:
            examples = create_window_size_3_examples(messages, boundaries, window_size)
        
        # Add dataset source
        for ex in examples:
            ex['metadata']['dataset'] = f'tiage_{dataset_name}'
            ex['metadata']['conversation_id'] = i
        
        all_examples.extend(examples)
    
    return all_examples


def balance_dataset(examples: List[Dict[str, Any]], 
                    balance_ratio: float = 1.0) -> List[Dict[str, Any]]:
    """Balance positive and negative examples."""
    positive = [ex for ex in examples if ex['metadata']['is_boundary']]
    negative = [ex for ex in examples if not ex['metadata']['is_boundary']]
    
    print(f"  Original: {len(positive)} positive, {len(negative)} negative")
    
    # Usually there are many more negative examples
    # Balance by downsampling negatives
    if balance_ratio > 0:
        target_negative = int(len(positive) * balance_ratio)
        if target_negative < len(negative):
            negative = random.sample(negative, target_negative)
    
    balanced = positive + negative
    random.shuffle(balanced)
    
    print(f"  Balanced: {len(positive)} positive, {len(negative)} negative")
    
    return balanced


def save_dataset(examples: List[Dict[str, Any]], 
                 output_path: Path,
                 format: str = 'jsonl'):
    """Save dataset in specified format."""
    
    # Remove metadata from saved version to reduce size
    clean_examples = []
    for ex in examples:
        clean_ex = {
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": ex["output"]
        }
        clean_examples.append(clean_ex)
    
    if format == 'jsonl':
        with open(output_path, 'w') as f:
            for ex in clean_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    else:  # json
        with open(output_path, 'w') as f:
            json.dump(clean_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(clean_examples)} examples to {output_path}")


def main():
    """Prepare complete training datasets."""
    print("Preparing Fine-tuning Datasets for Topic Detection")
    print("="*60)
    
    output_dir = Path("/Users/mhcoen/proj/episodic/evaluation/finetuning_data")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Prepare Window Size 1 Dataset
    print("\n1. Preparing Window Size 1 Dataset")
    print("-"*40)
    
    # Combine SuperDialseg and TIAGE data
    train_examples = []
    
    # SuperDialseg train
    try:
        train_examples.extend(prepare_superdialseg_data('train', window_size=1))
    except Exception as e:
        print(f"  Error loading SuperDialseg: {e}")
    
    # TIAGE datasets
    for dataset in ['wikipedia', 'nytimes']:
        try:
            train_examples.extend(prepare_tiage_data(dataset, window_size=1))
        except Exception as e:
            print(f"  Error loading TIAGE {dataset}: {e}")
    
    # Balance and save
    if train_examples:
        print(f"\nTotal training examples: {len(train_examples)}")
        balanced_train = balance_dataset(train_examples, balance_ratio=2.0)  # 2:1 negative:positive
        
        save_dataset(balanced_train, 
                     output_dir / "train_window1.jsonl",
                     format='jsonl')
    
    # 2. Prepare validation set
    print("\n2. Preparing Validation Set")
    print("-"*40)
    
    val_examples = []
    
    # SuperDialseg validation
    try:
        val_examples.extend(prepare_superdialseg_data('valid', window_size=1))
    except Exception as e:
        print(f"  Error loading SuperDialseg validation: {e}")
    
    # TIAGE CNN for validation
    try:
        val_examples.extend(prepare_tiage_data('cnn', window_size=1))
    except Exception as e:
        print(f"  Error loading TIAGE CNN: {e}")
    
    if val_examples:
        print(f"\nTotal validation examples: {len(val_examples)}")
        balanced_val = balance_dataset(val_examples, balance_ratio=2.0)
        
        save_dataset(balanced_val[:1000],  # Limit validation size
                     output_dir / "val_window1.jsonl",
                     format='jsonl')
    
    # 3. Create sample for window size 3 (smaller dataset due to complexity)
    print("\n3. Preparing Window Size 3 Sample Dataset")
    print("-"*40)
    
    window3_examples = []
    
    try:
        # Just use a subset for window size 3
        loader = SuperDialsegLoader("/Users/mhcoen/proj/episodic/datasets/superseg")
        conversations = loader.load_conversations(
            Path("/Users/mhcoen/proj/episodic/datasets/superseg"),
            split='train'
        )[:100]  # Just 100 conversations
        
        for i, conv in enumerate(conversations):
            messages, boundaries = loader.parse_conversation(conv)
            examples = create_window_size_3_examples(messages, boundaries)
            window3_examples.extend(examples)
    
        print(f"\nTotal window-3 examples: {len(window3_examples)}")
        balanced_window3 = balance_dataset(window3_examples, balance_ratio=2.0)
        
        save_dataset(balanced_window3,
                     output_dir / "train_window3_sample.jsonl",
                     format='jsonl')
    
    except Exception as e:
        print(f"  Error creating window-3 dataset: {e}")
    
    # 4. Print statistics
    print("\n\nDataset Statistics:")
    print("="*60)
    
    for file in output_dir.glob("*.jsonl"):
        with open(file) as f:
            count = sum(1 for _ in f)
        print(f"  {file.name}: {count} examples")
    
    print("\n\nNext Steps:")
    print("="*60)
    print("1. Install fine-tuning dependencies:")
    print("   pip install transformers peft accelerate bitsandbytes")
    print("2. Use the prepared JSONL files for fine-tuning")
    print("3. Start with window_size=1 dataset (simpler task)")
    print("4. Fine-tune TinyLlama or qwen2:0.5b with QLoRA")


if __name__ == "__main__":
    main()