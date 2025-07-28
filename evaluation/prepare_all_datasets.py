#!/usr/bin/env python3
"""Prepare training data from ALL available datasets."""

import json
import random
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.superdialseg_loader import SuperDialsegLoader
from evaluation.tiage_loader import TiageDatasetLoader
from evaluation.mp2d_loader import MP2DDatasetLoader


def create_training_examples(messages: List[Dict[str, Any]], 
                           boundaries: List[int],
                           window_size: int = 1) -> List[Dict[str, Any]]:
    """Create training examples from messages and boundaries."""
    examples = []
    
    if window_size == 1:
        # Pairwise comparison
        for i in range(1, len(messages)):
            is_boundary = i in boundaries
            
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


def load_superdialseg_data(base_path: Path, split: str = 'train') -> List[Dict[str, Any]]:
    """Load SuperDialseg dataset."""
    print(f"\nLoading SuperDialseg {split}...")
    loader = SuperDialsegLoader(str(base_path))
    
    # Map split names
    split_map = {'dev': 'validation', 'valid': 'validation'}
    actual_split = split_map.get(split, split)
    
    conversations = loader.load_conversations(base_path, split=actual_split)
    
    all_examples = []
    for i, conv in enumerate(conversations):
        if i % 500 == 0:
            print(f"  Processing conversation {i}/{len(conversations)}...")
        
        messages, boundaries = loader.parse_conversation(conv)
        examples = create_training_examples(messages, boundaries)
        
        for ex in examples:
            ex['metadata']['dataset'] = 'superdialseg'
            ex['metadata']['split'] = split
        
        all_examples.extend(examples)
    
    print(f"  Loaded {len(all_examples)} examples from SuperDialseg {split}")
    return all_examples


def load_tiage_data(base_path: Path) -> List[Dict[str, Any]]:
    """Load TIAGE dataset."""
    print(f"\nLoading TIAGE datasets...")
    
    # Check what's actually in the TIAGE directory
    tiage_files = list(base_path.glob("*.json"))
    print(f"  Found files: {[f.name for f in tiage_files]}")
    
    all_examples = []
    
    # Load train/validation/test splits if they exist
    for split in ['train', 'validation', 'test']:
        file_path = base_path / f"segmentation_file_{split}.json"
        if file_path.exists():
            print(f"  Loading TIAGE {split}...")
            with open(file_path) as f:
                data = json.load(f)
            
            # Parse TIAGE format
            for conv_id, conv_data in data.items():
                messages = []
                boundaries = []
                
                if 'utterances' in conv_data:
                    for i, utt in enumerate(conv_data['utterances']):
                        messages.append({
                            'role': 'user' if i % 2 == 0 else 'assistant',
                            'content': utt['text'] if isinstance(utt, dict) else utt
                        })
                    
                    # Get boundaries from segments
                    if 'segments' in conv_data:
                        for seg in conv_data['segments'][1:]:  # Skip first segment
                            if 'start_idx' in seg:
                                boundaries.append(seg['start_idx'])
                
                if messages:
                    examples = create_training_examples(messages, boundaries)
                    for ex in examples:
                        ex['metadata']['dataset'] = 'tiage'
                        ex['metadata']['split'] = split
                    all_examples.extend(examples)
    
    print(f"  Loaded {len(all_examples)} examples from TIAGE")
    return all_examples


def load_dialseg_data(base_path: Path) -> List[Dict[str, Any]]:
    """Load DialSeg711 dataset."""
    print(f"\nLoading DialSeg711...")
    
    test_file = base_path / "segmentation_file_test.json"
    if not test_file.exists():
        print("  DialSeg711 test file not found")
        return []
    
    # Load directly from JSON
    with open(test_file) as f:
        data = json.load(f)
    
    all_examples = []
    for conv_id, conv_data in data.items():
        messages = []
        boundaries = []
        
        if 'utterances' in conv_data:
            for i, utt in enumerate(conv_data['utterances']):
                messages.append({
                    'role': 'user' if i % 2 == 0 else 'assistant',
                    'content': utt['text'] if isinstance(utt, dict) else str(utt)
                })
            
            # Get boundaries from segments
            if 'segments' in conv_data:
                for seg in conv_data['segments'][1:]:  # Skip first segment
                    if 'start_idx' in seg:
                        boundaries.append(seg['start_idx'])
        
        if messages:
            examples = create_training_examples(messages, boundaries)
            for ex in examples:
                ex['metadata']['dataset'] = 'dialseg711'
                ex['metadata']['split'] = 'test'
            all_examples.extend(examples)
    
    print(f"  Loaded {len(all_examples)} examples from DialSeg711")
    return all_examples


def load_mp2d_data(base_path: Path) -> List[Dict[str, Any]]:
    """Load MP2D dataset samples."""
    print(f"\nLoading MP2D samples...")
    
    all_examples = []
    
    # Use MP2DLoader for proper parsing
    loader = MP2DDatasetLoader()
    
    # Load sample files
    for file_path in base_path.glob("MP2D*.json"):
        print(f"  Loading {file_path.name}...")
        
        try:
            # Load conversations
            conversations = loader.load_conversations(file_path)
            
            for conv in conversations:
                messages, boundaries = loader.parse_conversation(conv)
                
                if messages:
                    examples = create_training_examples(messages, boundaries)
                    for ex in examples:
                        ex['metadata']['dataset'] = 'mp2d'
                        ex['metadata']['split'] = 'sample'
                    all_examples.extend(examples)
        except Exception as e:
            print(f"    Error loading {file_path.name}: {e}")
    
    print(f"  Loaded {len(all_examples)} examples from MP2D samples")
    return all_examples


def balance_dataset(examples: List[Dict[str, Any]], 
                    balance_ratio: float = 1.5) -> List[Dict[str, Any]]:
    """Balance positive and negative examples."""
    positive = [ex for ex in examples if ex['metadata']['is_boundary']]
    negative = [ex for ex in examples if not ex['metadata']['is_boundary']]
    
    print(f"\n  Original distribution: {len(positive)} positive, {len(negative)} negative")
    
    # Balance by downsampling negatives
    target_negative = int(len(positive) * balance_ratio)
    if target_negative < len(negative):
        negative = random.sample(negative, target_negative)
    
    balanced = positive + negative
    random.shuffle(balanced)
    
    print(f"  Balanced distribution: {len(positive)} positive, {len(negative)} negative")
    print(f"  Total examples: {len(balanced)}")
    
    return balanced


def save_dataset(examples: List[Dict[str, Any]], 
                 output_path: Path,
                 format: str = 'jsonl'):
    """Save dataset in specified format."""
    # Remove metadata from saved version
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
    
    print(f"  Saved {len(clean_examples)} examples to {output_path}")


def main():
    """Prepare comprehensive training dataset from all sources."""
    print("Preparing Comprehensive Training Dataset")
    print("="*60)
    
    datasets_dir = Path("/Users/mhcoen/proj/episodic/datasets")
    output_dir = Path("/Users/mhcoen/proj/episodic/evaluation/finetuning_data")
    output_dir.mkdir(exist_ok=True)
    
    # Collect all training examples
    all_train_examples = []
    all_val_examples = []
    
    # 1. SuperDialseg
    try:
        superseg_path = datasets_dir / "superseg"
        all_train_examples.extend(load_superdialseg_data(superseg_path, 'train'))
        all_val_examples.extend(load_superdialseg_data(superseg_path, 'validation'))
    except Exception as e:
        print(f"  Error loading SuperDialseg: {e}")
    
    # 2. TIAGE
    try:
        tiage_path = datasets_dir / "tiage"
        tiage_examples = load_tiage_data(tiage_path)
        # Split TIAGE data
        train_size = int(0.8 * len(tiage_examples))
        all_train_examples.extend(tiage_examples[:train_size])
        all_val_examples.extend(tiage_examples[train_size:])
    except Exception as e:
        print(f"  Error loading TIAGE: {e}")
    
    # 3. DialSeg711
    try:
        dialseg_path = datasets_dir / "dialseg711"
        dialseg_examples = load_dialseg_data(dialseg_path)
        # Use DialSeg as additional validation data
        all_val_examples.extend(dialseg_examples)
    except Exception as e:
        print(f"  Error loading DialSeg711: {e}")
    
    # 4. MP2D samples
    try:
        mp2d_examples = load_mp2d_data(datasets_dir)
        # Add to training
        all_train_examples.extend(mp2d_examples)
    except Exception as e:
        print(f"  Error loading MP2D: {e}")
    
    # Print statistics
    print("\n\nDataset Statistics:")
    print("="*60)
    print(f"Total training examples: {len(all_train_examples)}")
    print(f"Total validation examples: {len(all_val_examples)}")
    
    # Count by dataset
    train_datasets = {}
    for ex in all_train_examples:
        ds = ex['metadata']['dataset']
        train_datasets[ds] = train_datasets.get(ds, 0) + 1
    
    print("\nTraining examples by dataset:")
    for ds, count in train_datasets.items():
        print(f"  {ds}: {count}")
    
    # Balance datasets
    print("\nBalancing datasets...")
    balanced_train = balance_dataset(all_train_examples, balance_ratio=1.5)
    balanced_val = balance_dataset(all_val_examples[:5000], balance_ratio=1.5)  # Limit val size
    
    # Save datasets
    print("\nSaving datasets...")
    save_dataset(balanced_train, output_dir / "train_all_datasets.jsonl")
    save_dataset(balanced_val, output_dir / "val_all_datasets.jsonl")
    
    # Create a test set from held-out data
    test_examples = all_val_examples[5000:6000] if len(all_val_examples) > 5000 else []
    if test_examples:
        save_dataset(test_examples, output_dir / "test_all_datasets.jsonl")
    
    print("\n\nFinal Dataset Summary:")
    print("="*60)
    print(f"Training set: {len(balanced_train)} examples")
    print(f"Validation set: {len(balanced_val)} examples")
    print(f"Test set: {len(test_examples)} examples")
    print("\nReady for fine-tuning!")


if __name__ == "__main__":
    main()