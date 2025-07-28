#!/usr/bin/env python3
"""
SuperDialseg Dataset Loader for Episodic Evaluation

This module handles downloading and loading the SuperDialseg dataset
for evaluating topic segmentation approaches.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
import zipfile
import gdown

class SuperDialsegLoader:
    """Loads and processes the SuperDialseg dataset."""
    
    def __init__(self, data_dir: str = None):
        """Initialize the loader with a data directory."""
        if data_dir is None:
            # Try project datasets directory first
            project_datasets = Path(__file__).parent.parent / "datasets"
            if project_datasets.exists():
                self.data_dir = project_datasets
            else:
                # Fallback to user directory
                self.data_dir = Path("~/.episodic/datasets/superdialseg").expanduser()
                self.data_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.data_dir = Path(data_dir).expanduser()
        
        # Google Drive folder from the repository
        # https://drive.google.com/drive/folders/19YiHVfeI_M4HivrErIi9bghvUvsw9-Ws?usp=sharing
        self.drive_urls = {
            'doc2dial': 'https://drive.google.com/uc?id=1FGj4XxLB6g4LABcTaOBdK_Fh4P5M0eDZ',
            'qmsum': 'https://drive.google.com/uc?id=1JmBq96xOunnbLneAykhUxOXJgI5l1Ezs',
            'folder': 'https://drive.google.com/drive/folders/19YiHVfeI_M4HivrErIi9bghvUvsw9-Ws'
        }
    
    def download_dataset(self, dataset_name: str = 'superseg') -> Path:
        """
        Load a dataset that was manually downloaded from Google Drive.
        
        Expected dataset names: 'superseg', 'dialseg711', 'tiage', 'zys'
        """
        dataset_path = self.data_dir / dataset_name
        
        # Check if dataset exists
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} not found at {dataset_path}")
            print(f"\nPlease download the dataset from Google Drive and extract it to:")
            print(f"  {dataset_path}")
            print(f"\nAvailable datasets: superseg, dialseg711, tiage, zys")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        print(f"Using dataset at {dataset_path}")
        return dataset_path
    
    def load_conversations(self, dataset_path: Path, split: str = 'train') -> List[Dict[str, Any]]:
        """Load conversations from a dataset split."""
        # Map split names to file names
        split_mapping = {
            'train': 'segmentation_file_train.json',
            'validation': 'segmentation_file_validation.json',
            'val': 'segmentation_file_validation.json',
            'test': 'segmentation_file_test.json'
        }
        
        filename = split_mapping.get(split, f'segmentation_file_{split}.json')
        split_file = dataset_path / filename
        
        if not split_file.exists():
            available = [f.name for f in dataset_path.glob('segmentation_file_*.json')]
            raise FileNotFoundError(
                f"Could not find {split} split file: {filename}\n"
                f"Available files: {available}"
            )
        
        conversations = []
        
        # Load the SuperDialseg format
        with open(split_file, 'r') as f:
            data = json.load(f)
            
            # Navigate the structure: dial_data -> superseg-v2 -> list of dialogues
            if 'dial_data' in data:
                for dataset_name, dialogues in data['dial_data'].items():
                    conversations.extend(dialogues)
            else:
                # Fallback for different formats
                if isinstance(data, list):
                    conversations = data
                elif isinstance(data, dict):
                    conversations = [data]
        
        return conversations
    
    def parse_conversation(self, conv_data: Dict[str, Any]) -> Tuple[List[Dict], List[int]]:
        """
        Parse a conversation into Episodic-compatible format.
        
        Returns:
            Tuple of (messages, boundary_indices)
            - messages: List of {role, content} dicts
            - boundary_indices: List of indices where topics change
        """
        messages = []
        boundaries = []
        
        # Handle SuperDialseg format
        if 'turns' in conv_data:
            utterances = conv_data['turns']
            
            # Track topic changes and segmentation labels
            prev_topic_id = None
            
            for i, turn in enumerate(utterances):
                # Extract message
                role = turn.get('role', 'user')
                content = turn.get('utterance', '')
                
                # Normalize role names
                if role == 'agent':
                    role = 'assistant'
                
                messages.append({
                    'role': role,
                    'content': content,
                    'index': i
                })
                
                # Detect boundaries using topic_id change
                if 'topic_id' in turn:
                    current_topic_id = turn['topic_id']
                    if prev_topic_id is not None and current_topic_id != prev_topic_id:
                        # Topic changed - mark boundary at previous position
                        boundaries.append(i - 1)
                    prev_topic_id = current_topic_id
        
        else:
            # Fallback to original parsing logic for other formats
            if 'utterances' in conv_data:
                utterances = conv_data['utterances']
            else:
                raise KeyError(f"Could not find turns/utterances in conversation: {list(conv_data.keys())}")
            
            # Convert utterances to messages
            for i, utt in enumerate(utterances):
                if isinstance(utt, str):
                    # Simple string format - parse role from content
                    if utt.startswith(('User:', 'Human:', 'Customer:')):
                        role = 'user'
                        content = utt.split(':', 1)[1].strip() if ':' in utt else utt
                    else:
                        role = 'assistant'
                        content = utt.split(':', 1)[1].strip() if ':' in utt else utt
                elif isinstance(utt, dict):
                    role = utt.get('speaker', utt.get('role', 'user'))
                    content = utt.get('text', utt.get('content', utt.get('utterance', '')))
                    # Normalize role names
                    role = 'user' if role.lower() in ['user', 'human', 'customer'] else 'assistant'
                else:
                    raise ValueError(f"Unknown utterance format: {type(utt)}")
                
                messages.append({
                    'role': role,
                    'content': content,
                    'index': i
                })
            
            # Extract boundaries from other formats
            if 'segment_boundaries' in conv_data:
                boundaries = conv_data['segment_boundaries']
            elif 'topics' in conv_data:
                boundaries = self._derive_boundaries_from_topics(conv_data['topics'], len(utterances))
            elif 'segments' in conv_data:
                boundaries = self._derive_boundaries_from_segments(conv_data['segments'])
        
        return messages, boundaries
    
    def _derive_boundaries_from_topics(self, topics: List[Any], num_utterances: int) -> List[int]:
        """Derive boundary indices from topic annotations."""
        boundaries = []
        for i, topic in enumerate(topics[:-1]):  # All except last topic
            if isinstance(topic, dict) and 'end_idx' in topic:
                boundaries.append(topic['end_idx'])
            elif isinstance(topic, list) and len(topic) == 2:
                # Format: [start_idx, end_idx]
                boundaries.append(topic[1])
        return boundaries
    
    def _derive_boundaries_from_segments(self, segments: List[Any]) -> List[int]:
        """Derive boundary indices from segment annotations."""
        boundaries = []
        current_idx = 0
        for i, segment in enumerate(segments[:-1]):  # All except last segment
            if isinstance(segment, int):
                current_idx += segment
                boundaries.append(current_idx - 1)
            elif isinstance(segment, dict) and 'length' in segment:
                current_idx += segment['length']
                boundaries.append(current_idx - 1)
        return boundaries
    
    def get_dataset_stats(self, dataset_name: str = 'superseg') -> Dict[str, Any]:
        """Get statistics about a dataset."""
        dataset_path = self.download_dataset(dataset_name)
        
        stats = {
            'dataset': dataset_name,
            'splits': {},
            'total_conversations': 0,
            'total_utterances': 0,
            'total_boundaries': 0
        }
        
        for split in ['train', 'dev', 'test', 'val', 'validation']:
            try:
                conversations = self.load_conversations(dataset_path, split)
                split_name = 'validation' if split in ['val', 'validation'] else split
                
                num_utterances = 0
                num_boundaries = 0
                
                for conv in conversations:
                    try:
                        messages, boundaries = self.parse_conversation(conv)
                        num_utterances += len(messages)
                        num_boundaries += len(boundaries)
                    except Exception as e:
                        print(f"Error parsing conversation: {e}")
                        continue
                
                stats['splits'][split_name] = {
                    'conversations': len(conversations),
                    'utterances': num_utterances,
                    'boundaries': num_boundaries,
                    'avg_utterances_per_conv': num_utterances / len(conversations) if conversations else 0,
                    'avg_boundaries_per_conv': num_boundaries / len(conversations) if conversations else 0
                }
                
                stats['total_conversations'] += len(conversations)
                stats['total_utterances'] += num_utterances
                stats['total_boundaries'] += num_boundaries
                
            except FileNotFoundError:
                continue
        
        return stats


if __name__ == "__main__":
    # Test the loader
    loader = SuperDialsegLoader()
    
    print("=== SuperDialseg Dataset Loader Test ===\n")
    
    # Get dataset statistics
    try:
        # Test with superseg dataset
        stats = loader.get_dataset_stats('superseg')
        print(f"Dataset: {stats['dataset']}")
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Total utterances: {stats['total_utterances']}")
        print(f"Total boundaries: {stats['total_boundaries']}")
        print("\nSplits:")
        for split, split_stats in stats['splits'].items():
            print(f"  {split}: {split_stats['conversations']} conversations")
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease download the datasets from Google Drive first.")