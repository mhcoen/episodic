"""Loader for the TIAGE dataset."""

import json
from typing import List, Dict, Any, Tuple


class TiageDatasetLoader:
    """Loader for the TIAGE dialogue dataset."""
    
    def __init__(self, dataset_path: str):
        """Initialize the loader with the dataset file path."""
        self.dataset_path = dataset_path
        
    def load_dialogues(self, max_dialogues: int = None) -> List[Tuple[List[Dict[str, Any]], List[int]]]:
        """
        Load dialogues from the TIAGE dataset.
        
        Returns:
            List of tuples (messages, boundaries) where:
            - messages: List of message dicts with 'role' and 'content'
            - boundaries: List of turn indices where topic changes occur
        """
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        dialogues = []
        
        # TIAGE format has dialogues under 'dial_data' -> 'tiage'
        for dialogue in data['dial_data']['tiage']:
            if max_dialogues and len(dialogues) >= max_dialogues:
                break
                
            messages = []
            boundaries = []
            prev_topic_id = None
            
            for i, turn in enumerate(dialogue['turns']):
                # Create message in expected format
                messages.append({
                    'role': turn['role'],
                    'content': turn['utterance']
                })
                
                # Detect topic boundaries based on topic_id changes
                if 'topic_id' in turn:
                    current_topic_id = turn['topic_id']
                    if prev_topic_id is not None and current_topic_id != prev_topic_id:
                        # Mark the previous turn as boundary
                        boundaries.append(i - 1)
                    prev_topic_id = current_topic_id
            
            dialogues.append((messages, boundaries))
        
        return dialogues
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        dialogues = data['dial_data']['tiage']
        total_turns = sum(len(d['turns']) for d in dialogues)
        total_boundaries = 0
        
        for dialogue in dialogues:
            prev_topic_id = None
            for turn in dialogue['turns']:
                if 'topic_id' in turn:
                    current_topic_id = turn['topic_id']
                    if prev_topic_id is not None and current_topic_id != prev_topic_id:
                        total_boundaries += 1
                    prev_topic_id = current_topic_id
        
        return {
            'num_dialogues': len(dialogues),
            'total_turns': total_turns,
            'avg_turns_per_dialogue': total_turns / len(dialogues) if dialogues else 0,
            'total_boundaries': total_boundaries,
            'avg_boundaries_per_dialogue': total_boundaries / len(dialogues) if dialogues else 0
        }