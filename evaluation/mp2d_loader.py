"""Loader for the MP2D dataset."""

import json
from typing import List, Dict, Any, Tuple


class MP2DDatasetLoader:
    """Loader for the MP2D (Multi-Party to Dialogue) dataset."""
    
    def __init__(self, dataset_path: str):
        """Initialize the loader with the dataset file path."""
        self.dataset_path = dataset_path
        
    def load_dialogues(self, max_dialogues: int = None) -> List[Tuple[List[Dict[str, Any]], List[int]]]:
        """
        Load dialogues from the MP2D dataset.
        
        Returns:
            List of tuples (messages, boundaries) where:
            - messages: List of message dicts with 'role' and 'content'
            - boundaries: List of turn indices where topic changes occur
        """
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        dialogues = []
        
        # MP2D format has 'texts' key containing list of samples
        samples = data.get('texts', [])
        
        for i, sample in enumerate(samples):
            if max_dialogues and len(dialogues) >= max_dialogues:
                break
            
            # Extract dialogue turns
            dialog = sample.get('dialog', [])
            messages = []
            
            # Convert Q&A pairs to messages
            for j, turn in enumerate(dialog):
                # Add question as user message
                messages.append({
                    'role': 'user',
                    'content': turn['question']
                })
                # Add answer as assistant message
                messages.append({
                    'role': 'assistant', 
                    'content': turn['answer']
                })
            
            # Extract topic boundaries
            # MP2D stores boundaries as indices of dialogue turns (not message indices)
            topic_shifts = sample.get('topic_shift', [])
            
            # Convert dialogue turn indices to message indices
            # Each dialogue turn creates 2 messages (Q and A)
            boundaries = []
            for shift_idx in topic_shifts:
                # Boundary occurs after the answer of the specified turn
                # So it's at message index (shift_idx * 2) + 1
                message_boundary_idx = (shift_idx * 2) + 1
                if message_boundary_idx < len(messages):
                    boundaries.append(message_boundary_idx)
            
            dialogues.append((messages, boundaries))
        
        return dialogues
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data.get('texts', [])
        
        # Calculate statistics
        total_dialogues = len(samples)
        total_turns = sum(len(s.get('dialog', [])) for s in samples)
        total_messages = total_turns * 2  # Each turn has Q and A
        total_boundaries = sum(len(s.get('topic_shift', [])) for s in samples)
        
        # Topic statistics
        all_topics = []
        for sample in samples:
            all_topics.extend(sample.get('topics', []))
        unique_topics = len(set(all_topics))
        
        return {
            'num_dialogues': total_dialogues,
            'total_turns': total_turns,
            'total_messages': total_messages,
            'avg_turns_per_dialogue': total_turns / total_dialogues if total_dialogues else 0,
            'avg_messages_per_dialogue': total_messages / total_dialogues if total_dialogues else 0,
            'total_boundaries': total_boundaries,
            'avg_boundaries_per_dialogue': total_boundaries / total_dialogues if total_dialogues else 0,
            'unique_topics': unique_topics,
            'total_topic_mentions': len(all_topics)
        }