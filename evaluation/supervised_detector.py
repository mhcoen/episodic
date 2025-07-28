#!/usr/bin/env python3
"""
Incremental supervised model for topic segmentation.

Uses a pre-trained transformer in a sliding window fashion for
near real-time topic boundary detection.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.detector_adapters import BaseDetectorAdapter


class SupervisedWindowDetector(BaseDetectorAdapter):
    """
    Sliding window detector using pre-trained transformer embeddings.
    
    This simulates a supervised approach by using a pre-trained model
    to get better representations than raw embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        window_size: int = 3,
        threshold: float = 0.5,
        use_gpu: bool = False
    ):
        super().__init__(f"supervised_{model_name.split('/')[-1]}_w{window_size}")
        self.window_size = window_size
        self.threshold = threshold
        
        # Load model and tokenizer
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding from transformer."""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy()[0]
    
    def _compute_window_similarity(self, window_a: List[str], window_b: List[str]) -> float:
        """Compute similarity between two windows of text."""
        # Get embeddings for each window
        emb_a = np.mean([self._get_embedding(text) for text in window_a], axis=0)
        emb_b = np.mean([self._get_embedding(text) for text in window_b], axis=0)
        
        # Compute cosine similarity
        similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        
        # Convert to distance (0 = similar, 1 = different)
        return 1 - similarity
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using supervised embeddings."""
        boundaries = []
        
        # Extract user messages
        user_messages = [(i, msg) for i, msg in enumerate(messages) if msg['role'] == 'user']
        
        if len(user_messages) < self.window_size * 2:
            return []
        
        # Process with sliding window
        for i in range(self.window_size, len(user_messages)):
            # Window A: previous window_size messages
            window_a_indices = range(i - self.window_size, i)
            window_a = [user_messages[j][1]['content'] for j in window_a_indices]
            
            # Window B: current message (in real-time we can't look ahead)
            window_b = [user_messages[i][1]['content']]
            
            # Compute dissimilarity
            distance = self._compute_window_similarity(window_a, window_b)
            
            if distance > self.threshold:
                # Mark boundary at the message before current user message
                msg_idx = user_messages[i][0]
                if msg_idx > 0:
                    boundaries.append(msg_idx - 1)
        
        return boundaries


class DialogueSegmentClassifier(BaseDetectorAdapter):
    """
    A more sophisticated approach that could be trained on SuperDialseg.
    
    For now, this uses heuristics based on transformer embeddings to
    simulate what a trained classifier might do.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialogRPT-human-vs-rand",
        context_size: int = 5,
        threshold: float = 0.5
    ):
        super().__init__(f"dialogue_classifier_{model_name.split('/')[-1]}")
        self.context_size = context_size
        self.threshold = threshold
        
        # Use a dialogue-specific model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def _score_continuation(self, context: List[str], candidate: str) -> float:
        """
        Score how well the candidate continues the context.
        Low scores indicate topic change.
        """
        # Format as dialogue
        dialogue = " ".join(context[-self.context_size:]) + " " + candidate
        
        # Get model prediction
        inputs = self.tokenizer(
            dialogue,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output or last hidden state
            if hasattr(outputs, 'pooler_output'):
                score = outputs.pooler_output.mean().item()
            else:
                score = outputs.last_hidden_state.mean().item()
        
        return score
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using dialogue coherence scoring."""
        boundaries = []
        context = []
        
        for i, msg in enumerate(messages):
            if i < 2:  # Need some context
                context.append(msg['content'])
                continue
                
            # Score continuation
            score = self._score_continuation(
                [m['content'] for m in messages[max(0, i-self.context_size):i]],
                msg['content']
            )
            
            # Low continuation score suggests topic change
            if score < self.threshold and msg['role'] == 'user' and i > 0:
                boundaries.append(i - 1)
            
            context.append(msg['content'])
        
        return boundaries


class HybridSupervisedDetector(BaseDetectorAdapter):
    """
    Combines supervised embeddings with our drift detection approach.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        window_size: int = 3,
        drift_threshold: float = 0.4,
        coherence_weight: float = 0.3
    ):
        super().__init__(f"hybrid_supervised_w{window_size}")
        self.window_detector = SupervisedWindowDetector(
            model_name=model_name,
            window_size=window_size,
            threshold=drift_threshold
        )
        self.coherence_weight = coherence_weight
        
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Combine multiple signals for boundary detection."""
        # Get boundaries from supervised detector
        supervised_boundaries = set(self.window_detector.detect_boundaries(messages))
        
        # Could add more signals here (keywords, time gaps, etc.)
        
        # For now, just return supervised boundaries
        return sorted(list(supervised_boundaries))


if __name__ == "__main__":
    # Test the supervised detectors
    print("Testing supervised detectors...")
    print("Note: First run will download transformer models (~50-500MB)")
    
    from evaluation.superdialseg_loader import SuperDialsegLoader
    from evaluation.metrics import SegmentationMetrics
    
    # Load sample data
    loader = SuperDialsegLoader()
    dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/superseg")
    conversations = loader.load_conversations(dataset_path, 'test')
    
    # Test on first conversation
    conv = conversations[0]
    messages, gold_boundaries = loader.parse_conversation(conv)
    
    print(f"\nConversation: {len(messages)} messages")
    print(f"Gold boundaries: {gold_boundaries}")
    
    # Test supervised detector (will download model on first run)
    print("\nTesting SupervisedWindowDetector...")
    detector = SupervisedWindowDetector(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        window_size=3,
        threshold=0.5
    )
    
    predicted = detector.detect_boundaries(messages)
    print(f"Predicted boundaries: {predicted}")
    
    if gold_boundaries:
        metrics_calc = SegmentationMetrics()
        metrics = metrics_calc.calculate_exact_metrics(
            predicted, gold_boundaries, len(messages)
        )
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")