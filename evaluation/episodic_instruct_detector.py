"""
Instruct LLM detector using Episodic's existing LLM infrastructure.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import sys
from pathlib import Path

# Add episodic to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from episodic.llm import LLM
from evaluation.detector_adapters import BaseDetectorAdapter

logger = logging.getLogger(__name__)


class EpisodicInstructDetector(BaseDetectorAdapter):
    """
    Topic detection using Episodic's LLM infrastructure with instruct prompts.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        threshold: float = 0.7,
        window_size: int = 3,
        use_simple_prompt: bool = True
    ):
        """
        Initialize the detector.
        
        Args:
            model: Model name to use (defaults to current Episodic model)
            threshold: Drift score threshold (0-1)
            window_size: Number of messages for context
            use_simple_prompt: Use simple pairwise comparison vs window
        """
        self.llm = LLM()
        if model:
            self.llm.model = model
        
        self.threshold = threshold
        self.window_size = window_size
        self.use_simple_prompt = use_simple_prompt
        
        self.config = {
            'model': self.llm.model,
            'threshold': threshold,
            'window_size': window_size,
            'simple_prompt': use_simple_prompt
        }
    
    def _create_drift_prompt(self, messages: List[Dict[str, Any]], current_idx: int) -> str:
        """Create drift scoring prompt."""
        if self.use_simple_prompt and current_idx > 0:
            # Simple pairwise comparison
            prev = messages[current_idx - 1]
            curr = messages[current_idx]
            
            return f"""Analyze the topic drift between these two consecutive messages.

Previous message ({prev['role']}): {prev['content']}

Current message ({curr['role']}): {curr['content']}

Rate the topic drift as a decimal number between 0.0 and 1.0:
- 0.0 = Same topic, natural continuation
- 0.3 = Slight drift but related
- 0.5 = Moderate drift
- 0.7 = Significant drift, different topic  
- 1.0 = Complete topic change

Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else.

Drift score:"""
        
        else:
            # Window-based context
            start_idx = max(0, current_idx - self.window_size + 1)
            context_messages = messages[start_idx:current_idx]
            current_message = messages[current_idx]
            
            context_str = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in context_messages
            ])
            
            return f"""Analyze topic drift from the conversation context to the new message.

Context:
{context_str}

New message:
{current_message['role'].capitalize()}: {current_message['content']}

Rate the topic drift as a decimal number between 0.0 and 1.0:
- 0.0 = Same topic, natural continuation
- 0.3 = Slight drift but related
- 0.5 = Moderate drift
- 0.7 = Significant drift, different topic
- 1.0 = Complete topic change

Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else.

Drift score:"""
    
    def _get_drift_score(self, prompt: str) -> float:
        """Get drift score from LLM."""
        try:
            # Use Episodic's LLM with minimal parameters
            response = self.llm.query(
                prompt,
                system_message="You are a precise topic drift analyzer. Respond only with decimal numbers.",
                temperature=0.0,
                max_tokens=10
            )
            
            # Parse the response
            response_text = response.strip()
            
            # Try to extract the number
            import re
            numbers = re.findall(r'[0-9]*\.?[0-9]+', response_text)
            if numbers:
                score = float(numbers[0])
                return np.clip(score, 0.0, 1.0)
            else:
                logger.warning(f"No number found in response: {response_text}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error getting drift score: {e}")
            return 0.5
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect topic boundaries."""
        if len(messages) < 2:
            return []
        
        boundaries = []
        drift_scores = []
        
        # Score each message transition
        for i in range(1, len(messages)):
            prompt = self._create_drift_prompt(messages, i)
            score = self._get_drift_score(prompt)
            drift_scores.append(score)
            
            if score >= self.threshold:
                boundaries.append(i - 1)
        
        # Log statistics
        if drift_scores:
            logger.info(f"Drift scores - Mean: {np.mean(drift_scores):.3f}, "
                       f"Max: {np.max(drift_scores):.3f}, "
                       f"Boundaries: {len(boundaries)}")
        
        return boundaries


class BatchedInstructDetector(EpisodicInstructDetector):
    """
    Batched version that creates a single prompt for all transitions.
    More efficient but may be less accurate.
    """
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries with a single batched prompt."""
        if len(messages) < 2:
            return []
        
        # Create batched prompt
        prompt = "Analyze topic drift for each message transition. Rate each from 0.0 to 1.0.\n\n"
        
        for i in range(1, len(messages)):
            prev = messages[i-1]
            curr = messages[i]
            prompt += f"Transition {i}:\n"
            prompt += f"From ({prev['role']}): {prev['content'][:100]}...\n"
            prompt += f"To ({curr['role']}): {curr['content'][:100]}...\n\n"
        
        prompt += "Provide drift scores as a comma-separated list of decimal numbers:\n"
        
        try:
            response = self.llm.query(
                prompt,
                system_message="You are a topic drift analyzer. Respond only with comma-separated decimal numbers.",
                temperature=0.0,
                max_tokens=100
            )
            
            # Parse scores
            import re
            numbers = re.findall(r'[0-9]*\.?[0-9]+', response)
            scores = [float(n) for n in numbers]
            
            # Find boundaries
            boundaries = []
            for i, score in enumerate(scores[:len(messages)-1]):
                if score >= self.threshold:
                    boundaries.append(i)
            
            return boundaries
            
        except Exception as e:
            logger.error(f"Batch detection error: {e}")
            return []