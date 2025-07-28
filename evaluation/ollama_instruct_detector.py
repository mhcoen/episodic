"""
Ollama-based instruct model detector for topic drift scoring.
"""

import json
import logging
import subprocess
from typing import List, Dict, Any, Optional
import numpy as np

from evaluation.detector_adapters import BaseDetectorAdapter

logger = logging.getLogger(__name__)


class OllamaInstructDetector(BaseDetectorAdapter):
    """
    Topic detection using Ollama instruct models for drift scoring.
    """
    
    def __init__(
        self,
        model_name: str = "mistral:instruct",
        threshold: float = 0.7,
        window_size: int = 1,  # 1 for pairwise, >1 for context window
        temperature: float = 0.0,
        verbose: bool = False
    ):
        """
        Initialize the Ollama instruct detector.
        
        Args:
            model_name: Ollama model name (e.g., 'mistral:instruct', 'llama3:instruct')
            threshold: Drift score threshold for boundaries (0-1)
            window_size: Number of messages for context (1 = pairwise comparison)
            temperature: Model temperature (0 = deterministic)
            verbose: Whether to print debug information
        """
        self.model_name = model_name
        self.threshold = threshold
        self.window_size = window_size
        self.temperature = temperature
        self.verbose = verbose
        
        self.config = {
            'model': model_name,
            'threshold': threshold,
            'window_size': window_size,
            'temperature': temperature
        }
        
        # Test if model is available
        self._test_model()
    
    def _test_model(self):
        """Test if the model is available in Ollama."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            if self.model_name not in result.stdout:
                logger.warning(f"Model {self.model_name} not found. Run: ollama pull {self.model_name}")
        except subprocess.CalledProcessError:
            logger.error("Ollama not found or not running. Make sure Ollama is installed.")
    
    def _create_drift_prompt(self, messages: List[Dict[str, Any]], current_idx: int) -> str:
        """Create a prompt for drift scoring."""
        if self.window_size == 1 and current_idx > 0:
            # Simple pairwise comparison
            prev = messages[current_idx - 1]
            curr = messages[current_idx]
            
            prompt = f"""Rate topic change from 0.0 to 1.0.

Message 1: {prev['content']}
Message 2: {curr['content']}

Output ONLY a number like 0.7
Nothing else. Just the number.
"""
        else:
            # Window-based context (like sliding window)
            # For a (3,3) window at position i, we compare:
            # Window A: messages[i-3:i] vs Window B: messages[i:i+3]
            
            # Calculate window boundaries
            window_a_start = max(0, current_idx - self.window_size)
            window_a_end = current_idx
            window_b_start = current_idx
            window_b_end = min(len(messages), current_idx + self.window_size)
            
            # Get messages for both windows
            window_a = messages[window_a_start:window_a_end]
            window_b = messages[window_b_start:window_b_end]
            
            # Format windows
            window_a_str = "\n".join([
                f"{msg['content'][:100]}"  # Truncate long messages
                for msg in window_a
            ])
            
            window_b_str = "\n".join([
                f"{msg['content'][:100]}"
                for msg in window_b
            ])
            
            prompt = f"""Compare topics between two windows of messages.

Window A (before):
{window_a_str}

Window B (after):
{window_b_str}

Rate topic change from 0.0 to 1.0.
Output ONLY a number like 0.7
Nothing else. Just the number.
"""
        
        return prompt
    
    def _get_ollama_response(self, prompt: str) -> str:
        """Get response from Ollama."""
        try:
            # Prepare the command
            cmd = [
                'ollama', 'run',
                '--nowordwrap',
                self.model_name,
                prompt
            ]
            
            # Add temperature if not default
            if self.temperature != 0.0:
                cmd.extend(['--temperature', str(self.temperature)])
            
            # Run ollama
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30  # 30 second timeout
            )
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama timeout for model {self.model_name}")
            return "0.5"
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama error: {e.stderr}")
            return "0.5"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "0.5"
    
    def _parse_drift_score(self, response: str) -> float:
        """Parse drift score from model response."""
        try:
            # Clean the response
            response = response.strip()
            
            # Try to find a number
            import re
            # Look for decimal numbers
            numbers = re.findall(r'(?:^|[\s])([0-9]*\.?[0-9]+)(?:[\s]|$)', response)
            
            if numbers:
                score = float(numbers[0])
                return np.clip(score, 0.0, 1.0)
            
            # Try without decimal point
            numbers = re.findall(r'(?:^|[\s])([0-9]+)(?:[\s]|$)', response)
            if numbers:
                # Assume it's a percentage or needs scaling
                score = float(numbers[0])
                if score > 1:
                    score = score / 10.0  # Assume it's out of 10
                return np.clip(score, 0.0, 1.0)
            
            logger.warning(f"Could not parse score from: {response}")
            return 0.5
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return 0.5
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect topic boundaries using Ollama instruct model."""
        if len(messages) < 2:
            return []
        
        boundaries = []
        drift_scores = []
        
        # Process each message transition
        for i in range(1, len(messages)):
            # Create prompt
            prompt = self._create_drift_prompt(messages, i)
            
            # Get model response
            response = self._get_ollama_response(prompt)
            
            # Parse score
            score = self._parse_drift_score(response)
            drift_scores.append(score)
            
            if self.verbose:
                print(f"\nTransition {i-1} -> {i}:")
                print(f"Response: {response}")
                print(f"Score: {score}")
            
            # Check if boundary
            if score >= self.threshold:
                boundaries.append(i - 1)
        
        # Log statistics
        if drift_scores:
            mean_score = np.mean(drift_scores)
            max_score = np.max(drift_scores)
            if self.verbose or logger.isEnabledFor(logging.INFO):
                logger.info(f"Drift scores - Mean: {mean_score:.3f}, "
                           f"Max: {max_score:.3f}, "
                           f"Boundaries: {len(boundaries)}")
        
        return boundaries


class FastOllamaDetector(OllamaInstructDetector):
    """
    Faster version using batched prompts or simpler models.
    """
    
    def __init__(self, model_name: str = "phi3:instruct", **kwargs):
        """Use a faster model by default."""
        super().__init__(model_name=model_name, **kwargs)
    
    def _create_batch_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Create a single prompt for all transitions."""
        prompt = "Rate topic drift for each transition (0.0-1.0):\n\n"
        
        for i in range(1, min(len(messages), 10)):  # Limit to 10 for context size
            prev = messages[i-1]
            curr = messages[i]
            prompt += f"T{i}: '{prev['content'][:50]}...' -> '{curr['content'][:50]}...'\n"
        
        prompt += "\nProvide scores as comma-separated decimals:\n"
        return prompt
    
    def detect_boundaries_batch(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using batched prompt (experimental)."""
        if len(messages) < 2:
            return []
        
        prompt = self._create_batch_prompt(messages)
        response = self._get_ollama_response(prompt)
        
        # Parse multiple scores
        import re
        numbers = re.findall(r'[0-9]*\.?[0-9]+', response)
        scores = [float(n) for n in numbers]
        
        boundaries = []
        for i, score in enumerate(scores[:len(messages)-1]):
            if score >= self.threshold:
                boundaries.append(i)
        
        return boundaries