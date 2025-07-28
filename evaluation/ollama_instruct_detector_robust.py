#!/usr/bin/env python3
"""Robust Ollama instruct model-based topic detector for window_size=3."""

import re
import subprocess
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RobustOllamaInstructDetector:
    """Topic boundary detector using Ollama instruct models with robust parsing."""
    
    def __init__(self, model_name="mistral:instruct", threshold=0.7, 
                 window_size=3, temperature=0.0, verbose=False):
        self.model_name = model_name
        self.threshold = threshold
        self.window_size = window_size
        self.temperature = temperature
        self.verbose = verbose
        
    def _create_simple_prompt(self, messages: List[Dict[str, Any]], 
                             current_idx: int) -> str:
        """Create a simpler prompt that works better with verbose models."""
        # Calculate window boundaries
        window_a_start = max(0, current_idx - self.window_size)
        window_a_end = current_idx
        window_b_start = current_idx
        window_b_end = min(len(messages), current_idx + self.window_size)
        
        # Get messages for both windows
        window_a = messages[window_a_start:window_a_end]
        window_b = messages[window_b_start:window_b_end]
        
        # Extract key content words (avoid common words)
        def extract_key_words(msgs):
            text = " ".join([m['content'] for m in msgs])
            words = re.findall(r'\b\w+\b', text.lower())
            common = {'the', 'a', 'an', 'is', 'it', 'for', 'and', 'or', 'but', 
                     'in', 'on', 'at', 'to', 'of', 'with', 'yes', 'no', 'i', 
                     'you', 'we', 'they', 'am', 'are', 'was', 'were', 'been', 
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                     'would', 'could', 'should', 'may', 'might', 'must', 'can',
                     'what', 'how', 'me', 'my', 'your', 'our', 'their', 'this',
                     'that', 'these', 'those', 'ok', 'okay', 'sure', 'please',
                     'thank', 'thanks', 'hello', 'hi', 'help', 'need', 'want'}
            key_words = [w for w in words if w not in common and len(w) > 2]
            # Get most common non-stopwords
            from collections import Counter
            word_counts = Counter(key_words)
            return [word for word, _ in word_counts.most_common(5)]
        
        keywords_a = extract_key_words(window_a)
        keywords_b = extract_key_words(window_b)
        
        # Use a VERY simple prompt
        prompt = f"""Topic A: {', '.join(keywords_a[:3]) if keywords_a else 'none'}
Topic B: {', '.join(keywords_b[:3]) if keywords_b else 'none'}
Different? (0=same, 1=different):"""
        
        return prompt
    
    def _get_ollama_response(self, prompt: str) -> str:
        """Get response from Ollama."""
        try:
            cmd = [
                'ollama', 'run',
                '--nowordwrap',
                self.model_name,
                prompt
            ]
            
            if self.temperature != 0.0:
                cmd.extend(['--temperature', str(self.temperature)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
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
        """Parse drift score from model response with very robust handling."""
        try:
            response = response.strip()
            
            # First, look for any float number
            float_matches = re.findall(r'\d*\.?\d+', response)
            
            if float_matches:
                # Take the first number found
                score = float(float_matches[0])
                
                # Handle common patterns
                if score > 10:
                    # Might be a percentage like 85
                    score = score / 100.0
                elif score > 1:
                    # Might be out of 10
                    score = score / 10.0
                
                return np.clip(score, 0.0, 1.0)
            
            # Look for words that indicate high/low change
            response_lower = response.lower()
            if any(word in response_lower for word in ['same', 'similar', 'no change', 'unchanged']):
                return 0.1
            elif any(word in response_lower for word in ['different', 'changed', 'new topic', 'completely']):
                return 0.9
            elif any(word in response_lower for word in ['somewhat', 'partially', 'slight']):
                return 0.5
            
            # Default
            if self.verbose:
                logger.warning(f"Could not parse score from: {response[:100]}")
            return 0.5
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Parse error: {e}")
            return 0.5
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect topic boundaries using Ollama instruct model."""
        if len(messages) < 2:
            return []
        
        boundaries = []
        drift_scores = []
        
        start_idx = self.window_size if self.window_size > 1 else 1
        
        for i in range(start_idx, len(messages)):
            prompt = self._create_simple_prompt(messages, i)
            
            if self.verbose:
                logger.info(f"Prompt for position {i}: {prompt}")
            
            response = self._get_ollama_response(prompt)
            
            if self.verbose:
                logger.info(f"Response: {response[:100]}")
            
            score = self._parse_drift_score(response)
            drift_scores.append(score)
            
            if score >= self.threshold:
                boundaries.append(i)
        
        if self.verbose:
            logger.info(f"Drift scores: {drift_scores}")
            logger.info(f"Detected boundaries: {boundaries}")
        
        return boundaries