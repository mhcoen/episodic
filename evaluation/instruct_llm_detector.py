"""
Instruct LLM-based topic drift detector.

Uses instruct-tuned language models to provide drift scores between messages.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
import os

from evaluation.detector_adapters import BaseDetectorAdapter

logger = logging.getLogger(__name__)


class InstructLLMDetector(BaseDetectorAdapter):
    """
    Topic detection using instruct-tuned LLMs to score drift.
    
    This detector uses instruct models to evaluate semantic drift between
    consecutive messages or within a window.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        threshold: float = 0.7,
        window_size: int = 3,
        use_api: bool = False,
        api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the instruct LLM detector.
        
        Args:
            model_name: Name of the instruct model to use
            threshold: Drift score threshold for detecting boundaries (0-1)
            window_size: Number of messages to consider for context
            use_api: Whether to use API (OpenAI/Together) instead of local model
            api_key: API key if using API
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name
        self.threshold = threshold
        self.window_size = window_size
        self.use_api = use_api
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = {
            'model': model_name,
            'threshold': threshold,
            'window_size': window_size
        }
        
        if use_api:
            if api_key:
                openai.api_key = api_key
            elif "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]
            else:
                raise ValueError("API key required for API-based models")
        else:
            # Load local model
            logger.info(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
    
    def _create_drift_prompt(self, messages: List[Dict[str, Any]], current_idx: int) -> str:
        """
        Create a prompt for the instruct model to evaluate drift.
        
        Returns a prompt asking for a drift score between 0 and 1.
        """
        # Get context window
        start_idx = max(0, current_idx - self.window_size + 1)
        context_messages = messages[start_idx:current_idx]
        current_message = messages[current_idx]
        
        # Build context string
        context_parts = []
        for msg in context_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        # Create the drift scoring prompt
        prompt = f"""Analyze the semantic drift between the conversation context and the new message.

Context (previous messages):
{chr(10).join(context_parts)}

New message:
{current_message['role'].capitalize()}: {current_message['content']}

Task: Rate how much the topic has drifted from the context to the new message.
- Score 0.0: Same topic, natural continuation
- Score 0.3: Slight drift but related topic
- Score 0.5: Moderate drift, somewhat related
- Score 0.7: Significant drift, different topic
- Score 1.0: Complete topic change

Respond with ONLY a number between 0.0 and 1.0.

Drift score:"""
        
        return prompt
    
    def _create_simple_drift_prompt(self, prev_message: Dict[str, Any], curr_message: Dict[str, Any]) -> str:
        """
        Create a simpler prompt comparing just two messages.
        """
        prompt = f"""Compare these two consecutive messages and rate the topic drift.

Message 1 ({prev_message['role']}): {prev_message['content']}
Message 2 ({curr_message['role']}): {curr_message['content']}

Rate the topic drift from 0.0 (same topic) to 1.0 (completely different topic).
Respond with ONLY a number.

Drift score:"""
        
        return prompt
    
    def _get_drift_score_api(self, prompt: str) -> float:
        """Get drift score using API."""
        try:
            if "gpt" in self.model_name.lower():
                # OpenAI API
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.0,
                    n=1
                )
                score_text = response.choices[0].text.strip()
            else:
                # Assume Together AI or similar API
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.0
                )
                score_text = response.choices[0].text.strip()
            
            # Parse the score
            try:
                score = float(score_text.split()[0].rstrip('.,'))
                return np.clip(score, 0.0, 1.0)
            except:
                logger.warning(f"Failed to parse score: {score_text}")
                return 0.5
                
        except Exception as e:
            logger.error(f"API error: {e}")
            return 0.5
    
    def _get_drift_score_local(self, prompt: str) -> float:
        """Get drift score using local model."""
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse score
            try:
                score = float(generated.strip().split()[0].rstrip('.,'))
                return np.clip(score, 0.0, 1.0)
            except:
                logger.warning(f"Failed to parse score: {generated}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return 0.5
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """
        Detect topic boundaries using instruct LLM drift scoring.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            List of indices where topic boundaries occur
        """
        if len(messages) < 2:
            return []
        
        boundaries = []
        drift_scores = []
        
        # Score drift for each message (starting from index 1)
        for i in range(1, len(messages)):
            # Create prompt based on window size
            if self.window_size == 1:
                # Simple pairwise comparison
                prompt = self._create_simple_drift_prompt(messages[i-1], messages[i])
            else:
                # Window-based comparison
                prompt = self._create_drift_prompt(messages, i)
            
            # Get drift score
            if self.use_api:
                score = self._get_drift_score_api(prompt)
            else:
                score = self._get_drift_score_local(prompt)
            
            drift_scores.append(score)
            
            # Check if this is a boundary
            if score >= self.threshold:
                boundaries.append(i - 1)  # Boundary after previous message
        
        # Log some statistics
        if drift_scores:
            logger.info(f"Drift scores - Mean: {np.mean(drift_scores):.3f}, "
                       f"Max: {np.max(drift_scores):.3f}, "
                       f"Boundaries: {len(boundaries)}")
        
        return boundaries


class FastInstructDetector(BaseDetectorAdapter):
    """
    Faster version using smaller instruct models or batch processing.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        threshold: float = 0.7,
        batch_size: int = 8,
        device: Optional[str] = None
    ):
        """
        Initialize fast instruct detector using encoder-decoder models.
        """
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = {
            'model': model_name,
            'threshold': threshold,
            'batch_size': batch_size
        }
        
        # Load model
        logger.info(f"Loading {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _create_t5_prompt(self, prev_message: Dict[str, Any], curr_message: Dict[str, Any]) -> str:
        """Create prompt for T5-style models."""
        return (
            f"Rate topic drift from 0 to 10: "
            f"Message 1: {prev_message['content'][:100]} "
            f"Message 2: {curr_message['content'][:100]}"
        )
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using batch processing."""
        if len(messages) < 2:
            return []
        
        boundaries = []
        prompts = []
        
        # Create all prompts
        for i in range(1, len(messages)):
            prompt = self._create_t5_prompt(messages[i-1], messages[i])
            prompts.append(prompt)
        
        # Process in batches
        drift_scores = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Generate scores
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )
            
            # Decode and parse scores
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                try:
                    # T5 outputs might be "3", "7", etc.
                    score = float(decoded.strip()) / 10.0
                    drift_scores.append(np.clip(score, 0.0, 1.0))
                except:
                    drift_scores.append(0.5)
        
        # Find boundaries
        for i, score in enumerate(drift_scores):
            if score >= self.threshold:
                boundaries.append(i)  # i is already offset by 1
        
        return boundaries