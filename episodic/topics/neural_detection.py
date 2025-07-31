"""
Neural topic detection using fine-tuned transformer models.

This module provides GPU-aware neural topic detection to replace LLM-based detection.
"""

import logging
import os
from typing import Optional, List, Dict, Any, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch or transformers not available, neural topic detection disabled")

from episodic.config import config
from episodic.benchmark import benchmark_resource

# Set up logging
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_model_cache = {}
_tokenizer_cache = {}


def _get_device():
    """Get the best available device (GPU/MPS/CPU)."""
    if not TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _load_model(model_path: str = None) -> Tuple[Any, Any, Any]:
    """Load the fine-tuned model and tokenizer."""
    if not TORCH_AVAILABLE:
        return None, None, None
    
    global _model_cache, _tokenizer_cache
    
    # Default to DistilBERT model if not specified
    if model_path is None:
        # Look for model in evaluation directory first (development)
        dev_path = os.path.expanduser("~/proj/episodic/evaluation/finetuned_models_42/distilbert_base_uncased_42_window.pt")
        # Then check user's .episodic directory (production)
        prod_path = os.path.expanduser("~/.episodic/models/distilbert_base_uncased_42_window.pt")
        
        if os.path.exists(dev_path):
            model_path = dev_path
        elif os.path.exists(prod_path):
            model_path = prod_path
        else:
            logger.warning(f"Neural topic detection model not found at {dev_path} or {prod_path}")
            return None, None, None
    
    # Check cache
    if model_path in _model_cache:
        return _model_cache[model_path], _tokenizer_cache[model_path], _get_device()
    
    try:
        device = _get_device()
        logger.info(f"Loading neural topic detection model from {model_path} on {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Add special tokens as done during training
        special_tokens = {'additional_special_tokens': ['[BOUNDARY?]']}
        tokenizer.add_special_tokens(special_tokens)
        
        # Load model with correct vocab size
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Resize embeddings to match tokenizer
        model.resize_token_embeddings(len(tokenizer))
        
        # Load fine-tuned weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Cache for reuse
        _model_cache[model_path] = model
        _tokenizer_cache[model_path] = tokenizer
        
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Failed to load neural topic detection model: {e}")
        return None, None, None


def detect_topic_change_neural(
    recent_messages: List[Dict[str, Any]], 
    new_message: str,
    current_topic: Optional[Tuple[str, str]] = None,
    model_path: Optional[str] = None,
    assistant_response: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Detect topic change using neural model with (4,2) window configuration.
    
    Args:
        recent_messages: List of recent conversation nodes
        new_message: The new user message to analyze  
        current_topic: Optional tuple of (topic_name, start_node_id)
        model_path: Optional path to custom model file
        assistant_response: The assistant's response (for post-response detection)
        
    Returns:
        Tuple of (topic_changed: bool, new_topic_name: Optional[str], cost_info: Optional[Dict])
    """
    # Load model
    model, tokenizer, device = _load_model(model_path)
    
    if model is None:
        logger.warning("Neural model not available, returning no topic change")
        return False, None, None
    
    try:
        # Extract messages in the same format as training
        # We need full message objects with roles
        
        # We need at least 4 messages before
        if len(recent_messages) < 4:
            logger.debug(f"Not enough messages for (4,2) window: {len(recent_messages)} < 4")
            return False, None, None
        
        # Add the new user message to the end
        all_messages = list(recent_messages) + [{"role": "user", "content": new_message}]
        
        # If we have the assistant response, add it too
        if assistant_response:
            all_messages.append({"role": "assistant", "content": assistant_response})
        
        # For post-response detection, we check if there's a boundary right before the new exchange
        # We need 4 messages before and 2 after (user + assistant)
        
        if assistant_response:
            # We have a complete exchange, check boundary before it
            before_messages = all_messages[-6:-2] if len(all_messages) >= 6 else all_messages[:-2]
            after_messages = all_messages[-2:]  # User + Assistant
        else:
            # Pre-response detection (fallback)
            before_messages = all_messages[-5:-1] if len(all_messages) >= 5 else all_messages[:-1]
            after_messages = [all_messages[-1], {"role": "assistant", "content": ""}]
        
        # Format text exactly as in training
        group1_text = " [SEP] ".join([
            f"{msg['role']}: {msg.get('content', '')}" for msg in before_messages
        ])
        group2_text = " [SEP] ".join([
            f"{msg['role']}: {msg.get('content', '')}" for msg in after_messages
        ])
        
        # Combine with boundary marker
        window_text = group1_text + " [BOUNDARY?] " + group2_text
        
        # Tokenize and prepare input
        inputs = tokenizer(
            window_text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with benchmark_resource("Neural Topic Detection", f"device: {device}"):
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
        
        # Class 1 = topic boundary, Class 0 = no boundary
        # Due to model limitations with unseen conversation types, we'll be conservative
        # Only report topic change if confidence is high
        confidence_threshold = config.get("neural_confidence_threshold", 0.8)
        
        if predicted_class == 1 and confidence >= confidence_threshold:
            topic_changed = True
        else:
            topic_changed = False
        
        if config.get("debug"):
            logger.debug(f"Neural detection result: {'Topic change' if topic_changed else 'Same topic'}")
            logger.debug(f"Confidence: {confidence:.3f} (threshold: {confidence_threshold})")
            logger.debug(f"Predicted class: {predicted_class}, Probabilities: {probabilities[0].tolist()}")
            logger.debug(f"Window composition: {len(before_messages)} before, {len(after_messages)} after")
            logger.debug(f"Window text preview: {window_text[:300]}...")
            # Also print to console for visibility
            print(f"\nDEBUG: Class {predicted_class}, Probs: [{probabilities[0][0]:.3f}, {probabilities[0][1]:.3f}]")
            print(f"DEBUG: Window has {len(before_messages)} before, {len(after_messages)} after messages")
        
        # Create cost info for compatibility
        cost_info = {
            "model": f"neural_distilbert_{device}",
            "confidence": confidence,
            "tokens": len(tokenizer.encode(window_text))
        }
        
        return topic_changed, None, cost_info
        
    except Exception as e:
        logger.error(f"Neural topic detection error: {e}")
        return False, None, None


def check_neural_model_available(model_path: Optional[str] = None) -> bool:
    """Check if neural model is available and can be loaded."""
    model, tokenizer, device = _load_model(model_path)
    return model is not None