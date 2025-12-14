"""
HuggingFace Transformers wrapper for custom models.

Supports loading any HuggingFace-format model for various tasks:
- Sequence classification (topic boundary detection)
- Text generation (chat, completion)
- Summarization
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional

from .base import DetectionModel

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, HuggingFace wrapper disabled")


class HuggingFaceDetector(DetectionModel):
    """
    Generic HuggingFace model wrapper for topic boundary detection.

    Supports models saved in HuggingFace format (local directory or Hub ID).
    Automatically detects model architecture and loads appropriate classes.
    """

    # Supported task types and their model classes
    TASK_CLASSES = {
        "sequence-classification": "AutoModelForSequenceClassification",
        "text-generation": "AutoModelForCausalLM",
        "summarization": "AutoModelForSeq2SeqLM",
    }

    def __init__(
        self,
        model_path: str,
        task: str = "sequence-classification",
        device: Optional[str] = None,
        temperature: float = 1.0,
        max_length: int = 512,
    ):
        """
        Initialize the HuggingFace detector.

        Args:
            model_path: Path to local HuggingFace model directory or Hub model ID
            task: Model task type (sequence-classification, text-generation, summarization)
            device: Force specific device (cuda/mps/cpu), or None for auto
            temperature: Softmax temperature for calibration (default 1.0)
            max_length: Maximum sequence length for tokenization
        """
        self._model_path = os.path.expanduser(model_path)
        self._task = task
        self._force_device = device
        self._temperature = temperature
        self._max_length = max_length

        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False

    @property
    def name(self) -> str:
        return f"HuggingFaceDetector({os.path.basename(self._model_path)})"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _get_device(self):
        """Get the best available device."""
        if not TRANSFORMERS_AVAILABLE:
            return None
        if self._force_device:
            return torch.device(self._force_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_model_class(self):
        """Get the appropriate model class for the task."""
        if self._task == "sequence-classification":
            return AutoModelForSequenceClassification
        elif self._task == "text-generation":
            return AutoModelForCausalLM
        elif self._task == "summarization":
            return AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unknown task type: {self._task}")

    def load(self) -> bool:
        """Load the model into memory."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available, cannot load model")
            return False

        if self._loaded:
            return True

        # Check if path exists (for local models)
        if not self._model_path.startswith(("http://", "https://")) and \
           "/" not in self._model_path.replace(os.sep, "/").lstrip("/").split("/")[0]:
            # Looks like a Hub ID (e.g., "bert-base-uncased")
            pass
        elif not os.path.exists(self._model_path):
            logger.error(f"Model path not found: {self._model_path}")
            return False

        try:
            self._device = self._get_device()
            logger.info(f"Loading HuggingFace model from {self._model_path} on {self._device}")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

            # Load model with appropriate class
            model_class = self._get_model_class()
            self._model = model_class.from_pretrained(self._model_path)
            self._model.to(self._device)
            self._model.eval()

            self._loaded = True
            logger.info(f"HuggingFace model loaded successfully on {self._device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            self._loaded = False
            return False

    def predict(
        self,
        before_messages: List[str],
        after_messages: List[str]
    ) -> Tuple[bool, float]:
        """
        Predict whether there is a topic boundary between message windows.

        Args:
            before_messages: List of messages before the potential boundary
            after_messages: List of messages after the potential boundary

        Returns:
            Tuple of (is_boundary, confidence)
        """
        if not self._loaded:
            if not self.load():
                logger.error("Model not loaded, cannot predict")
                return False, 0.0

        try:
            # Format input based on task type
            if self._task == "sequence-classification":
                return self._predict_classification(before_messages, after_messages)
            elif self._task == "text-generation":
                return self._predict_generation(before_messages, after_messages)
            else:
                logger.error(f"Prediction not implemented for task: {self._task}")
                return False, 0.0

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return False, 0.0

    def _predict_classification(
        self,
        before_messages: List[str],
        after_messages: List[str]
    ) -> Tuple[bool, float]:
        """Predict using sequence classification model."""
        # Format input text
        before_text = " ".join(before_messages)
        after_text = " ".join(after_messages)

        # Try different input formats based on what the model expects
        # Format 1: Single sequence with separator
        combined_text = f"{before_text} [SEP] {after_text}"

        # Tokenize
        inputs = self._tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

            # Apply temperature scaling
            if self._temperature != 1.0:
                logits = logits / self._temperature

            probs = torch.softmax(logits, dim=-1)

            # Assume binary classification: class 1 = boundary
            if probs.shape[-1] >= 2:
                boundary_prob = probs[0, 1].item()
            else:
                boundary_prob = probs[0, 0].item()

            is_boundary = boundary_prob > 0.5

        return is_boundary, boundary_prob

    def _predict_generation(
        self,
        before_messages: List[str],
        after_messages: List[str]
    ) -> Tuple[bool, float]:
        """
        Predict using text generation model.

        For generative models, we prompt the model to classify.
        """
        before_text = " ".join(before_messages)
        after_text = " ".join(after_messages)

        prompt = f"""Analyze if there is a topic change between these conversation segments.

Before:
{before_text}

After:
{after_text}

Is there a topic boundary? Answer with just 'yes' or 'no':"""

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=self._temperature,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_lower = response.lower().strip()

        # Parse yes/no response
        if "yes" in response_lower:
            return True, 0.8  # High confidence for explicit yes
        elif "no" in response_lower:
            return False, 0.8
        else:
            return False, 0.5  # Uncertain

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if TRANSFORMERS_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._loaded = False
        logger.info("HuggingFace model unloaded")

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_info()
        info.update({
            "model_path": self._model_path,
            "task": self._task,
            "device": str(self._device) if self._device else None,
            "temperature": self._temperature,
            "max_length": self._max_length,
            "transformers_available": TRANSFORMERS_AVAILABLE,
        })
        return info
