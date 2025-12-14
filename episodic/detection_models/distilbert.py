"""
DistilBERT-based topic boundary detection model.

Wraps fine-tuned DistilBERT models for topic boundary detection.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional

from .base import DetectionModel

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch or transformers not available, DistilBERT detector disabled")


class DistilBertDetector(DetectionModel):
    """
    Topic boundary detector using fine-tuned DistilBERT.

    Uses a (4,2) window configuration: 4 messages before + 2 messages after
    the potential boundary point.
    """

    def __init__(
        self,
        model_path: str,
        architecture: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        temperature: float = 1.0
    ):
        """
        Initialize the DistilBERT detector.

        Args:
            model_path: Path to the model weights file (.pt)
            architecture: Base architecture name for tokenizer
            device: Force specific device (cuda/mps/cpu), or None for auto
            temperature: Softmax temperature for calibration (default 1.0)
        """
        self._model_path = os.path.expanduser(model_path)
        self._architecture = architecture
        self._force_device = device
        self._temperature = temperature

        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False

    @property
    def name(self) -> str:
        return f"DistilBertDetector({os.path.basename(self._model_path)})"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _get_device(self):
        """Get the best available device."""
        if not TORCH_AVAILABLE:
            return None
        if self._force_device:
            return torch.device(self._force_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def load(self) -> bool:
        """Load the model into memory."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot load model")
            return False

        if self._loaded:
            return True

        if not os.path.exists(self._model_path):
            logger.error(f"Model file not found: {self._model_path}")
            return False

        try:
            self._device = self._get_device()
            logger.info(f"Loading DistilBERT model from {self._model_path} on {self._device}")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._architecture)

            # Load model architecture
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._architecture,
                num_labels=2,
                ignore_mismatched_sizes=True
            )

            # Load fine-tuned weights
            checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=False)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()

            self._loaded = True
            logger.info(f"DistilBERT model loaded successfully on {self._device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
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
            # Format input text
            before_text = " ".join(before_messages)
            after_text = " ".join(after_messages)
            combined_text = f"{before_text} [SEP] {after_text}"

            # Tokenize
            inputs = self._tokenizer(
                combined_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
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

                # Class 1 = boundary, Class 0 = no boundary
                boundary_prob = probs[0, 1].item()
                is_boundary = boundary_prob > 0.5

            return is_boundary, boundary_prob

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return False, 0.0

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._loaded = False
        logger.info("DistilBERT model unloaded")

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_info()
        info.update({
            "model_path": self._model_path,
            "architecture": self._architecture,
            "device": str(self._device) if self._device else None,
            "temperature": self._temperature,
            "torch_available": TORCH_AVAILABLE,
        })
        return info
