"""Legacy detector wrapper for NeuralStrategy.

Split out of neural_strategy.py; re-imported there (NeuralStrategy instantiates
it). Adapts a directly-loaded model/tokenizer to the DetectionModel interface.
"""

import logging
from typing import Dict, List, Any, Optional

from episodic.detection_models import DetectionModel

logger = logging.getLogger(__name__)


class _LegacyDetectorWrapper(DetectionModel):
    """
    Wrapper for legacy direct-loaded models to conform to DetectionModel interface.

    Used for backward compatibility when model_path is specified directly
    instead of using the model registry.
    """

    def __init__(self, model, tokenizer, device, temperature: float = 1.0):
        """
        Initialize wrapper with pre-loaded model components.

        Args:
            model: Pre-loaded transformer model
            tokenizer: Pre-loaded tokenizer
            device: torch device
            temperature: Softmax temperature for calibration
        """
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "LegacyDetector"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        # Already loaded in __init__
        return self._model is not None

    def predict(
        self,
        before_messages: List[str],
        after_messages: List[str]
    ) -> tuple:
        """
        Predict boundary using legacy model.

        Args:
            before_messages: List of formatted message strings before boundary
            after_messages: List of formatted message strings after boundary

        Returns:
            Tuple of (is_boundary, confidence)
        """
        try:
            import torch

            # Format as in training
            group1_text = " [SEP] ".join(before_messages)
            group2_text = " [SEP] ".join(after_messages)
            window_text = group1_text + " [BOUNDARY?] " + group2_text

            # Tokenize
            inputs = self._tokenizer(
                window_text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Inference with temperature scaling
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits / self._temperature
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                boundary_prob = probs[0][1].item()

            # Class 1 = boundary
            return pred_class == 1, boundary_prob

        except Exception as e:
            logger.error(f"Legacy detector prediction error: {e}")
            return False, 0.0

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_loaded": self.is_loaded,
            "device": str(self._device) if self._device else None,
            "temperature": self._temperature,
        }
