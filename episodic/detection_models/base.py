"""
Base interface for detection models.

All detection model wrappers must implement this interface to be used
with the detection model system.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


class DetectionModel(ABC):
    """
    Abstract base class for topic boundary detection models.

    All custom detection models must implement this interface to be
    compatible with the detection model factory system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded in memory."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """
        Load the model into memory.

        Returns:
            True if loading succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def predict(
        self,
        before_messages: List[str],
        after_messages: List[str]
    ) -> Tuple[bool, float]:
        """
        Predict whether there is a topic boundary between message windows.

        Args:
            before_messages: List of messages before the potential boundary.
            after_messages: List of messages after the potential boundary.

        Returns:
            Tuple of (is_boundary, confidence) where:
            - is_boundary: True if a topic boundary is detected
            - confidence: Float between 0 and 1 indicating confidence
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory to free resources.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information (name, loaded status, etc.)
        """
        return {
            "name": self.name,
            "is_loaded": self.is_loaded,
        }
