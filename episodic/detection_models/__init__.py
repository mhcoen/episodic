"""
Detection models package.

Provides a unified interface for custom local detection models
used for topic boundary detection.
"""

import os
import logging
from typing import Optional, Dict, Any

from .base import DetectionModel
from .distilbert import DistilBertDetector

logger = logging.getLogger(__name__)

# Registry of available wrapper types
WRAPPER_REGISTRY = {
    "distilbert": DistilBertDetector,
}

# Model cache to avoid reloading
_model_cache: Dict[str, DetectionModel] = {}


def get_detector(
    model_name: str,
    model_config: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Optional[DetectionModel]:
    """
    Get a detection model instance by name.

    Args:
        model_name: Full model name (e.g., "custom/topic-boundary-distilbert")
        model_config: Optional model configuration dict. If not provided,
                     will be looked up from models.json
        use_cache: Whether to use/store in model cache (default True)

    Returns:
        DetectionModel instance, or None if model not found/loadable
    """
    # Check cache first
    if use_cache and model_name in _model_cache:
        return _model_cache[model_name]

    # Get config if not provided
    if model_config is None:
        model_config = _get_model_config(model_name)

    if model_config is None:
        logger.error(f"Model config not found for: {model_name}")
        return None

    # Get wrapper type
    wrapper_type = model_config.get("wrapper")
    if wrapper_type not in WRAPPER_REGISTRY:
        logger.error(f"Unknown wrapper type: {wrapper_type}")
        return None

    # Get model path
    model_path = model_config.get("path")
    if not model_path:
        logger.error(f"No path specified for model: {model_name}")
        return None

    # Expand path
    model_path = os.path.expanduser(model_path)

    # Create detector instance
    wrapper_class = WRAPPER_REGISTRY[wrapper_type]

    try:
        if wrapper_type == "distilbert":
            detector = wrapper_class(
                model_path=model_path,
                architecture=model_config.get("architecture", "distilbert-base-uncased"),
                temperature=model_config.get("temperature", 1.0)
            )
        else:
            # Generic instantiation for future wrappers
            detector = wrapper_class(model_path=model_path, **model_config)

        # Cache if enabled
        if use_cache:
            _model_cache[model_name] = detector

        return detector

    except Exception as e:
        logger.error(f"Failed to create detector for {model_name}: {e}")
        return None


def _get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get model configuration from models.json.

    Args:
        model_name: Full model name (e.g., "custom/topic-boundary-distilbert")

    Returns:
        Model configuration dict, or None if not found
    """
    try:
        from episodic.model_config import get_model_config
        config = get_model_config()

        # Parse provider/model from name
        if "/" in model_name:
            provider, name = model_name.split("/", 1)
        else:
            provider = "custom"
            name = model_name

        # Look up model
        return config.get_model_info(provider, name)

    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        return None


def clear_cache() -> None:
    """Clear the model cache, unloading all models."""
    global _model_cache
    for model in _model_cache.values():
        try:
            model.unload()
        except Exception as e:
            logger.warning(f"Error unloading model: {e}")
    _model_cache.clear()


def list_available_wrappers() -> list:
    """List available wrapper types."""
    return list(WRAPPER_REGISTRY.keys())


__all__ = [
    "DetectionModel",
    "DistilBertDetector",
    "get_detector",
    "clear_cache",
    "list_available_wrappers",
    "WRAPPER_REGISTRY",
]
