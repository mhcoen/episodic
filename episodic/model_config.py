"""
Model configuration loader for Episodic.

This module loads model definitions from models.json and provides
access to model information including types, parameters, and capabilities.
"""
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from episodic.config import config
from episodic.debug_utils import debug_print


class ModelConfig:
    """Manages model configuration loaded from JSON files."""
    
    def __init__(self):
        """Initialize the model configuration."""
        self._models_data = {}
        self.load_models()
    
    def load_models(self):
        """Load model configuration from JSON files."""
        # Load models from ~/.episodic/models.json
        user_models_path = Path.home() / ".episodic" / "models.json"
        
        # If user models don't exist, create from template
        if not user_models_path.exists():
            self._create_default_models(user_models_path)
        
        # Load user models
        if user_models_path.exists():
            try:
                with open(user_models_path, 'r') as f:
                    self._models_data = json.load(f)
                    debug_print(f"Loaded models from {user_models_path}", category="models")
            except Exception as e:
                debug_print(f"Error loading models: {e}", category="models")
                self._models_data = {"providers": {}}
        else:
            self._models_data = {"providers": {}}
        
    
    def _create_default_models(self, user_models_path: Path):
        """Create default models.json from template."""
        package_dir = Path(__file__).parent
        template_path = package_dir / "models_template.json"
        
        if template_path.exists():
            try:
                # Ensure directory exists
                user_models_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy template to user directory
                import shutil
                shutil.copy2(template_path, user_models_path)
                debug_print(f"Created default models.json from template", category="models")
            except Exception as e:
                debug_print(f"Error creating default models: {e}", category="models")
    
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return self._models_data.get("providers", {}).get(provider, {})
    
    def get_provider_models(self, provider: str) -> List[Dict[str, Any]]:
        """Get list of models for a provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("models", [])
    
    def get_all_providers(self) -> Dict[str, Any]:
        """Get all provider configurations."""
        return self._models_data.get("providers", {})
    
    def get_model_info(self, provider: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        models = self.get_provider_models(provider)

        # First try exact match
        for model in models:
            if model.get("name") == model_name:
                return model

        # Try partial match
        for model in models:
            if model_name in model.get("name", ""):
                return model

        # Fall back to template for models not in user config
        # This allows new providers/models in template to work without user updating their config
        template_model = self._get_model_from_template(provider, model_name)
        if template_model:
            return template_model

        return None

    def _get_model_from_template(self, provider: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model info from template file as fallback."""
        try:
            package_dir = Path(__file__).parent
            template_path = package_dir / "models_template.json"

            if not template_path.exists():
                return None

            with open(template_path, 'r') as f:
                template_data = json.load(f)

            provider_config = template_data.get("providers", {}).get(provider, {})
            models = provider_config.get("models", [])

            for model in models:
                if model.get("name") == model_name:
                    return model

            return None
        except Exception:
            return None
    
    def detect_model_type(self, model_name: str) -> str:
        """Detect model type using patterns and known models."""
        model_lower = model_name.lower()

        # Custom local models are always detection type
        if model_lower.startswith('custom/'):
            return "detection"

        # Strip provider prefix if present (e.g., "anthropic/claude-..." -> "claude-...")
        model_without_prefix = model_lower.split('/')[-1] if '/' in model_lower else model_lower

        # Check all known models first (exact match, with or without provider prefix)
        for provider_name, provider_data in self._models_data.get("providers", {}).items():
            for model in provider_data.get("models", []):
                known_model = model.get("name", "").lower()
                if known_model == model_lower or known_model == model_without_prefix:
                    return model.get("type", "unknown")
        
        # Check patterns
        type_patterns = self._models_data.get("type_patterns", {})
        
        # Check instruct patterns
        for pattern in type_patterns.get("instruct", []):
            if pattern in model_lower:
                return "instruct"
        
        # Check chat patterns
        import re
        for pattern in type_patterns.get("chat", []):
            if '*' in pattern:
                # Convert glob to regex
                regex_pattern = pattern.replace('*', '.*')
                if re.search(regex_pattern, model_lower):
                    return "chat"
            elif pattern in model_lower:
                return "chat"
        
        # Check base patterns
        for pattern in type_patterns.get("base", []):
            if pattern in model_lower:
                return "base"
        
        # Default based on provider prefix
        if model_lower.startswith(('openai/', 'anthropic/')):
            return "chat"
        elif model_lower.startswith('huggingface/'):
            return "instruct"
        
        return "chat"  # Default
    
    def get_type_indicator(self, model_type: str) -> str:
        """Get the type indicator string for a model type."""
        # Built-in fallback indicators
        default_indicators = {
            "detection": "[D]",
            "chat": "[C]",
            "instruct": "[I]",
            "chat_instruct": "[CI]",
            "base": "[B]",
        }
        indicators = self._models_data.get("type_indicators", {})
        return indicators.get(model_type) or default_indicators.get(model_type, "[?]")
    
    def get_model_parameters(self, provider: str, model_name: str) -> Optional[str]:
        """Get parameter count for a model."""
        model_info = self.get_model_info(provider, model_name)
        if model_info:
            return model_info.get("parameters")
        return None
    
    def reload(self):
        """Reload model configuration from disk."""
        self.load_models()

    def is_local_model(self, model_name: str) -> bool:
        """
        Check if a model is a local/custom model (not API-based).

        Args:
            model_name: Full model name (e.g., "custom/topic-boundary-distilbert")

        Returns:
            True if the model is local, False if it's API-based
        """
        if "/" in model_name:
            provider = model_name.split("/")[0]
        else:
            provider = "custom"

        provider_config = self.get_provider_config(provider)
        return provider_config.get("local", False)

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get the file path for a local model.

        Args:
            model_name: Full model name (e.g., "custom/topic-boundary-distilbert")

        Returns:
            Expanded file path, or None if not a local model or path not found
        """
        if "/" in model_name:
            provider, name = model_name.split("/", 1)
        else:
            provider = "custom"
            name = model_name

        model_info = self.get_model_info(provider, name)
        if model_info and "path" in model_info:
            return os.path.expanduser(model_info["path"])
        return None


# Global instance
_model_config = None


def get_model_config() -> ModelConfig:
    """Get the global model configuration instance."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config


def reload_model_config():
    """Reload the model configuration."""
    global _model_config
    if _model_config:
        _model_config.reload()
    else:
        _model_config = ModelConfig()