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
        self._user_models_data = {}
        self._combined_data = {}
        self.load_models()
    
    def load_models(self):
        """Load model configuration from JSON files."""
        # Load default models from package
        package_dir = Path(__file__).parent
        default_models_path = package_dir / "models.json"
        
        if default_models_path.exists():
            try:
                with open(default_models_path, 'r') as f:
                    self._models_data = json.load(f)
                    debug_print(f"Loaded default models from {default_models_path}", category="models")
            except Exception as e:
                debug_print(f"Error loading default models: {e}", category="models")
                self._models_data = {"providers": {}}
        
        # Load user models from ~/.episodic/models.json
        user_models_path = Path.home() / ".episodic" / "models.json"
        if user_models_path.exists():
            try:
                with open(user_models_path, 'r') as f:
                    self._user_models_data = json.load(f)
                    debug_print(f"Loaded user models from {user_models_path}", category="models")
            except Exception as e:
                debug_print(f"Error loading user models: {e}", category="models")
                self._user_models_data = {}
        
        # Combine default and user models
        self._combine_models()
    
    def _combine_models(self):
        """Combine default and user model configurations."""
        import copy
        self._combined_data = copy.deepcopy(self._models_data)
        
        if not self._user_models_data:
            return
        
        # Merge providers
        user_providers = self._user_models_data.get("providers", {})
        for provider_name, provider_data in user_providers.items():
            if provider_name in self._combined_data.get("providers", {}):
                # Merge models for existing provider
                existing_models = {m["name"]: m for m in self._combined_data["providers"][provider_name].get("models", [])}
                
                for model in provider_data.get("models", []):
                    model_name = model.get("name")
                    if model_name:
                        # Override or add model
                        existing_models[model_name] = model
                
                # Convert back to list
                self._combined_data["providers"][provider_name]["models"] = list(existing_models.values())
                
                # Update other provider properties
                for key, value in provider_data.items():
                    if key != "models":
                        self._combined_data["providers"][provider_name][key] = value
            else:
                # Add new provider
                self._combined_data["providers"][provider_name] = provider_data
        
        # Merge type patterns
        if "type_patterns" in self._user_models_data:
            for pattern_type, patterns in self._user_models_data["type_patterns"].items():
                if pattern_type in self._combined_data.get("type_patterns", {}):
                    # Extend patterns (remove duplicates)
                    existing = set(self._combined_data["type_patterns"][pattern_type])
                    existing.update(patterns)
                    self._combined_data["type_patterns"][pattern_type] = list(existing)
                else:
                    self._combined_data["type_patterns"][pattern_type] = patterns
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return self._combined_data.get("providers", {}).get(provider, {})
    
    def get_provider_models(self, provider: str) -> List[Dict[str, Any]]:
        """Get list of models for a provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("models", [])
    
    def get_all_providers(self) -> Dict[str, Any]:
        """Get all provider configurations."""
        return self._combined_data.get("providers", {})
    
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
        
        return None
    
    def detect_model_type(self, model_name: str) -> str:
        """Detect model type using patterns and known models."""
        model_lower = model_name.lower()
        
        # Check all known models first
        for provider_name, provider_data in self._combined_data.get("providers", {}).items():
            for model in provider_data.get("models", []):
                if model.get("name", "").lower() in model_lower:
                    return model.get("type", "unknown")
        
        # Check patterns
        type_patterns = self._combined_data.get("type_patterns", {})
        
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
        indicators = self._combined_data.get("type_indicators", {})
        return indicators.get(model_type, "[?]")
    
    def get_model_parameters(self, provider: str, model_name: str) -> Optional[str]:
        """Get parameter count for a model."""
        model_info = self.get_model_info(provider, model_name)
        if model_info:
            return model_info.get("parameters")
        return None
    
    def reload(self):
        """Reload model configuration from disk."""
        self.load_models()


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