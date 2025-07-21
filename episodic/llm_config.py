"""
Configuration management for LLM providers.

This module provides information about available LLM providers and their models.
It does NOT manage user preferences or selected models - that's handled by episodic.config.
"""
import os
from typing import Dict, Any, List

from episodic.model_config import get_model_config

# Map of provider names to their corresponding environment variable names
PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "azure": "AZURE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY"
}

# Local providers that don't require API keys
LOCAL_PROVIDERS = ["ollama", "lmstudio", "local"]

# Get provider configuration from models.json
def _get_provider_config():
    """Get provider configuration from model config."""
    model_config = get_model_config()
    providers = model_config.get_all_providers()
    
    # Convert to the expected format
    result = {}
    for provider_name, provider_data in providers.items():
        models = provider_data.get("models", [])
        
        # For backwards compatibility with existing code
        if provider_name in ["ollama", "lmstudio"] and isinstance(models[0], dict) if models else False:
            # Convert dict models to simple strings for these providers
            simple_models = [m.get("name") for m in models if m.get("name")]
            provider_data = provider_data.copy()
            provider_data["models"] = simple_models
        
        result[provider_name] = provider_data
    
    return result

# Dynamic provider configuration - loaded from models.json
PROVIDER_CONFIG = _get_provider_config()

def get_current_provider() -> str:
    """
    Get the current provider based on the selected model.
    This is now determined by looking at which provider contains the current model.
    """
    from episodic.config import config
    current_model = config.get("model", "gpt-3.5-turbo")
    
    # Find which provider has this model
    for provider, details in PROVIDER_CONFIG.items():
        models = details.get("models", [])
        if isinstance(models, list):
            for model in models:
                if (isinstance(model, dict) and model.get("name") == current_model) or \
                   (isinstance(model, str) and model == current_model):
                    return provider
    
    # Default to openai if model not found
    return "openai"

def get_available_providers() -> Dict[str, Any]:
    """Get all available configured providers."""
    # Reload to get latest config
    global PROVIDER_CONFIG
    PROVIDER_CONFIG = _get_provider_config()
    return PROVIDER_CONFIG.copy()

def get_provider_models(provider: str) -> list:
    """Get available models for a specific provider."""
    if provider not in PROVIDER_CONFIG:
        return []
    return PROVIDER_CONFIG[provider].get("models", [])

def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get configuration for a specific provider."""
    if provider not in PROVIDER_CONFIG:
        return {}
    return PROVIDER_CONFIG[provider].copy()

def has_api_key(provider: str) -> bool:
    """
    Check if the API key for a provider is available.
    
    Args:
        provider: The name of the provider to check
        
    Returns:
        True if the provider is local or has an API key, False otherwise
    """
    # Local providers don't require API keys
    if provider in LOCAL_PROVIDERS:
        return True
    
    # Check if the provider is in the API keys map
    if provider not in PROVIDER_API_KEYS:
        return False
    
    # Check if the environment variable is set
    env_var = PROVIDER_API_KEYS[provider]
    return os.environ.get(env_var) is not None

def get_providers_with_api_keys() -> Dict[str, bool]:
    """
    Get a dictionary of all providers and whether they have API keys available.
    
    Returns:
        A dictionary mapping provider names to booleans indicating if they have API keys
    """
    return {provider: has_api_key(provider) for provider in PROVIDER_CONFIG}

def find_provider_for_model(model_name: str) -> str:
    """
    Find which provider offers a specific model.
    
    Args:
        model_name: The name of the model to find
        
    Returns:
        The provider name, or None if not found
    """
    for provider, details in PROVIDER_CONFIG.items():
        models = details.get("models", [])
        if isinstance(models, list):
            for model in models:
                if (isinstance(model, dict) and model.get("name") == model_name) or \
                   (isinstance(model, str) and model == model_name):
                    return provider
    return None

def validate_model_selection(model_name: str) -> None:
    """
    Validate that a model exists and its provider has an API key.
    
    Args:
        model_name: The name of the model to validate
        
    Raises:
        ValueError: If the model doesn't exist or its provider lacks an API key
    """
    provider = find_provider_for_model(model_name)
    
    if not provider:
        raise ValueError(f"Model '{model_name}' not found in any provider")
    
    if not has_api_key(provider):
        env_var = PROVIDER_API_KEYS.get(provider)
        if env_var:
            raise ValueError(f"Cannot use model '{model_name}' because the API key for {provider} is not set. "
                           f"Please set the {env_var} environment variable.")
        else:
            raise ValueError(f"Cannot use model '{model_name}' because the API key for {provider} is not available.")