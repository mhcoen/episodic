"""
Configuration management for LLM providers.

This module provides information about available LLM providers and their models.
It does NOT manage user preferences or selected models - that's handled by episodic.config.
"""
import os
from typing import Dict, Any, List

# Map of provider names to their corresponding environment variable names
PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "azure": "AZURE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY"
}

# Local providers that don't require API keys
LOCAL_PROVIDERS = ["ollama", "lmstudio", "local"]

# Static provider configuration - what models are available
PROVIDER_CONFIG = {
    "openai": {
        "models": [
            {"name": "gpt-4o-mini", "display_name": "GPT-4o Mini"},
            {"name": "gpt-4o", "display_name": "GPT-4o"},
            {"name": "gpt-o3", "display_name": "GPT-o3"},
            {"name": "gpt-3.5-turbo", "display_name": "GPT-3.5 Turbo"},
            {"name": "gpt-4", "display_name": "GPT-4"},
            {"name": "gpt-4.5", "display_name": "GPT-4.5"}
        ]
    },
    "anthropic": {
        "models": [
            {"name": "claude-opus-4-20250514", "display_name": "Claude 4 Opus"},
            {"name": "claude-3-opus-20240229", "display_name": "Claude 3 Opus"},
            {"name": "claude-3-sonnet-20240229", "display_name": "Claude 3 Sonnet"},
            {"name": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku"}
        ]
    },
    "groq": {
        "models": [
            {"name": "llama3-8b-8192", "display_name": "Llama 3 8B"},
            {"name": "llama3-70b-8192", "display_name": "Llama 3 70B"},
            {"name": "mixtral-8x7b-32768", "display_name": "Mixtral 8x7B"},
            {"name": "gemma-7b-it", "display_name": "Gemma 7B"}
        ]
    },
    "ollama": {
        "api_base": "http://localhost:11434",
        "models": ["llama3", "mistral", "codellama", "phi3"]
    },
    "lmstudio": {
        "api_base": "http://localhost:1234/v1",
        "models": ["local-model"]
    },
    "local": {
        "models": []
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "models": [
            {"name": "openrouter/anthropic/claude-opus-4-20250514", "display_name": "Claude 4 Opus (OR)"},
            {"name": "openrouter/anthropic/claude-3-opus", "display_name": "Claude 3 Opus (OR)"},
            {"name": "openrouter/anthropic/claude-3-sonnet", "display_name": "Claude 3 Sonnet (OR)"},
            {"name": "openrouter/anthropic/claude-3-haiku", "display_name": "Claude 3 Haiku (OR)"},
            {"name": "openrouter/openai/gpt-4-turbo-preview", "display_name": "GPT-4 Turbo (OR)"},
            {"name": "openrouter/openai/gpt-4", "display_name": "GPT-4 (OR)"},
            {"name": "openrouter/openai/gpt-3.5-turbo", "display_name": "GPT-3.5 Turbo (OR)"},
            {"name": "openrouter/google/gemini-pro", "display_name": "Gemini Pro (OR)"},
            {"name": "openrouter/meta-llama/llama-2-70b-chat", "display_name": "Llama 2 70B (OR)"},
            {"name": "openrouter/mistralai/mixtral-8x7b-instruct", "display_name": "Mixtral 8x7B (OR)"},
            {"name": "openrouter/cohere/command-r-plus", "display_name": "Command R+ (OR)"}
        ]
    }
}

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