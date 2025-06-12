"""
Configuration management for LLM providers.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration path
CONFIG_PATH = Path.home() / ".episodic" / "llm_config.json"

def ensure_config_dir():
    """Ensure the configuration directory exists."""
    config_dir = CONFIG_PATH.parent
    config_dir.mkdir(parents=True, exist_ok=True)

def get_default_config() -> Dict[str, Any]:
    """Return the default configuration."""
    return {
        "default_provider": "openai",
        "default_model": "gpt-4o-mini",
        "providers": {
            "openai": {
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            },
            "anthropic": {
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            },
            "ollama": {
                "models": ["llama3", "mistral", "codellama", "phi3"]
            },
            "lmstudio": {
                "api_base": "http://localhost:1234/v1",
                "models": ["local-model"]
            },
            "local": {
                "models": []
            }
        }
    }

def load_config() -> Dict[str, Any]:
    """Load LLM configuration from file."""
    ensure_config_dir()

    if not CONFIG_PATH.exists():
        # Create default config if it doesn't exist
        default_config = get_default_config()
        save_config(default_config)
        return default_config

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config: Dict[str, Any]) -> None:
    """Save LLM configuration to file."""
    ensure_config_dir()

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def get_current_provider() -> str:
    """Get the current default provider."""
    config = load_config()
    return config.get("default_provider", "openai")

def set_current_provider(provider: str) -> None:
    """Set the current default provider."""
    config = load_config()
    if provider not in config.get("providers", {}):
        raise ValueError(f"Provider '{provider}' not configured")

    config["default_provider"] = provider
    save_config(config)

def get_available_providers() -> Dict[str, Any]:
    """Get all available configured providers."""
    config = load_config()
    return config.get("providers", {})

def get_provider_models(provider: str) -> list:
    """Get available models for a specific provider."""
    config = load_config()
    providers = config.get("providers", {})
    if provider not in providers:
        return []
    return providers[provider].get("models", [])

def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get configuration for a specific provider."""
    config = load_config()
    providers = config.get("providers", {})
    if provider not in providers:
        return {}
    return providers[provider]

def get_default_model() -> str:
    """Get the current default model."""
    config = load_config()
    return config.get("default_model", "gpt-4o-mini")

def ensure_provider_matches_model() -> None:
    """
    Ensure that the default provider matches the provider of the default model.
    This fixes any inconsistencies in the configuration.
    """
    config = load_config()
    current_model = config.get("default_model", "gpt-4o-mini")
    current_provider = config.get("default_provider", "openai")

    # Find which provider has this model
    correct_provider = None

    for provider, details in config.get("providers", {}).items():
        models = details.get("models", [])
        if isinstance(models, list):
            for model in models:
                if (isinstance(model, dict) and model.get("name") == current_model) or \
                   (isinstance(model, str) and model == current_model):
                    correct_provider = provider
                    break
        if correct_provider:
            break

    # If we found a provider for the model and it's different from the current provider,
    # update the provider
    if correct_provider and correct_provider != current_provider:
        config["default_provider"] = correct_provider
        save_config(config)

def set_default_model(model_name: str) -> None:
    """
    Set the default model and update the default provider to match.

    Args:
        model_name: The name of the model to set as default

    Raises:
        ValueError: If the model is not found in any provider
    """
    config = load_config()

    # Find which provider has this model
    provider_for_model = None
    model_exists = False

    for provider, details in config.get("providers", {}).items():
        models = details.get("models", [])
        if isinstance(models, list):
            for model in models:
                if (isinstance(model, dict) and model.get("name") == model_name) or \
                   (isinstance(model, str) and model == model_name):
                    provider_for_model = provider
                    model_exists = True
                    break
        if model_exists:
            break

    if not model_exists:
        raise ValueError(f"Model '{model_name}' not found in any provider")

    # Set the default model and its provider
    config["default_model"] = model_name
    config["default_provider"] = provider_for_model
    save_config(config)

def add_local_model(name: str, path: str, backend: str = "llama.cpp") -> None:
    """Add a local model configuration."""
    config = load_config()
    if "local" not in config.get("providers", {}):
        config["providers"]["local"] = {"models": []}

    # Add model to local provider
    local_models = config["providers"]["local"].get("models", [])

    # Check if model already exists
    for i, model in enumerate(local_models):
        if model.get("name") == name:
            # Update existing model
            local_models[i] = {"name": name, "path": path, "backend": backend}
            save_config(config)
            return

    # Add new model
    local_models.append({"name": name, "path": path, "backend": backend})
    config["providers"]["local"]["models"] = local_models
    save_config(config)
