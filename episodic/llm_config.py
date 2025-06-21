"""
Configuration management for LLM providers.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Default configuration path
CONFIG_PATH = Path.home() / ".episodic" / "llm_config.json"

# Map of provider names to their corresponding environment variable names
PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "azure": "AZURE_API_KEY",
    "groq": "GROQ_API_KEY"
}

# Local providers that don't require API keys
LOCAL_PROVIDERS = ["ollama", "lmstudio", "local"]

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
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4", "gpt-4.5"]
            },
            "anthropic": {
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            },
            "groq": {
                "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
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
        config = json.load(f)

    # Check if the OpenAI models list includes "gpt-4" and "gpt-4.5"
    # If not, add them to ensure they're available
    if "providers" in config and "openai" in config["providers"]:
        openai_models = config["providers"]["openai"].get("models", [])
        models_to_add = []

        if "gpt-4" not in openai_models:
            models_to_add.append("gpt-4")

        if "gpt-4.5" not in openai_models:
            models_to_add.append("gpt-4.5")

        if models_to_add:
            # Add the missing models to the OpenAI models list
            config["providers"]["openai"]["models"] = openai_models + models_to_add
            # Save the updated configuration
            save_config(config)
            print(f"Added {', '.join(models_to_add)} to OpenAI models list")

    return config

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
        ValueError: If the provider for the model does not have an API key
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

    # Check if the provider has an API key
    if not has_api_key(provider_for_model):
        env_var = PROVIDER_API_KEYS.get(provider_for_model)
        if env_var:
            raise ValueError(f"Cannot use model '{model_name}' because the API key for {provider_for_model} is not set. "
                            f"Please set the {env_var} environment variable.")
        else:
            raise ValueError(f"Cannot use model '{model_name}' because the API key for {provider_for_model} is not available.")

    # Set the default model and its provider
    config["default_model"] = model_name
    config["default_provider"] = provider_for_model
    save_config(config)

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
    providers = get_available_providers()
    return {provider: has_api_key(provider) for provider in providers}

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
