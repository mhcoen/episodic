"""
This module provides integration with various LLM providers through LiteLLM.
It handles sending queries to the APIs and processing the responses.
"""

import os
from typing import Dict, List, Optional, Any, Union
import litellm
from litellm import cost_per_token
from litellm.caching import Cache
from episodic.config import config
from episodic.llm_config import get_current_provider, get_provider_models, get_provider_config

# Replace the existing caching initialization with this:
from litellm import Cache

# Initialize context caching with in-memory cache
# This will be enabled by default but can be toggled via CLI
if config.get("use_context_cache", True):
    try:
        cache = Cache(
            type="local",
            cache_responses=True,
            cache_time=3600  # Cache entries expire after 1 hour
        )
        litellm.cache = cache
    except Exception as e:
        print(f"Warning: Failed to initialize cache: {str(e)}")

# Default setting for context caching (enabled by default)
if config.get("use_context_cache") is None:
    config.set("use_context_cache", True)


def get_model_string(model_name: str) -> str:
    """
    Convert a model name to the appropriate format for the current provider.

    Args:
        model_name: The base model name

    Returns:
        The properly formatted model string for the current provider
    """
    provider = get_current_provider()

    # For local models, use the format "local/model_name"
    if provider == "local":
        local_models = get_provider_models("local")
        for model in local_models:
            if model.get("name") == model_name:
                backend = model.get("backend", "llama.cpp")
                return f"{backend}/{model_name}"

        # If model not found in local models, raise error
        raise ValueError(f"Local model '{model_name}' not configured")

    # For LMStudio, the model name doesn't matter as much
    elif provider == "lmstudio":
        return model_name

    # For Ollama, prepend the provider name
    elif provider == "ollama":
        return f"ollama/{model_name}"

    # For cloud providers, prepend the provider name if not already included
    if "/" not in model_name:
        return f"{provider}/{model_name}"

    return model_name

def _execute_llm_query(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int
) -> tuple[str, dict]:
    """
    Internal function to execute LLM queries.
    """
    provider = get_current_provider()
    full_model = get_model_string(model)

    if config.get("debug", False):
        print("\n=== DEBUG: Messages sent to LLM ===")
        for msg in messages:
            print(f"[{msg['role']}]: {msg['content']}")
        print("===================================\n")

    # Provider-specific handling
    if provider == "lmstudio":
        provider_config = get_provider_config("lmstudio")
        response = litellm.completion(
            model=full_model,
            messages=messages,
            api_base=provider_config.get("api_base"),
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "ollama":
        provider_config = get_provider_config("ollama")
        response = litellm.completion(
            model=full_model,
            messages=messages,
            api_base=provider_config.get("api_base"),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
    else:
        response = litellm.completion(
            model=full_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    total_cost = sum(cost_per_token(
        model=full_model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    ))

    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost_usd": total_cost
    }

def query_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    return _execute_llm_query(messages, model, temperature, max_tokens)

def query_with_context(
    prompt: str,
    context_messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> tuple[str, dict]:
    messages = [{"role": "system", "content": system_message}]
    messages.extend(context_messages)
    messages.append({"role": "user", "content": prompt})
    return _execute_llm_query(messages, model, temperature, max_tokens)

# Backward compatibility for code that might be using get_openai_client directly
def get_openai_client():
    """
    Initialize and return an OpenAI client using the API key from environment variables.

    This function is maintained for backward compatibility.

    Returns:
        OpenAI client object

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    return openai.OpenAI(api_key=api_key)