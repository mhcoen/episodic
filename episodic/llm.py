"""
This module provides integration with various LLM providers through LiteLLM.
It handles sending queries to the APIs and processing the responses.
"""

import os
from typing import Dict, List, Optional, Any, Union
import litellm
from episodic.config import config
from episodic.llm_config import get_current_provider, get_provider_models, get_provider_config

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

    # For cloud providers, prepend the provider name if not already included
    if "/" not in model_name:
        return f"{provider}/{model_name}"

    return model_name

def query_llm(
    prompt: str, 
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> tuple[str, dict]:
    """
    Send a query to an LLM provider via LiteLLM and return the response.

    Args:
        prompt: The user's query to send to the LLM
        model: The model to use (default: gpt-4o-mini)
        system_message: The system message to set the context for the LLM
        temperature: Controls randomness (0-1, lower is more deterministic)
        max_tokens: Maximum number of tokens in the response

    Returns:
        A tuple containing:
        - The LLM's response as a string
        - A dictionary with cost information (input_tokens, output_tokens, total_tokens, cost_usd)

    Raises:
        Exception: If there's an error communicating with the LLM provider
    """
    try:
        # Create messages array
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # Print messages if debug is enabled
        if config.get("debug", False):
            print("\n=== DEBUG: Messages sent to LLM ===")
            for msg in messages:
                print(f"[{msg['role']}]: {msg['content']}")
            print("===================================\n")

        # Get the current provider
        provider = get_current_provider()

        # Get the full model string with provider prefix
        full_model = get_model_string(model)

        # Handle LMStudio specially since it needs an api_base
        if provider == "lmstudio":
            provider_config = get_provider_config("lmstudio")
            response = litellm.completion(
                model=full_model,
                messages=messages,
                api_base=provider_config.get("api_base"),
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # Call LiteLLM for other providers
            response = litellm.completion(
                model=full_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Extract cost information
        cost_info = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost_usd": getattr(response, "cost", 0.0)  # LiteLLM provides cost in USD
        }

        return response.choices[0].message.content, cost_info
    except Exception as e:
        raise Exception(f"Error querying LLM API: {str(e)}")

def query_with_context(
    prompt: str,
    context_messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> tuple[str, dict]:
    """
    Send a query to an LLM provider with conversation context and return the response.

    Args:
        prompt: The user's query to send to the LLM
        context_messages: List of previous messages in the conversation
                         [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        model: The model to use (default: gpt-4o-mini)
        system_message: The system message to set the context for the LLM
        temperature: Controls randomness (0-1, lower is more deterministic)
        max_tokens: Maximum number of tokens in the response

    Returns:
        A tuple containing:
        - The LLM's response as a string
        - A dictionary with cost information (input_tokens, output_tokens, total_tokens, cost_usd)

    Raises:
        Exception: If there's an error communicating with the LLM provider
    """
    try:
        # Prepare messages with system message first, then context, then the new prompt
        messages = [{"role": "system", "content": system_message}]
        messages.extend(context_messages)
        messages.append({"role": "user", "content": prompt})

        # Print messages if debug is enabled
        if config.get("debug", False):
            print("\n=== DEBUG: Messages sent to LLM ===")
            for msg in messages:
                print(f"[{msg['role']}]: {msg['content']}")
            print("===================================\n")

        # Get the current provider
        provider = get_current_provider()

        # Get the full model string with provider prefix
        full_model = get_model_string(model)

        # Handle LMStudio specially since it needs an api_base
        if provider == "lmstudio":
            provider_config = get_provider_config("lmstudio")
            response = litellm.completion(
                model=full_model,
                messages=messages,
                api_base=provider_config.get("api_base"),
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # Call LiteLLM for other providers
            response = litellm.completion(
                model=full_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Extract cost information
        cost_info = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost_usd": getattr(response, "cost", 0.0)  # LiteLLM provides cost in USD
        }

        return response.choices[0].message.content, cost_info
    except Exception as e:
        raise Exception(f"Error querying LLM API with context: {str(e)}")

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
