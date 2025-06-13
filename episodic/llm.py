"""
This module provides integration with various LLM providers through LiteLLM.
It handles sending queries to the APIs and processing the responses.
"""

import os
from typing import Dict, List, Optional, Any, Union
import litellm
from episodic.config import config
from episodic.llm_config import get_current_provider, get_provider_models, get_provider_config

def calculate_cost(response, model_name):
    """
    Calculate the cost of an LLM request based on token usage.

    Args:
        response: The response from litellm.completion
        model_name: The full model name including provider prefix

    Returns:
        The calculated cost in USD
    """
    # Try to get cost from response first
    cost = getattr(response, "cost", None)
    if cost is not None:
        return cost

    # If cost is not available in response, calculate it manually
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # Get model pricing from litellm
    model_pricing = litellm.model_prices.get(model_name)
    if model_pricing:
        input_cost = input_tokens * model_pricing.get("input_cost_per_token", 0)
        output_cost = output_tokens * model_pricing.get("output_cost_per_token", 0)
        return input_cost + output_cost

    # If no pricing information is available, try to estimate based on model name
    if "gpt-4" in model_name:
        # GPT-4 family pricing (approximate)
        return (input_tokens * 0.00003) + (output_tokens * 0.00006)
    elif "gpt-3.5" in model_name:
        # GPT-3.5 family pricing (approximate)
        return (input_tokens * 0.000001) + (output_tokens * 0.000002)

    # Default to 0 if we can't determine the cost
    return 0.0

litellm.model_prices = {
    # OpenAI models
    "openai/gpt-4o-mini": {
        "input_cost_per_token": 0.00001,  # $0.01 per 1K tokens
        "output_cost_per_token": 0.00003,  # $0.03 per 1K tokens
    },
    "openai/gpt-4o": {
        "input_cost_per_token": 0.00005,  # $0.05 per 1K tokens
        "output_cost_per_token": 0.00015,  # $0.15 per 1K tokens
    },
    "openai/gpt-3.5-turbo": {
        "input_cost_per_token": 0.0000005,  # $0.0005 per 1K tokens
        "output_cost_per_token": 0.0000015,  # $0.0015 per 1K tokens
    },

    # Anthropic models
    "anthropic/claude-3-opus-20240229": {
        "input_cost_per_token": 0.00001,  # $0.01 per 1K tokens
        "output_cost_per_token": 0.00003,  # $0.03 per 1K tokens
    },
    "anthropic/claude-3-sonnet-20240229": {
        "input_cost_per_token": 0.000003,  # $0.003 per 1K tokens
        "output_cost_per_token": 0.000015,  # $0.015 per 1K tokens
    },
    "anthropic/claude-3-haiku-20240307": {
        "input_cost_per_token": 0.00000025,  # $0.00025 per 1K tokens
        "output_cost_per_token": 0.00000125,  # $0.00125 per 1K tokens
    },

    # Groq models
    "groq/llama3-8b-8192": {
        "input_cost_per_token": 0.0000002,  # $0.0002 per 1K tokens
        "output_cost_per_token": 0.0000002,  # $0.0002 per 1K tokens
    },
    "groq/llama3-70b-8192": {
        "input_cost_per_token": 0.0000007,  # $0.0007 per 1K tokens
        "output_cost_per_token": 0.0000007,  # $0.0007 per 1K tokens
    },
    "groq/mixtral-8x7b-32768": {
        "input_cost_per_token": 0.0000006,  # $0.0006 per 1K tokens
        "output_cost_per_token": 0.0000006,  # $0.0006 per 1K tokens
    },
    "groq/gemma-7b-it": {
        "input_cost_per_token": 0.0000001,  # $0.0001 per 1K tokens
        "output_cost_per_token": 0.0000001,  # $0.0001 per 1K tokens
    }
}

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
            "cost_usd": calculate_cost(response, full_model)  # Calculate cost based on token usage
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
            "cost_usd": calculate_cost(response, full_model)  # Calculate cost based on token usage
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
