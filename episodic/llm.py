"""
This module provides integration with various LLM providers through LiteLLM.
It handles sending queries to the APIs and processing the responses.
"""

import os
from typing import Dict, List, Optional, Any, Union
import litellm
from litellm import cost_per_token
from litellm.caching import Cache
from litellm.utils import supports_prompt_caching
from episodic.config import config
from episodic.llm_config import get_current_provider, get_provider_models, get_provider_config

# Set default configuration value for context caching (enabled by default)
if config.get("use_context_cache") is None:
    config.set("use_context_cache", True)

def initialize_cache():
    """
    Initialize LiteLLM for prompt caching to reduce token usage.
    Disables response caching in favor of prompt caching.
    """
    if config.get("use_context_cache", True):
        try:
            # Disable response caching - we only want prompt caching
            litellm.cache = None
            
            # Prompt caching is handled per-request via cache_control parameters
            # in the _apply_prompt_caching function. No global setup needed.
            print("Prompt caching enabled for supported models (response caching disabled)")
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize prompt caching: {str(e)}")
            return False
    else:
        # Ensure all caching is completely disabled
        litellm.cache = None
        print("All caching disabled")
        return False

# Initialize cache on module load
initialize_cache()

def disable_cache():
    """
    Disable the LiteLLM cache at runtime.
    """
    import litellm
    litellm.cache = None
    config.set("use_context_cache", False)
    print("Cache disabled")

def enable_cache():
    """
    Enable the LiteLLM cache at runtime.
    """
    config.set("use_context_cache", True)
    result = initialize_cache()
    if result:
        print("Cache enabled")
    else:
        print("Failed to enable cache")
    return result


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

def _apply_prompt_caching(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """
    Apply prompt caching to system messages if supported by the model.
    Also filters out empty system messages for Anthropic models.
    
    Args:
        messages: List of message dictionaries
        model: The model string being used
        
    Returns:
        Modified messages list with cache_control applied for Anthropic models
    """
    try:
        # Check if this is an Anthropic model
        is_anthropic = "anthropic" in model.lower() or "claude" in model.lower()
        
        # For non-Anthropic models, check if they support prompt caching
        if not is_anthropic and not supports_prompt_caching(model=model):
            return messages
            
        # For OpenAI models, prompt caching is automatic - no modifications needed
        if not is_anthropic:
            return messages
            
        # Create a copy of messages to avoid modifying the original
        cached_messages = []
        
        # Check if the model actually supports prompt caching
        supports_caching = supports_prompt_caching(model=model)
        
        for msg in messages:
            if msg.get("role") == "system":
                # Only process non-empty system messages for Anthropic models
                content = msg.get("content", "")
                if content and content.strip():
                    if supports_caching:
                        # Apply cache control if supported
                        cached_msg = {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": content,
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ]
                        }
                        cached_messages.append(cached_msg)
                    else:
                        # Keep as simple message without cache control
                        cached_messages.append(msg)
                # Skip empty system messages entirely for Anthropic models
                # (Anthropic doesn't allow empty text content blocks)
            else:
                # Keep other messages unchanged
                cached_messages.append(msg)
                
        return cached_messages
        
    except Exception as e:
        # If prompt caching fails, return original messages
        if config.get("debug", False):
            print(f"Warning: Failed to apply prompt caching: {e}")
        return messages

def _execute_llm_query(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int
) -> tuple[str, dict]:
    """
    Internal function to execute LLM queries with prompt caching support.
    """
    provider = get_current_provider()
    full_model = get_model_string(model)

    # Apply prompt caching if enabled and supported
    if config.get("use_context_cache", True):
        messages = _apply_prompt_caching(messages, full_model)

    if config.get("debug", False):
        print("\n=== DEBUG: Messages sent to LLM ===")
        for msg in messages:
            print(f"[{msg['role']}]: {msg.get('content', msg)}")
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

    # Check for cached tokens in the response (OpenAI prompt caching)
    cached_tokens = 0
    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
        cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
    
    # Calculate actual cost considering cached tokens (cached tokens are discounted)
    actual_prompt_tokens = response.usage.prompt_tokens - cached_tokens
    actual_cost = sum(cost_per_token(
        model=full_model,
        prompt_tokens=actual_prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    ))
    
    # Add cost savings from caching
    if cached_tokens > 0:
        # Cached tokens typically cost 50% less
        cached_cost = sum(cost_per_token(
            model=full_model,
            prompt_tokens=cached_tokens,
            completion_tokens=0
        )) * 0.5  # 50% discount for cached tokens
        total_cost_with_cache = actual_cost + cached_cost
    else:
        total_cost_with_cache = actual_cost
    
    cost_info = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost_usd": total_cost_with_cache
    }
    
    # Add cache information if available
    if cached_tokens > 0:
        cost_info["cached_tokens"] = cached_tokens
        cost_info["non_cached_tokens"] = actual_prompt_tokens
        cost_info["cache_savings_usd"] = total_cost - total_cost_with_cache
        if config.get("debug", False):
            print(f"🎯 Cache hit! {cached_tokens} tokens cached, {actual_prompt_tokens} new tokens")
            print(f"💰 Cost savings: ${total_cost - total_cost_with_cache:.6f}")
    
    return response.choices[0].message.content, cost_info

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
    node_id: str,
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    context_depth: int = 5
) -> tuple[str, dict]:
    """
    Query the LLM with context from the conversation history.

    Args:
        node_id: The ID of the node to use as the starting point for context
        model: The model to use for the query
        system_message: The system message to include at the beginning of the context
        temperature: The temperature to use for the query
        max_tokens: The maximum number of tokens to generate
        context_depth: The number of ancestor nodes to include in the context

    Returns:
        A tuple of (response_text, cost_info)
    """
    from episodic.db import get_node, get_ancestry

    # Get the node and its content
    node = get_node(node_id)
    if not node:
        raise ValueError(f"Node {node_id} not found")

    prompt = node["content"]

    # Get the ancestry of the node
    ancestry = get_ancestry(node_id)

    # Limit the ancestry to the specified depth
    if context_depth > 0:
        ancestry = ancestry[-context_depth:]

    # Build the context messages from the ancestry
    context_messages = []
    for ancestor in ancestry[:-1]:  # Exclude the current node
        context_messages.append({
            "role": ancestor.get("role", "user"),
            "content": ancestor.get("content", "")
        })

    # Build the messages list
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
