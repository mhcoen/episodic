"""
This module provides integration with various LLM providers through LiteLLM.
It handles sending queries to the APIs and processing the responses.
"""

import os
import logging
from typing import Dict, List, Any, Union
import litellm
from litellm.utils import supports_prompt_caching
from episodic.config import config
from episodic.llm_config import get_current_provider, get_provider_config
from episodic.llm_manager import llm_manager

# Set up logging
logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose output
litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "ERROR"  # Only show errors
litellm.set_verbose = False

# Also suppress the LiteLLM logger
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.ERROR)

# Also suppress the specific provider list messages

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
            logger.info("Prompt caching enabled for supported models (response caching disabled)")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize prompt caching: {str(e)}")
            return False
    else:
        # Ensure all caching is completely disabled
        litellm.cache = None
        logger.info("All caching disabled")
        return False

def load_api_keys_from_config():
    """
    Load API keys from config into environment variables for LiteLLM.
    
    This function checks the config for API keys and sets them as environment
    variables if they're not already set. This allows users to store API keys
    in the Episodic config instead of requiring environment variables.
    """
    # Map of config keys to environment variable names
    api_key_mapping = {
        'openai_api_key': 'OPENAI_API_KEY',
        'anthropic_api_key': 'ANTHROPIC_API_KEY', 
        'google_api_key': 'GOOGLE_API_KEY',
        'groq_api_key': 'GROQ_API_KEY',
        'together_api_key': 'TOGETHER_API_KEY',
        'mistral_api_key': 'MISTRAL_API_KEY',
        'cohere_api_key': 'COHERE_API_KEY',
        'deepseek_api_key': 'DEEPSEEK_API_KEY',
        'deepinfra_api_key': 'DEEPINFRA_API_KEY',
        'perplexity_api_key': 'PERPLEXITY_API_KEY',
        'fireworks_api_key': 'FIREWORKS_API_KEY',
        'anyscale_api_key': 'ANYSCALE_API_KEY',
        'replicate_api_key': 'REPLICATE_API_KEY',
        'huggingface_api_key': 'HUGGINGFACE_API_KEY',
        'ai21_api_key': 'AI21_API_KEY',
        'voyage_api_key': 'VOYAGE_API_KEY',
        'openrouter_api_key': 'OPENROUTER_API_KEY',
        'azure_api_key': 'AZURE_API_KEY',
        'azure_api_base': 'AZURE_API_BASE',
        'azure_api_version': 'AZURE_API_VERSION',
        'bedrock_access_key_id': 'AWS_ACCESS_KEY_ID',
        'bedrock_secret_access_key': 'AWS_SECRET_ACCESS_KEY',
        'bedrock_region': 'AWS_DEFAULT_REGION',
        'vertex_project': 'VERTEX_PROJECT',
        'vertex_location': 'VERTEX_LOCATION'
    }
    
    # Load each API key from config if not already in environment
    for config_key, env_var in api_key_mapping.items():
        if not os.environ.get(env_var):
            api_key = config.get(config_key)
            if api_key:
                os.environ[env_var] = api_key
                logger.debug(f"Loaded {env_var} from config")
    
    # Special handling for OpenRouter - set custom headers if configured
    if config.get('openrouter_api_key'):
        site_url = config.get('openrouter_site_url')
        app_name = config.get('openrouter_app_name', 'Episodic')
        if site_url or app_name:
            # These will be picked up by LiteLLM for OpenRouter requests
            if site_url:
                os.environ['OPENROUTER_SITE_URL'] = site_url
            if app_name:
                os.environ['OPENROUTER_APP_NAME'] = app_name

# Initialize cache on module load
initialize_cache()

# Load API keys from config
load_api_keys_from_config()

def disable_cache():
    """
    Disable the LiteLLM cache at runtime.
    """
    import litellm
    litellm.cache = None
    config.set("use_context_cache", False)
    logger.info("Cache disabled")

def enable_cache():
    """
    Enable the LiteLLM cache at runtime.
    """
    config.set("use_context_cache", True)
    result = initialize_cache()
    if result:
        logger.info("Cache enabled")
    else:
        logger.warning("Failed to enable cache")
    return result


def get_model_string(model_name: str) -> str:
    """
    Convert a model name to the appropriate format for the current provider.

    Args:
        model_name: The base model name

    Returns:
        The properly formatted model string for the current provider
    """
    # If model already has a provider prefix, return it as-is
    if "/" in model_name:
        return model_name
    
    # Find which provider actually has this model
    actual_provider = None
    from episodic.llm_config import get_available_providers, get_provider_models
    providers = get_available_providers()
    for provider_name, provider_config in providers.items():
        models = get_provider_models(provider_name)
        if models:
            for model in models:
                if isinstance(model, dict):
                    m_name = model.get("name", "unknown")
                else:
                    m_name = model
                if m_name == model_name:
                    actual_provider = provider_name
                    break
            if actual_provider:
                break
    
    # If we found the provider, use it; otherwise fall back to default
    if not actual_provider:
        actual_provider = get_current_provider()
    
    provider = actual_provider

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

    # For Google/Gemini models, use "gemini/" prefix for Google AI Studio
    elif provider == "google":
        # Google AI Studio expects "gemini/" prefix
        if not model_name.startswith("gemini/"):
            return f"gemini/{model_name}"
        return model_name
    
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
            logger.warning(f"Failed to apply prompt caching: {e}")
        return messages

def _execute_llm_query(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = None,
    stream: bool = False,
    **kwargs
) -> Union[tuple[str, dict], tuple[Any, None]]:
    """
    Internal function to execute LLM queries with prompt caching support.
    
    Args:
        messages: List of message dictionaries
        model: Model name
        temperature: Temperature for generation
        stream: If True, returns a generator for streaming responses
        
    Returns:
        If stream=False: (response_text, cost_info)
        If stream=True: (stream_generator, None) - cost info will be calculated during streaming
    """
    provider = get_current_provider()
    full_model = get_model_string(model)
    
    if config.get("debug", False):
        logger.debug(f"Model: {model} -> Full model string: {full_model} (Provider: {provider})")
    
    # Get model parameters from config
    main_params = config.get_model_params('main', model=full_model)
    
    # Merge provided kwargs with config params (kwargs take precedence)
    api_params = main_params.copy()
    api_params.update(kwargs)
    
    # Use provided temperature if given, otherwise use config
    if temperature is not None:
        api_params['temperature'] = temperature
    elif 'temperature' not in api_params:
        api_params['temperature'] = 0.7  # Default if not set

    # Apply prompt caching if enabled and supported
    if config.get("use_context_cache", True):
        messages = _apply_prompt_caching(messages, full_model)

    if config.get("debug", False):
        logger.debug("=== DEBUG: Messages sent to LLM ===")
        for msg in messages:
            logger.debug(f"[{msg['role']}]: {msg.get('content', msg)}")
        logger.debug("===================================")

    # Filter out unsupported parameters for Google/Gemini models
    if provider == "google":
        # Gemini doesn't support presence_penalty or frequency_penalty
        unsupported_params = ['presence_penalty', 'frequency_penalty']
        for param in unsupported_params:
            api_params.pop(param, None)
    
    # Filter out unsupported parameters for Ollama models
    if provider == "ollama" or "ollama/" in full_model.lower():
        # Ollama doesn't support presence_penalty or frequency_penalty
        unsupported_params = ['presence_penalty', 'frequency_penalty']
        for param in unsupported_params:
            api_params.pop(param, None)
    
    # Filter out unsupported parameters for Anthropic models
    if provider == "anthropic" or "claude" in full_model.lower() or "anthropic" in full_model.lower():
        # Anthropic doesn't support presence_penalty or frequency_penalty
        unsupported_params = ['presence_penalty', 'frequency_penalty']
        for param in unsupported_params:
            api_params.pop(param, None)
    
    # Provider-specific handling
    if provider == "lmstudio":
        provider_config = get_provider_config("lmstudio")
        response = llm_manager.make_api_call(
            messages=messages,
            model=full_model,
            stream=stream,
            api_base=provider_config.get("api_base"),
            **api_params
        )
    elif provider == "ollama":
        provider_config = get_provider_config("ollama")
        response = llm_manager.make_api_call(
            messages=messages,
            model=full_model,
            stream=stream,
            api_base=provider_config.get("api_base"),
            **api_params
        )
    else:
        response = llm_manager.make_api_call(
            messages=messages,
            model=full_model,
            stream=stream,
            **api_params
        )

    # If streaming, return the generator directly
    if stream:
        return response  # llm_manager returns (generator, None) for streaming

    # Non-streaming response handling - llm_manager returns (response_text, cost_info)
    response_text, cost_info = response
    return response_text, cost_info

def query_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = None,
    **kwargs
) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    return _execute_llm_query(messages, model, temperature, **kwargs)

def query_with_context(
    node_id: str,
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    context_depth: int = 5,
    stream: bool = False
) -> Union[tuple[str, dict], tuple[Any, None]]:
    """
    Query the LLM with context from the conversation history.

    Args:
        node_id: The ID of the node to use as the starting point for context
        model: The model to use for the query
        system_message: The system message to include at the beginning of the context
        temperature: The temperature to use for the query
        context_depth: The number of ancestor nodes to include in the context
        stream: If True, returns a streaming response

    Returns:
        If stream=False: A tuple of (response_text, cost_info)
        If stream=True: A tuple of (stream_generator, None)
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

    return _execute_llm_query(messages, model, temperature, stream=stream)

def process_stream_response(stream_generator, model: str):
    """
    Process a streaming response, yielding chunks as they come.
    
    Args:
        stream_generator: The streaming response generator from LiteLLM
        model: The model string for cost calculation
        
    Yields:
        Content chunks as they arrive from the stream
    """
    try:
        for chunk in stream_generator:
            if hasattr(chunk, 'choices') and chunk.choices:
                # Extract content from the chunk
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
    except GeneratorExit:
        # Handle early termination gracefully
        pass
    except Exception as e:
        logger.error(f"Error processing stream: {e}")
        raise


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
