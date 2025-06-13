# Integrating LLM Providers with LiteLLM

This guide explains how to integrate various LLM providers with LiteLLM in your project, including both cloud providers and local options like Ollama and LMStudio.

## Ollama Integration

Ollama is directly supported by LiteLLM and is straightforward to integrate:

```python
# In your llm_config.py
def get_default_config() -> Dict[str, Any]:
    return {
        "default_provider": "openai",
        "providers": {
            "openai": {
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            },
            "ollama": {
                "models": ["llama3", "mistral", "codellama", "phi3"]
            },
            # Other providers...
        }
    }
```

To use Ollama models:
1. Install Ollama from [ollama.ai](https://ollama.ai/download)
2. Pull models you want to use: `ollama pull llama3`
3. Install LiteLLM with Ollama support: `pip install litellm[ollama]`
4. In your code, use the format: `ollama/model_name`

```python
# Example usage
response = litellm.completion(
    model="ollama/llama3",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

## LMStudio Integration

LMStudio works by exposing an OpenAI-compatible API endpoint, so you'll integrate it as a custom endpoint:

1. In LMStudio:
   - Load your model
   - Start the local server (usually at http://localhost:1234)

2. Configure LiteLLM to use LMStudio:

```python
# In your llm_config.py
def get_default_config() -> Dict[str, Any]:
    return {
        "default_provider": "openai",
        "providers": {
            # Other providers...
            "lmstudio": {
                "api_base": "http://localhost:1234/v1",  # Default LMStudio endpoint
                "models": ["local-model"]  # You can name these whatever makes sense
            }
        }
    }
```

3. In your `get_model_string` function, add special handling for LMStudio:

```python
def get_model_string(model_name: str) -> str:
    """Convert model name to LiteLLM format based on provider."""
    provider = get_current_provider()
    
    # Handle different provider types
    if provider == "lmstudio":
        # For LMStudio, we need to set the API base in the call
        return model_name  # The actual model name doesn't matter for LMStudio
```

4. When making calls to LMStudio, set the API base:

```python
# In your query_llm function
if provider == "lmstudio":
    provider_config = get_provider_config("lmstudio")
    response = litellm.completion(
        model=full_model,
        messages=messages,
        api_base=provider_config.get("api_base"),
        temperature=temperature,
        max_tokens=max_tokens
    )
```

## Groq Integration

Groq is a cloud provider known for its fast inference speeds. It's directly supported by LiteLLM:

```python
# In your llm_config.py
def get_default_config() -> Dict[str, Any]:
    return {
        "default_provider": "openai",
        "providers": {
            # Other providers...
            "groq": {
                "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
            },
        }
    }
```

To use Groq:
1. Sign up for an account at [groq.com](https://console.groq.com/signup)
2. Get your API key from the Groq console
3. Set the environment variable: `export GROQ_API_KEY=your_groq_api_key_here`
4. Use the models with the format: `groq/model_name`

```python
# Example usage
response = litellm.completion(
    model="groq/llama3-8b-8192",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

Groq is known for its extremely fast inference speeds, making it a good choice for applications where response time is critical.

## CLI Command for Switching Between Providers

Add a command to easily switch between your providers:

```python
# In your handle_llm_providers method
elif subcommand == "local" and len(args) > 1:
    local_provider = args[1]
    if local_provider in ["ollama", "lmstudio"]:
        try:
            set_current_provider(local_provider)
            print(f"Switched to local provider: {local_provider}")
        except ValueError as e:
            print(f"Error: {str(e)}")
    else:
        print(f"Unknown local provider: {local_provider}")
        print("Available local providers: ollama, lmstudio")
```

## Installation Requirements

```bash
# For Ollama support
pip install litellm[ollama]

# For LMStudio, no special package needed as it uses OpenAI-compatible API

# For cloud providers (OpenAI, Anthropic, Groq, etc.)
# No special packages needed beyond the base litellm installation
```

This setup gives you flexibility to switch between cloud providers (OpenAI, Anthropic, Groq) and your local providers (Ollama and LMStudio) while maintaining a consistent interface throughout your application.