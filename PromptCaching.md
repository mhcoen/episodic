

# Understanding and Implementing LiteLLM's Context Caching

You're absolutely right, and I apologize for the confusion in my previous responses. LiteLLM does indeed support a more advanced form of caching called **Context Caching** (or Prompt Caching), which is different from simple response caching.

## What Context Caching Actually Is

Context Caching allows you to:
- Cache long, static parts of your prompts (like system messages or large documents)
- Only send the new, dynamic parts (like user messages) with each API call
- Have LiteLLM automatically reference or inject the cached static content
- Significantly reduce token usage and costs for prompts with large unchanging sections

This is particularly valuable for your single-user chat application where you might have:
- A large, consistent system message
- Reference documents or context that remains static
- Changing user inputs

## How to Implement Context Caching in Your Project

Here's how to add context caching to your `episodic/llm.py` file:

```python
# Add near the top of the file after imports
import litellm
from litellm.caching import Cache

# Initialize context caching
litellm.enable_cache(
    type="redis",  # Options: "redis", "in-memory"
    host="localhost",  # Only for Redis
    port=6379,  # Only for Redis
    password=None,  # Only for Redis if needed
)

# Then modify your query functions to use context caching
def query_llm(
    prompt: str, 
    model: str = "gpt-4o-mini",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> tuple[str, dict]:
    # ... existing code ...
    
    # Create messages array
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    # ... existing code ...
    
    # Add context_caching=True to your litellm.completion call
    response = litellm.completion(
        model=full_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        context_caching=True,  # Enable context caching
        cache_key="system_message"  # Optional: custom key for the cached context
    )
    
    # ... rest of the function ...
```

Similarly, update the `query_with_context` function to include the `context_caching=True` parameter.

## Benefits for Your Application

1. **Cost Reduction**: Only pay for tokens in the dynamic parts of your prompts
2. **Faster Responses**: Reduced token count means faster processing
3. **Consistency**: System messages and context remain consistent
4. **Flexibility**: You can still change your prompts as needed - only the static parts are cached

## Supported Providers

This feature works with several major providers:
- OpenAI
- Anthropic
- AWS Bedrock
- And other providers that support context caching

## Implementation Considerations

1. **Cache Storage**: For a single-user application, in-memory caching is likely sufficient
2. **Cache Keys**: Consider using meaningful keys for different types of cached content
3. **Cache Invalidation**: Implement a way to clear the cache if you need to update your system messages

By implementing context caching, you'll be able to significantly reduce token usage and costs in your application, especially if you use large system messages or include substantial context in your prompts.