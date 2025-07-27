# OpenRouter Integration

Episodic now supports OpenRouter, which provides access to multiple LLM providers through a single API.

## Setup

1. **Get an OpenRouter API key**
   - Sign up at https://openrouter.ai
   - Create an API key at https://openrouter.ai/keys

2. **Set your API key in Episodic**
   ```
   /set openrouter-api-key YOUR_OPENROUTER_API_KEY
   ```

3. **List available models**
   ```
   /model list
   ```
   
   You'll see OpenRouter models listed like:
   - `openrouter/anthropic/claude-3-opus`
   - `openrouter/openai/gpt-4-turbo-preview`
   - `openrouter/google/gemini-pro`
   - `openrouter/meta-llama/llama-2-70b-chat`

4. **Select an OpenRouter model**
   ```
   /model chat openrouter/anthropic/claude-3-opus
   ```

## Benefits

- **Single API key**: Access models from multiple providers
- **Automatic fallback**: OpenRouter can automatically fall back to other models if one is unavailable
- **Often better pricing**: OpenRouter sometimes offers better rates than going direct
- **No need for multiple API keys**: One OpenRouter key replaces many provider keys

## Available Models

OpenRouter provides access to models from:
- Anthropic (Claude 3 family)
- OpenAI (GPT-4, GPT-3.5)
- Google (Gemini Pro)
- Meta (Llama 2)
- Mistral AI
- Cohere
- And many more

## Configuration Options

Optional settings for OpenRouter:

```bash
# Set your app's URL (for OpenRouter analytics)
/set openrouter-site-url https://your-app.com

# Set your app name
/set openrouter-app-name "My App"
```

## Pricing

OpenRouter shows transparent pricing for each model. Use `/model list` to see current prices per 1M tokens for each model.

## Tips

1. **Model Selection**: OpenRouter models always start with `openrouter/` prefix
2. **Cost Tracking**: Episodic's cost tracking works with OpenRouter models
3. **Streaming**: Full streaming support for all OpenRouter models
4. **Context Caching**: Supported for compatible models

## Troubleshooting

If models don't appear after setting your API key:
1. Restart Episodic
2. Check your API key is valid at https://openrouter.ai/keys
3. Ensure you have credits in your OpenRouter account