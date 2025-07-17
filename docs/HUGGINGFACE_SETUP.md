# HuggingFace Integration Guide

## Setting up HuggingFace API Key

1. Get your API key from [HuggingFace](https://huggingface.co/settings/tokens)
2. Set the environment variable:
   ```bash
   export HUGGINGFACE_API_KEY="your-api-key-here"
   ```

## Important Note on Model Availability

The HuggingFace Inference API (which LiteLLM uses) has limited model availability. Many popular models like Qwen 2.5, Llama 3.2, Falcon 3, and DeepSeek are not available through this API, even though they exist on HuggingFace.

## Available Models

### Working Chat Models (Verified)
These models have been tested and work correctly with Episodic through the HF Inference API:

1. **Falcon 7B Instruct** - Model #109
   - Fast, general-purpose instruction-following model
   - Good for quick responses

2. **BLOOM 176B** - Model #110
   - Large multilingual model
   - Good for diverse language tasks

3. **GPT-NeoX 20B** - Model #111
   - Open-source GPT-style model
   - Good balance of performance and quality

4. **StableLM 7B** - Model #112
   - Stability AI's language model
   - Optimized for helpful responses

### Models Listed But Not Available
These well-regarded models are included in the list but don't work with the HF Inference API:

- ❌ Qwen 2.5 series - "model not supported"
- ❌ Llama 3.2/3.3 models - "model not supported"
- ❌ Falcon 3 series - "model not supported"
- ❌ Falcon 40B - "not a chat model" error
- ❌ DeepSeek models - "model not supported"
- ❌ Yi models - "model not supported"
- ❌ MiniChat models - "model not supported"
- ❌ Mistral v0.3 - "model not supported"

### Usage Examples

```bash
# Set a working HuggingFace model
/model chat 31  # Sets Falcon 7B Instruct

# Or use the full name
/model chat huggingface/tiiuae/falcon-7b-instruct

# Check available models
/model list
```

## Pricing

HuggingFace uses a subscription model instead of per-token pricing:

- **Free Tier**: ~30,000 tokens/month
- **Pro Tier**: $9/month for unlimited usage

The tier information is displayed when viewing models or costs.

## Troubleshooting

### "Not a chat model" Error
Some HuggingFace models (like Falcon 40B) don't support the chat completion endpoint. Use one of the verified working models listed above.

### Authentication Errors
Make sure your `HUGGINGFACE_API_KEY` environment variable is set correctly. You can verify with:
```bash
echo $HUGGINGFACE_API_KEY
```

### LiteLLM Async Warning
The warning about `close_litellm_async_clients` is harmless and doesn't affect functionality. It's a cleanup issue in the LiteLLM library.
EOF < /dev/null