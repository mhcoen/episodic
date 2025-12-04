# Troubleshooting Topic Drift Testing

## HTTP 429 Errors from HuggingFace

If you see errors like:
```
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2/resolve/main/tokenizer_config.json
```

This means HuggingFace is rate-limiting your requests. Solutions:

### Option 1: Pre-download the Model (Recommended)
```python
# Run this once to download the model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-mpnet-base-v2')
print("Model downloaded successfully!")
```

### Option 2: Use HuggingFace Token
1. Get a free token from https://huggingface.co/settings/tokens
2. Set environment variable:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

### Option 3: Use a Different Embedding Provider
```bash
# In episodic:
/set embedding_provider openai
/set embedding_model text-embedding-ada-002
```

### Option 4: Wait and Retry
The rate limit usually resets after a few minutes.

## Why These Errors Occur

1. **First-time model download** - Sentence transformers need to download model files
2. **Rate limiting** - HuggingFace limits anonymous requests
3. **Auto-compression** - When topics change, compression may trigger embedding calculations

## Preventing Issues

1. **Pre-download models before testing**
2. **Use a HuggingFace account token**
3. **Disable auto-compression during testing**:
   ```bash
   /set comp-auto false
   ```