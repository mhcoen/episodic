# Compression Model Selection Guide

## Overview

Compression/summarization in Episodic is handled by a dedicated model that can be configured separately from your main chat model. This allows you to optimize for the specific task of summarization.

## Recommended Models for Compression

### Best Choices (Instruct/Completion Models)

1. **GPT-3.5-turbo-instruct** (OpenAI)
   ```bash
   /model compression gpt-3.5-turbo-instruct
   ```
   - Optimized for following instructions
   - Good balance of speed and quality
   - Cost-effective

2. **Claude-3-haiku** (Anthropic)
   ```bash
   /model compression anthropic/claude-3-haiku-20240307
   ```
   - Fast and efficient
   - Good at structured summaries
   - Excellent instruction following

3. **Gemini 2.5 Flash** (via Google)
   ```bash
   /model compression google/gemini-2.5-flash
   ```
   - Fast and efficient
   - Good at structured summaries
   - Excellent comprehension

### Local Models (Ollama)

For privacy-conscious users or offline usage:

1. **Llama 3 Instruct**
   ```bash
   /model compression ollama/llama3:instruct
   ```

2. **Mistral Instruct**
   ```bash
   /model compression ollama/mistral:instruct
   ```

3. **Phi-3**
   ```bash
   /model compression ollama/phi3
   ```

## Why Instruct Models for Compression?

1. **Task-Specific Optimization**: Instruct models are fine-tuned to follow specific instructions, making them ideal for summarization tasks.

2. **Better Structure**: They produce more structured and organized summaries compared to chat models.

3. **Efficiency**: Instruct models often require fewer tokens to produce quality summaries.

4. **Consistency**: They provide more consistent output format and quality.

## Configuration

### Set Compression Model
```bash
/model compression <model-name>
```

### Configure Compression Parameters
```bash
# Set lower temperature for more focused summaries
/mset compression.temperature 0.3

# Limit summary length
/mset compression.max_tokens 500

# Reduce repetition
/mset compression.frequency_penalty 0.2
```

### View Current Settings
```bash
/model list
/mset compression
```

## Compression Prompt Customization

You can customize the compression prompt by editing:
`prompts/compression.md`

The prompt template supports variables:
- `{topic_name}` - The name of the topic being compressed
- `{conversation_text}` - The formatted conversation content

## Best Practices

1. **Use Instruct Models**: Prefer instruct or completion models over chat models
2. **Lower Temperature**: Use temperature 0.3-0.5 for consistent summaries
3. **Appropriate Length**: Set max_tokens based on your needs (300-800 typical)
4. **Test Different Models**: Different models excel at different types of content
5. **Monitor Costs**: Some models are more cost-effective for high-volume compression

## Example Setup

```bash
# Set an efficient instruct model for compression
/model compression gpt-3.5-turbo-instruct

# Configure for focused, concise summaries
/mset compression.temperature 0.3
/mset compression.max_tokens 400
/mset compression.frequency_penalty 0.2

# Enable automatic compression
/set auto_compress_topics true
/set compression_min_nodes 10
```

This setup will automatically compress topics with 10+ messages using an instruct model optimized for summarization.