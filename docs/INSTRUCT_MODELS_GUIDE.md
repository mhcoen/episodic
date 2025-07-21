# Instruct Models Guide for Episodic

## Overview

Episodic uses different models for different tasks. While chat models are great for conversations, **instruct models** are better for specific tasks like detection, compression, and synthesis.

## Model Contexts in Episodic

### 1. Chat Context (Conversational)
- **Purpose**: Main conversation with the user
- **Best Models**: Chat models (GPT-4o, Claude, etc.)
- **Example**: `/model chat gpt-4o-mini`

### 2. Detection Context (Instruction-Following)
- **Purpose**: Detect topic changes (binary Yes/No classification)
- **Best Models**: Instruct models
- **Why**: More consistent at following specific classification instructions
- **Recommended**:
  ```bash
  /model detection gpt-3.5-turbo-instruct
  /model detection ollama/llama3:instruct
  /model detection anthropic/claude-3-haiku-20240307
  ```

### 3. Compression Context (Instruction-Following)
- **Purpose**: Summarize conversation segments
- **Best Models**: Instruct models
- **Why**: Better at structured summarization tasks
- **Recommended**:
  ```bash
  /model compression gpt-3.5-turbo-instruct
  /model compression ollama/mistral:instruct
  /model compression google/gemini-2.5-flash
  ```

### 4. Synthesis Context (Instruction-Following)
- **Purpose**: Synthesize web search results into coherent answers
- **Best Models**: Instruct models
- **Why**: Better at following synthesis guidelines and formatting
- **Recommended**:
  ```bash
  /model synthesis gpt-3.5-turbo-instruct
  /model synthesis anthropic/claude-3-haiku-20240307
  /model synthesis ollama/phi3
  ```

## Why Use Instruct Models?

### For Topic Detection
- **Consistency**: Instruct models give more consistent Yes/No answers
- **Less Chatty**: Won't try to explain their reasoning
- **Better Classification**: Optimized for decision tasks
- **Lower Cost**: Can use smaller, faster models

### For Compression
- **Structured Output**: Better at following summary templates
- **Conciseness**: Less likely to be verbose
- **Task Focus**: Won't add conversational elements
- **Quality**: Better at extracting key information

### For Synthesis
- **Format Following**: Better at markdown formatting requirements
- **Source Integration**: More reliable at citing sources
- **Objectivity**: Less likely to add opinions
- **Structure**: Better at organizing information

## Optimal Configuration Examples

### High Quality Setup
```bash
# Chat: Best conversational model
/model chat gpt-4o

# Detection: Fast, accurate instruct model
/model detection gpt-3.5-turbo-instruct
/mset detection.temperature 0.1
/mset detection.max_tokens 10

# Compression: Quality instruct model
/model compression gpt-3.5-turbo-instruct
/mset compression.temperature 0.3
/mset compression.max_tokens 500

# Synthesis: High-quality instruct model
/model synthesis gpt-3.5-turbo-instruct
/mset synthesis.temperature 0.5
/mset synthesis.max_tokens 1000
```

### Cost-Effective Local Setup
```bash
# Chat: Good local model
/model chat ollama/llama3

# Detection: Fast local instruct
/model detection ollama/llama3:instruct
/mset detection.temperature 0

# Compression: Local instruct
/model compression ollama/mistral:instruct
/mset compression.temperature 0.3

# Synthesis: Local instruct
/model synthesis ollama/phi3
/mset synthesis.temperature 0.5
```

### Privacy-Focused Setup
```bash
# All local models
/model chat ollama/llama3
/model detection ollama/llama3:instruct
/model compression ollama/mistral:instruct
/model synthesis ollama/phi3
```

## Model Parameters by Context

### Detection Parameters
```bash
/mset detection.temperature 0.1    # Very low for consistency
/mset detection.max_tokens 10      # Just need Yes/No
/mset detection.top_p 0.9          # Slight variation allowed
```

### Compression Parameters
```bash
/mset compression.temperature 0.3   # Low for focused summaries
/mset compression.max_tokens 500    # Reasonable summary length
/mset compression.frequency_penalty 0.2  # Reduce repetition
```

### Synthesis Parameters
```bash
/mset synthesis.temperature 0.5     # Moderate for coherent synthesis
/mset synthesis.max_tokens 1000     # Comprehensive answers
/mset synthesis.presence_penalty 0.1  # Encourage covering all points
```

## Quick Commands

View all current models:
```bash
/model
```

Set all to instruct models:
```bash
/model detection gpt-3.5-turbo-instruct
/model compression gpt-3.5-turbo-instruct
/model synthesis gpt-3.5-turbo-instruct
```

View model-specific parameters:
```bash
/mset detection
/mset compression
/mset synthesis
```

## Best Practices

1. **Use Chat Models for Chat**: Keep conversational models for the main chat
2. **Use Instruct for Tasks**: All other contexts benefit from instruct models
3. **Lower Temperature**: Task-oriented contexts need lower temperatures
4. **Test Different Models**: Some instruct models are better at specific tasks
5. **Monitor Costs**: Instruct models are often cheaper than chat models
6. **Local Options**: Ollama provides good local instruct variants

## Common Issues

### "Not a chat model" Error
Some models like `gpt-3.5-turbo-instruct` may not work in chat context but excel in task contexts.

### Inconsistent Detection
If topic detection is inconsistent, try:
- Lower temperature (0.0-0.2)
- Different instruct model
- Check the detection prompt

### Verbose Summaries
If compression is too verbose:
- Use instruct model instead of chat
- Lower max_tokens
- Increase frequency_penalty