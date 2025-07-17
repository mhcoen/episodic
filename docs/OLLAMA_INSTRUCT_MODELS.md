# Ollama Instruct Models Guide

## Available Instruct Models in Ollama

### Recommended Instruct Models

1. **Mistral Instruct**
   ```bash
   ollama pull mistral:instruct
   ollama pull mistral:7b-instruct
   ```
   - Excellent for instructions
   - 7B parameters
   - Fast and efficient

2. **Llama 3 Instruct**
   ```bash
   ollama pull llama3:instruct
   ollama pull llama3:8b-instruct
   ollama pull llama3:70b-instruct
   ```
   - Meta's latest instruct model
   - Multiple sizes available
   - Great for all tasks

3. **Phi-3**
   ```bash
   ollama pull phi3
   ollama pull phi3:mini
   ollama pull phi3:medium
   ```
   - Microsoft's efficient model
   - Excellent instruction following
   - Very fast inference

4. **Gemma Instruct**
   ```bash
   ollama pull gemma:instruct
   ollama pull gemma:7b-instruct
   ```
   - Google's instruct model
   - Good for structured tasks

5. **Vicuna**
   ```bash
   ollama pull vicuna
   ```
   - Fine-tuned on instruction data
   - Good general performance

6. **Zephyr**
   ```bash
   ollama pull zephyr
   ```
   - HuggingFace's instruct model
   - Optimized for helpfulness

## Installing Instruct Models

```bash
# Install recommended models for Episodic
ollama pull llama3:instruct
ollama pull mistral:instruct
ollama pull phi3

# Verify installation
ollama list
```

## Configuring Episodic with Ollama Instruct Models

### Option 1: Edit config files before first run
Edit `~/.episodic/config.json` or the template files:

```json
{
  "topic_detection_model": "ollama/llama3:instruct",
  "compression_model": "ollama/mistral:instruct",
  "synthesis_model": "ollama/phi3"
}
```

### Option 2: Use commands after setup
```bash
# Set instruct models for all non-chat contexts
/model detection ollama/llama3:instruct
/model compression ollama/mistral:instruct
/model synthesis ollama/phi3

# Configure for optimal performance
/mset detection.temperature 0.1
/mset compression.temperature 0.3
/mset synthesis.temperature 0.5
```

## Model Selection by Task

### Topic Detection
- **Best**: `llama3:instruct` or `phi3`
- **Why**: Need consistent Yes/No answers
- **Config**: Very low temperature (0.1)

### Compression
- **Best**: `mistral:instruct` or `llama3:instruct`
- **Why**: Good at following summary structure
- **Config**: Low temperature (0.3)

### Synthesis
- **Best**: `mistral:instruct` or `zephyr`
- **Why**: Good at organizing information
- **Config**: Medium temperature (0.5)

## Performance Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| phi3 | 3.8B | Very Fast | Good | Detection |
| mistral:instruct | 7B | Fast | Excellent | All tasks |
| llama3:instruct | 8B | Fast | Excellent | All tasks |
| gemma:instruct | 7B | Fast | Good | Structured tasks |
| vicuna | 13B | Medium | Very Good | Complex tasks |
| zephyr | 7B | Fast | Good | Synthesis |

## Quick Setup Script

```bash
#!/bin/bash
# Install all recommended instruct models
ollama pull llama3:instruct
ollama pull mistral:instruct
ollama pull phi3

echo "Instruct models installed successfully!"
```

## Verifying Model Format

To check if a model supports instruct format:
```bash
ollama show <model-name>
```

Look for mentions of "instruct", "instruction", or specific prompt templates in the model card.