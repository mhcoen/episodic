# Setting up Instruct Models with Ollama

## Install Instruct Models

Run these commands to pull the instruct models:

```bash
# Mistral Instruct (7B) - Recommended for good balance of speed and quality
ollama pull mistral:instruct

# Llama 3 Instruct (8B) - Latest Meta model with instruct tuning
ollama pull llama3:instruct

# Llama 3.1 Instruct (8B) - Updated version
ollama pull llama3.1:instruct

# Smaller/Faster options:
ollama pull phi3:instruct      # Microsoft Phi-3 (3.8B) - Very fast
ollama pull gemma2:instruct    # Google Gemma 2 (2B/9B) - Good quality

# Larger/Better options (if you have the RAM):
ollama pull llama3:70b-instruct    # 70B version - needs ~40GB RAM
ollama pull mixtral:instruct        # Mixtral 8x7B - needs ~26GB RAM
```

## Check Installed Models

```bash
# List all installed models
ollama list

# Test a model
ollama run mistral:instruct "Rate the topic drift between 'How's the weather?' and 'What's 2+2?' on a scale of 0 to 1. Respond with only a number."
```

## Recommended Models for Topic Detection

1. **mistral:instruct** - Best balance of speed and accuracy
2. **llama3:instruct** or **llama3.1:instruct** - Excellent instruction following
3. **phi3:instruct** - Very fast for real-time detection
4. **gemma2:instruct** - Good alternative with different training

## Memory Requirements

- phi3:instruct: ~2GB
- gemma2:2b-instruct: ~1.5GB  
- mistral:instruct: ~4GB
- llama3:instruct: ~4.5GB
- mixtral:instruct: ~26GB
- llama3:70b-instruct: ~40GB

## Usage Tips

1. Start with `mistral:instruct` - it's fast and accurate
2. Use `phi3:instruct` if you need maximum speed
3. Use `llama3:instruct` for best instruction following
4. Test different models to see which works best for your use case