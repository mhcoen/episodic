# Model Configuration Guide

## Overview

Episodic uses a JSON-based model configuration system that allows you to:
- View all available models with their types and parameters
- Add custom models
- Override model properties
- Configure provider-specific settings

## Current Models (2025)

### Latest Available Models

**OpenAI:**
- GPT-5.2 (December 2025) - Latest flagship model
- GPT-5 (August 2025)
- o3 (June 2025) - Advanced reasoning model
- o4-mini (April 2025) - Compact reasoning model
- GPT-4o, GPT-4o Mini, GPT-3.5 Turbo

**Anthropic:**
- Claude Opus 4.1 (August 2025)
- Claude Sonnet 4.5 (September 2025)
- Claude Sonnet 4 (May 2025)
- Claude Haiku 4.5 (October 2025)

**Google:**
- Gemini 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite

**Ollama (Local):**
- Llama 4 Scout and Maverick (April 2025)
- Llama 3.3, Llama 3
- DeepSeek R1 (January 2025)
- Mistral, Phi-4, and others

**OpenRouter:**
- Provides access to models from multiple providers including:
  - Latest Anthropic and OpenAI models
  - Llama 4 Scout and Maverick
  - DeepSeek R1 and Chat V3
  - And many more

**HuggingFace:**
- Llama 4 Scout and Maverick
- Llama 3.3, Llama 3, Llama 2
- Qwen 3, Mistral, DeepSeek, and others

### Deprecated Models

The following models have been removed or deprecated:
- **Claude 3.5 Sonnet** - Deprecated August 2025, retired November 2025
- **Claude 3.5 Haiku** - Deprecated and removed November 2025
- **Claude 3 Opus** - Deprecated, retiring January 2026 (still available)

## Configuration Files

### Models Configuration: `~/.episodic/models.json`
This file contains all model definitions for Episodic. It includes:
- Provider configurations
- Model definitions with types, parameters, and context windows
- Type detection patterns
- Type indicators for display

When you first run Episodic, it will create this file from the built-in template (`episodic/models_template.json`).

You can edit this file to:
- Add new providers
- Add models to existing providers
- Modify model properties
- Customize type detection patterns

## Model Configuration Examples

### Using New 2025 Models

```bash
# Use latest OpenAI models
/model chat gpt-5.2
/model chat o3  # For reasoning tasks

# Use latest Anthropic models
/model chat claude-sonnet-4-5-20250929
/model chat claude-opus-4-1-20250805

# Use Llama 4 locally via Ollama
/model chat ollama/llama4-scout
/model chat ollama/llama4-maverick

# Use DeepSeek for reasoning
/model chat ollama/deepseek-r1

# Access models via OpenRouter
/model chat openrouter/meta-llama/llama-4-scout:free
/model chat openrouter/deepseek/deepseek-r1
```

### Adding a New Model

Edit `~/.episodic/models.json` and add to the appropriate provider:

```json
{
  "providers": {
    "openai": {
      "models": [
        {
          "name": "gpt-5.2",
          "display_name": "GPT-5.2",
          "type": "chat",
          "context_window": 272000,
          "pricing": {
            "input": 1.25,
            "output": 10.0,
            "unit": "per_1m_tokens",
            "last_updated": "2025-11-17"
          }
        }
      ]
    }
  }
}
```

### Adding a New Provider

```json
{
  "providers": {
    "my-provider": {
      "display_name": "My Custom Provider",
      "api_base": "https://api.myprovider.com/v1",
      "models": [
        {
          "name": "my-model",
          "display_name": "My Custom Model",
          "type": "both",
          "parameters": "30B",
          "context_window": 32768
        }
      ]
    }
  }
}
```

### Overriding Existing Models

To change properties of existing models:

```json
{
  "providers": {
    "google": {
      "models": [
        {
          "name": "gemini-2.5-pro",
          "display_name": "Gemini 2.5 Pro (Custom)",
          "type": "both",
          "parameters": "500B",
          "context_window": 4194304
        }
      ]
    }
  }
}
```

### Adding Custom Type Patterns

```json
{
  "type_patterns": {
    "instruct": [
      "my-instruct-pattern",
      "custom-inst"
    ],
    "chat": [
      "my-chat-model",
      "conversational-*"
    ]
  }
}
```

## Model Properties

### Required Properties
- `name`: Model identifier used with the API
- `display_name`: Human-readable name shown in the UI
- `type`: Model type - one of: `chat`, `instruct`, `base`, `both`

### Optional Properties
- `parameters`: Model size (e.g., "7B", "175B+", "~1T")
- `context_window`: Maximum context length in tokens
- `detect_params`: For Ollama models, whether to detect parameters at runtime

## Model Types

- **[D] Detection**: Local models for topic boundary detection (e.g., fine-tuned DistilBERT)
- **[C] Chat**: Models optimized for conversation
- **[I] Instruct**: Models optimized for following instructions
- **[B] Base**: Base/completion models without special training
- **[CI] Both**: Models that work well for both chat and instructions

## Custom Local Models

Episodic supports custom local models for any purpose—topic detection, domain-specific chat, specialized summarization, etc. These models run locally without API calls.

### Wrapper Types

Episodic provides two wrappers for loading custom models:

| Wrapper | Format | Use Case |
|---------|--------|----------|
| `huggingface` | HuggingFace format (directory with config.json, model files) | **Recommended** - works with any fine-tuned transformer |
| `distilbert` | Raw PyTorch state_dict (.pt file) | Legacy - for specific DistilBERT checkpoints |

### Adding a HuggingFace Model (Recommended)

Most fine-tuned models are saved in HuggingFace format. Add them to `~/.episodic/models.json`:

```json
{
  "providers": {
    "custom": {
      "display_name": "Custom Models (local)",
      "local": true,
      "models": [
        {
          "name": "my-domain-classifier",
          "display_name": "Domain-Specific Classifier",
          "type": "detection",
          "path": "~/.episodic/models/my-fine-tuned-bert",
          "wrapper": "huggingface",
          "task": "sequence-classification",
          "parameters": "110M"
        },
        {
          "name": "my-chat-model",
          "display_name": "Custom Chat Model",
          "type": "chat",
          "path": "~/.episodic/models/my-fine-tuned-llama",
          "wrapper": "huggingface",
          "task": "text-generation",
          "parameters": "7B"
        }
      ]
    }
  }
}
```

### HuggingFace Model Properties

- `name`: Model identifier (used as `custom/<name>`)
- `display_name`: Human-readable name shown in the UI
- `type`: Model type (`detection`, `chat`, `instruct`, etc.)
- `path`: Path to HuggingFace model directory (contains `config.json`, model weights, tokenizer)
- `wrapper`: Must be `huggingface`
- `task`: HuggingFace task type:
  - `sequence-classification` - For classifiers (topic detection, sentiment, etc.)
  - `text-generation` - For causal LMs (chat, completion)
  - `summarization` - For seq2seq models (summarization, translation)
- `temperature`: Optional softmax temperature (default: 1.0)
- `max_length`: Optional max sequence length (default: 512)
- `parameters`: Model size for display

### Adding a Legacy DistilBERT Model

For raw PyTorch checkpoints (`.pt` files with state_dict):

```json
{
  "name": "topic-boundary-distilbert",
  "display_name": "Topic Boundary DistilBERT",
  "type": "detection",
  "path": "~/.episodic/models/final_calibrated.pt",
  "wrapper": "distilbert",
  "architecture": "distilbert-base-uncased",
  "parameters": "66M"
}
```

### Using Custom Models

```bash
# Set detection model to custom local model
/model detection custom/my-domain-classifier

# Set chat model to custom local model
/model chat custom/my-chat-model

# View current models
/model
```

### Model Storage

Store your models in `~/.episodic/models/`:
```
~/.episodic/
├── models/
│   ├── my-fine-tuned-bert/       # HuggingFace format directory
│   │   ├── config.json
│   │   ├── pytorch_model.bin     # or model.safetensors
│   │   ├── tokenizer.json
│   │   └── ...
│   ├── my-fine-tuned-llama/      # Another HuggingFace model
│   │   └── ...
│   └── legacy_model.pt           # Legacy .pt format
└── models.json                    # Model configuration
```

### Saving Models in HuggingFace Format

If you fine-tune a model, save it in HuggingFace format:

```python
# After training
model.save_pretrained("~/.episodic/models/my-model")
tokenizer.save_pretrained("~/.episodic/models/my-model")
```

This creates the directory structure Episodic expects.

## Example: Complete User Configuration

```json
{
  "_comment": "User model overrides and additions",
  
  "providers": {
    "openai": {
      "models": [
        {
          "name": "gpt-4-vision",
          "display_name": "GPT-4 Vision",
          "type": "chat",
          "parameters": "175B+",
          "context_window": 128000,
          "capabilities": ["vision"]
        }
      ]
    },
    
    "local-llm": {
      "display_name": "Local LLM Server",
      "api_base": "http://192.168.1.100:8080/v1",
      "models": [
        {
          "name": "mixtral-moe",
          "display_name": "Mixtral MoE Local",
          "type": "both",
          "parameters": "8x7B"
        },
        {
          "name": "codellama-70b",
          "display_name": "CodeLlama 70B Local",
          "type": "instruct",
          "parameters": "70B"
        }
      ]
    }
  },
  
  "type_patterns": {
    "instruct": [
      "instruction-tuned",
      "-it$"
    ]
  }
}
```

## Reloading Configuration

The model configuration is loaded when Episodic starts. To reload after making changes:

1. Restart Episodic, or
2. The configuration will be reloaded when accessing model lists

## API Key Configuration

Remember to set the appropriate environment variables for your providers:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- etc.

Or configure them in your `~/.episodic/config.json` file.