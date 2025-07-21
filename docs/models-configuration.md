# Model Configuration Guide

## Overview

Episodic uses a JSON-based model configuration system that allows you to:
- View all available models with their types and parameters
- Add custom models
- Override model properties
- Configure provider-specific settings

## Configuration Files

### Default Models: `episodic/models.json`
This file contains the default model definitions shipped with Episodic. It includes:
- Provider configurations
- Model definitions with types, parameters, and context windows
- Type detection patterns
- Type indicators for display

### User Models: `~/.episodic/models.json`
You can create this file to:
- Add new providers
- Add models to existing providers
- Override model properties
- Customize type detection patterns

## User Model Configuration

### Adding a New Model

Create `~/.episodic/models.json`:

```json
{
  "providers": {
    "openai": {
      "models": [
        {
          "name": "gpt-4-turbo-preview",
          "display_name": "GPT-4 Turbo Preview",
          "type": "chat",
          "parameters": "175B+",
          "context_window": 128000
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

- **[C] Chat**: Models optimized for conversation
- **[I] Instruct**: Models optimized for following instructions
- **[B] Base**: Base/completion models without special training
- **[CI] Both**: Models that work well for both chat and instructions

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