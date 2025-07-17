#!/usr/bin/env python3
"""Test the improved model type detection."""

from episodic.model_utils import detect_model_type

# Test cases
test_models = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-o3",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct",
    "gpt-4",
    
    # Anthropic
    "claude-opus-4-20250514",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    
    # Ollama/Local
    "ollama/llama3:latest",
    "ollama/llama3:instruct",
    "ollama/mistral:instruct",
    "ollama/phi3",
    "ollama/gemma:2b-instruct",
    "ollama/qwen2:1.5b",
    "ollama/deepseek-r1:8b",
    "ollama/codellama:13b",
    
    # HuggingFace
    "huggingface/Qwen/Qwen2.5-72B-Instruct",
    "huggingface/meta-llama/Llama-3.3-70B-Instruct",
    "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
    "huggingface/tiiuae/Falcon3-10B-Instruct",
    "huggingface/deepseek-ai/DeepSeek-V3",
    "huggingface/01-ai/Yi-1.5-34B-Chat",
    "huggingface/GeneZC/MiniChat-2-3B",
    "huggingface/bigscience/bloom",
    "huggingface/EleutherAI/gpt-neox-20b",
    "huggingface/stabilityai/stablelm-tuned-alpha-7b",
    
    # Groq
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/mixtral-8x7b-32768",
    "groq/gemma-7b-it",
]

# Test and display results
print("Model Type Detection Test")
print("=" * 80)
print(f"{'Model Name':<55} {'Detected Type':<15}")
print("-" * 80)

type_counts = {'chat': 0, 'instruct': 0, 'base': 0, 'unknown': 0}

for model in test_models:
    model_type = detect_model_type(model)
    type_counts[model_type] += 1
    
    # Color code output
    if model_type == 'instruct':
        color = '\033[92m'  # Green
    elif model_type == 'chat':
        color = '\033[94m'  # Blue
    elif model_type == 'base':
        color = '\033[95m'  # Magenta
    else:
        color = '\033[91m'  # Red for unknown
    
    print(f"{model:<55} {color}{model_type:<15}\033[0m")

print("\n" + "=" * 80)
print("Summary:")
print(f"  Chat models:     {type_counts['chat']}")
print(f"  Instruct models: {type_counts['instruct']}")
print(f"  Base models:     {type_counts['base']}")
print(f"  Unknown:         {type_counts['unknown']}")
print(f"\nTotal models tested: {len(test_models)}")