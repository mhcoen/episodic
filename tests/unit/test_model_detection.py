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
    
    # Google
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-ultra",
    "google/gemini-1.5-pro",
]


def test_detect_model_type_returns_known_type():
    """Model type detection returns a supported type for known models."""
    allowed_types = {"chat", "instruct", "base", "unknown", "both"}
    for model in test_models:
        model_type = detect_model_type(model)
        assert model_type in allowed_types
