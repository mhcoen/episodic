#!/usr/bin/env python3
"""Test new HuggingFace models."""

import subprocess
import sys
import tempfile
import os

# Test a few of the new models
models_to_test = [
    ("huggingface/Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B"),
    ("huggingface/meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B"),
    ("huggingface/mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B v0.3"),
    ("huggingface/tiiuae/Falcon3-7B-Instruct", "Falcon 3 7B"),
    ("huggingface/deepseek-ai/deepseek-llm-7b-chat", "DeepSeek 7B"),
    ("huggingface/01-ai/Yi-1.5-6B-Chat", "Yi 1.5 6B"),
    ("huggingface/GeneZC/MiniChat-2-3B", "MiniChat 2 3B")
]

print("Testing New HuggingFace Models")
print("=" * 60)

for model, display_name in models_to_test:
    print(f"\nTesting: {display_name}...", end=" ", flush=True)
    
    test_script = f"""
/init --erase
/model chat {model}
Say hello in 5 words
/exit
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "episodic", "--execute", script_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy()
        )
        
        # Check various error patterns
        if "not a chat model" in result.stdout or "not a chat model" in result.stderr:
            print("❌ Not a chat model")
        elif "model_not_supported" in result.stdout or "model_not_supported" in result.stderr:
            print("❌ Model not supported")
        elif "does not exist" in result.stdout or "does not exist" in result.stderr:
            print("❌ Model does not exist")
        elif "rate limit" in result.stdout.lower() or "rate limit" in result.stderr.lower():
            print("⏱️  Rate limited")
        elif "Hello" in result.stdout or "Hi" in result.stdout or "Greetings" in result.stdout:
            print("✅ Works!")
        elif "error" in result.stderr.lower():
            # Extract specific error
            for line in result.stderr.split('\n'):
                if "error" in line.lower() and ("message" in line or "Error:" in line):
                    print(f"❌ {line.strip()[:60]}...")
                    break
            else:
                print("❌ Unknown error")
        else:
            print("⚠️  Unclear result")
                
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout")
    finally:
        os.unlink(script_path)

print("\n" + "=" * 60)
print("Note: Some models may require specific API endpoints or configurations.")