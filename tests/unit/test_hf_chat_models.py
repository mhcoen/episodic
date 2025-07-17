#!/usr/bin/env python3
"""Test which HuggingFace models work with chat format."""

import subprocess
import sys
import tempfile
import os

# Test different HF models to see which ones work
models_to_test = [
    "huggingface/tiiuae/falcon-7b-instruct",
    "huggingface/tiiuae/falcon-40b-instruct",
    "huggingface/tiiuae/falcon-180B-chat",
    "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    "huggingface/mistralai/Mistral-7B-Instruct-v0.2",
    "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "huggingface/google/flan-t5-xxl",
    "huggingface/bigscience/bloom",
    "huggingface/EleutherAI/gpt-neox-20b",
    "huggingface/stabilityai/stablelm-tuned-alpha-7b"
]

print("Testing HuggingFace Chat Models")
print("=" * 60)

for model in models_to_test:
    model_name = model.split("/")[-1]
    print(f"\nTesting: {model_name}...", end=" ", flush=True)
    
    test_script = f"""
/init --erase
/model chat {model}
Say 'Hello' in 5 words or less
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
        
        if "not a chat model" in result.stdout or "not a chat model" in result.stderr:
            print("❌ Not a chat model")
        elif "model_not_supported" in result.stdout or "model_not_supported" in result.stderr:
            print("❌ Model not supported")
        elif "does not exist" in result.stdout or "does not exist" in result.stderr:
            print("❌ Model does not exist")
        elif "Hello" in result.stdout or "Hi" in result.stdout:
            print("✅ Works!")
        elif "[LLM response skipped]" in result.stdout:
            print("⚠️  Response skipped")
        else:
            # Check for other errors
            if "error" in result.stderr.lower():
                error_line = [line for line in result.stderr.split('\n') if 'error' in line.lower()]
                if error_line:
                    print(f"❌ Error: {error_line[0][:50]}...")
                else:
                    print("❌ Unknown error")
            else:
                print("⚠️  Unclear result")
                
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout")
    finally:
        os.unlink(script_path)

print("\n" + "=" * 60)
print("Note: Some models may require different formatting or API endpoints.")