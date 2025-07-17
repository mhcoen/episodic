#!/usr/bin/env python3
"""Test HuggingFace API key and model access."""

import subprocess
import sys
import tempfile
import os

test_script = """
/init --erase
/model chat huggingface/tiiuae/falcon-7b-instruct
Tell me a fun fact in 10 words or less
/exit
"""

print("Testing HuggingFace API integration...")
print("-" * 50)

# Check if API key is set
api_key = os.environ.get("HUGGINGFACE_API_KEY")
if api_key:
    print(f"✅ HUGGINGFACE_API_KEY is set (length: {len(api_key)})")
else:
    print("❌ HUGGINGFACE_API_KEY not found in environment")

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(test_script)
    script_path = f.name

try:
    result = subprocess.run(
        [sys.executable, "-m", "episodic", "--execute", script_path],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy()  # Pass environment variables
    )
    
    if "Invalid username or password" in result.stdout or "Invalid username or password" in result.stderr:
        print("\n❌ Authentication failed - API key may be invalid")
    elif "Fun fact:" in result.stdout or "fact" in result.stdout.lower():
        print("\n✅ HuggingFace model responded successfully!")
    elif "[LLM response skipped]" in result.stdout:
        print("\n⚠️  Response was skipped")
    else:
        print("\n⚠️  Unexpected result")
    
    # Check for the async warning
    if "close_litellm_async_clients" in result.stderr:
        print("\n📌 LiteLLM async warning present (this is harmless)")
    
    # Save output
    with open("huggingface_test_output.txt", "w") as f:
        f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
    print("\nFull output saved to: huggingface_test_output.txt")
        
finally:
    os.unlink(script_path)