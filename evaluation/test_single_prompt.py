#!/usr/bin/env python3
"""Test single prompt with qwen2:0.5b"""

import subprocess

prompt = """Rate topic change from 0.0 to 1.0.

Message 1: How's the weather today?
Message 2: It's sunny and warm.

Output ONLY a number like 0.7
Nothing else. Just the number.
"""

print("Testing qwen2:0.5b with direct prompt...")
print("Prompt:", prompt)
print("-"*50)

result = subprocess.run(
    ['ollama', 'run', 'qwen2:0.5b', prompt],
    capture_output=True,
    text=True,
    timeout=10
)

print("Response:", repr(result.stdout))
print("Error:", repr(result.stderr))