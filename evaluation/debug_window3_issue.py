#!/usr/bin/env python3
"""Debug why window=3 is causing issues."""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test messages
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thanks!"},
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "It's sunny and warm today."},
    {"role": "user", "content": "Great! Perfect for outdoor activities."},
    {"role": "assistant", "content": "Yes, it's ideal for hiking or picnics."},
    # Topic change
    {"role": "user", "content": "Can you help me with Python code?"},
    {"role": "assistant", "content": "Of course! What do you need help with?"},
]

# Test window comparison at position 6 (topic change)
window_size = 3
current_idx = 6

# Calculate windows
window_a_start = max(0, current_idx - window_size)
window_a_end = current_idx
window_b_start = current_idx
window_b_end = min(len(messages), current_idx + window_size)

window_a = messages[window_a_start:window_a_end]
window_b = messages[window_b_start:window_b_end]

# Create prompt
window_a_str = "\n".join([msg['content'][:100] for msg in window_a])
window_b_str = "\n".join([msg['content'][:100] for msg in window_b])

prompt = f"""Compare topics between two windows of messages.

Window A (before):
{window_a_str}

Window B (after):
{window_b_str}

Rate topic change from 0.0 to 1.0.
Output ONLY a number like 0.7
Nothing else. Just the number.
"""

print("Testing window-based prompt with qwen2:0.5b")
print("="*60)
print("Prompt:")
print(prompt)
print("="*60)

# Test with qwen2:0.5b
try:
    result = subprocess.run(
        ['ollama', 'run', 'qwen2:0.5b', prompt],
        capture_output=True,
        text=True,
        timeout=15
    )
    
    print("Response:", repr(result.stdout))
    print("Length:", len(result.stdout))
    
    # Try to extract number
    import re
    numbers = re.findall(r'\d*\.?\d+', result.stdout)
    print("Numbers found:", numbers)
    
except Exception as e:
    print(f"Error: {e}")

# Now test with simpler prompt
print("\n\n" + "="*60)
print("Testing with SIMPLER prompt")
print("="*60)

simple_prompt = """Rate from 0 to 1:
Before: weather, sunny, outdoor activities
After: Python code, programming help

Number only:"""

print("Prompt:")
print(simple_prompt)
print("="*60)

try:
    result = subprocess.run(
        ['ollama', 'run', 'qwen2:0.5b', simple_prompt],
        capture_output=True,
        text=True,
        timeout=15
    )
    
    print("Response:", repr(result.stdout))
    print("Length:", len(result.stdout))
    
except Exception as e:
    print(f"Error: {e}")