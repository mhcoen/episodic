#!/usr/bin/env python3
"""Test TinyLlama with different window=3 approaches."""

import subprocess
import sys
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

print("Testing TinyLlama with different approaches")
print("="*60)

# First, test if TinyLlama can follow simple instructions
print("1. BASIC INSTRUCTION TEST:")
test_prompt = "Output only the number 0.7 and nothing else:"
print(f"Prompt: {test_prompt}")
result = subprocess.run(['ollama', 'run', 'tinyllama:latest', test_prompt], 
                       capture_output=True, text=True, timeout=15)
print(f"Response: {repr(result.stdout)}")

# Test window approach
print("\n" + "="*60)
window_size = 3
current_idx = 6
window_a = messages[max(0, current_idx - window_size):current_idx]
window_b = messages[current_idx:min(len(messages), current_idx + window_size)]

# Try different prompt styles
prompts = [
    # Approach 1: Very simple
    f"""Topic change score (0-1):
Before: weather, sunny, outdoor, hiking
After: Python, code, programming
Score: """,
    
    # Approach 2: Direct instruction
    f"""Rate 0 to 1. Weather talk changes to programming talk.
Output number: """,
    
    # Approach 3: Examples
    f"""Examples:
Same topic = 0.0
Small change = 0.3
Big change = 0.8

Weather to programming = """,
    
    # Approach 4: Binary then scale
    f"""Topics change from outdoor/weather to programming/code.
Different topics? (0=same, 1=different): """
]

for i, prompt in enumerate(prompts, 2):
    print(f"\n{i}. APPROACH {i}:")
    print(f"Prompt: {prompt}")
    print("-"*40)
    
    try:
        result = subprocess.run(['ollama', 'run', 'tinyllama:latest', prompt], 
                               capture_output=True, text=True, timeout=15)
        response = result.stdout.strip()
        print(f"Response: {repr(response)}")
        
        # Try to extract number
        import re
        numbers = re.findall(r'\d*\.?\d+', response)
        if numbers:
            print(f"Numbers found: {numbers}")
            print(f"First number: {float(numbers[0])}")
    except Exception as e:
        print(f"Error: {e}")

# Also test with the gemma:2b-instruct we have
print("\n\n" + "="*60)
print("TESTING GEMMA:2B-INSTRUCT")
print("="*60)

simple_prompt = """Rate topic change 0 to 1:
weather outdoor → python code
Number: """

print(f"Prompt: {simple_prompt}")
result = subprocess.run(['ollama', 'run', 'gemma:2b-instruct', simple_prompt], 
                       capture_output=True, text=True, timeout=15)
print(f"Response: {repr(result.stdout)}")