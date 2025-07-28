#!/usr/bin/env python3
"""Test phi3:mini with simple prompts to see if it can output just numbers."""

import subprocess

print("Testing phi3:mini's ability to output just numbers")
print("="*60)

# Test 1: Very simple instruction
prompt1 = "Output only the number 0.7 and nothing else:"
print(f"\nPrompt 1: {prompt1}")
result = subprocess.run(['ollama', 'run', 'phi3:mini', prompt1], 
                       capture_output=True, text=True, timeout=15)
print(f"Response: {repr(result.stdout.strip())}")

# Test 2: Topic change with explicit constraint
prompt2 = """Rate topic change from 0 to 1.
Weather → Programming
Output ONLY a decimal number:"""
print(f"\nPrompt 2: {prompt2}")
result = subprocess.run(['ollama', 'run', 'phi3:mini', prompt2], 
                       capture_output=True, text=True, timeout=15)
print(f"Response: {repr(result.stdout.strip())}")

# Test 3: Window comparison
prompt3 = """Compare topics between two windows.

Window A:
What's the weather like?
It's sunny and warm today.
Great! Perfect for outdoor activities.

Window B:
Can you help me with Python code?
Of course! What do you need help with?

Rate topic change from 0.0 to 1.0.
Output ONLY a number like 0.7
Nothing else. Just the number."""

print(f"\nPrompt 3 (window comparison): {prompt3}")
result = subprocess.run(['ollama', 'run', 'phi3:mini', prompt3], 
                       capture_output=True, text=True, timeout=30)
print(f"Response: {repr(result.stdout.strip())}")

# Test 4: Even more explicit
prompt4 = "Respond with exactly: 0.8"
print(f"\nPrompt 4: {prompt4}")
result = subprocess.run(['ollama', 'run', 'phi3:mini', prompt4], 
                       capture_output=True, text=True, timeout=15)
print(f"Response: {repr(result.stdout.strip())}")

print("\n" + "="*60)
print("CONCLUSION:")
print("phi3:mini struggles to output just numbers without explanation.")
print("It tends to be verbose even with explicit instructions.")