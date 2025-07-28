#!/usr/bin/env python3
"""Test flat window approach - just concatenate all text."""

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

print("Testing different prompt approaches with qwen2:0.5b")
print("="*60)

# Test 1: Structured window approach (current)
window_size = 3
current_idx = 6
window_a = messages[max(0, current_idx - window_size):current_idx]
window_b = messages[current_idx:min(len(messages), current_idx + window_size)]

window_a_str = "\n".join([msg['content'] for msg in window_a])
window_b_str = "\n".join([msg['content'] for msg in window_b])

structured_prompt = f"""Compare topics between two windows of messages.

Window A (before):
{window_a_str}

Window B (after):
{window_b_str}

Rate topic change from 0.0 to 1.0.
Output ONLY a number like 0.7
Nothing else. Just the number.
"""

print("1. STRUCTURED APPROACH (current):")
print(structured_prompt)
print("-"*40)

result = subprocess.run(['ollama', 'run', 'qwen2:0.5b', structured_prompt], 
                       capture_output=True, text=True, timeout=15)
print("Response:", repr(result.stdout))

# Test 2: Flat concatenation approach
print("\n" + "="*60)
all_text = " ".join([msg['content'] for msg in messages[max(0, current_idx - window_size):min(len(messages), current_idx + window_size)]])

flat_prompt = f"""Rate topic change in this conversation from 0.0 to 1.0:

{all_text}

Output only a number.
"""

print("2. FLAT CONCATENATION APPROACH:")
print(flat_prompt)
print("-"*40)

result = subprocess.run(['ollama', 'run', 'qwen2:0.5b', flat_prompt], 
                       capture_output=True, text=True, timeout=15)
print("Response:", repr(result.stdout))

# Test 3: Even simpler - just the boundary
print("\n" + "="*60)
before = messages[current_idx - 1]['content']
after = messages[current_idx]['content']

boundary_prompt = f"""Rate topic change from 0 to 1:
"{before}" to "{after}"
Number only:"""

print("3. SIMPLE BOUNDARY APPROACH:")
print(boundary_prompt)
print("-"*40)

result = subprocess.run(['ollama', 'run', 'qwen2:0.5b', boundary_prompt], 
                       capture_output=True, text=True, timeout=15)
print("Response:", repr(result.stdout))

# Test 4: Keywords approach
print("\n" + "="*60)
# Extract key words from each window
import re
def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    # Filter common words
    common = {'the', 'a', 'an', 'is', 'it', 'for', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'with', 'yes', 'no', 'i', 'you', 'we', 'they', 'am', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'how', 'help', 'me', 'my', 'your', 'our', 'their'}
    return [w for w in words if w not in common and len(w) > 2]

keywords_a = extract_keywords(window_a_str)[:5]
keywords_b = extract_keywords(window_b_str)[:5]

keyword_prompt = f"""Rate topic change 0-1:
Before: {', '.join(keywords_a)}
After: {', '.join(keywords_b)}
Number:"""

print("4. KEYWORDS APPROACH:")
print(keyword_prompt)
print("-"*40)

result = subprocess.run(['ollama', 'run', 'qwen2:0.5b', keyword_prompt], 
                       capture_output=True, text=True, timeout=15)
print("Response:", repr(result.stdout))

print("\n" + "="*60)
print("SUMMARY:")
print("Small models need extremely simple prompts!")
print("The flatter and simpler, the better.")