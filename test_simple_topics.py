#!/usr/bin/env python3
"""Simple test to verify topic extraction."""

import subprocess
import sys

# Create a simple test script
test_script = """
/init --erase
/set main.max_tokens 30
/set main.temperature 0

# Topic 1: Space
Tell me about Mars rovers in one sentence.
What year did Curiosity land on Mars?
How many rovers are on Mars currently?
What powers the Mars rovers?
Can rovers communicate with Earth directly?

# Topic 2: Cooking  
I want to learn about French cooking basics.
What are the five mother sauces?
Which sauce is made with butter and flour?
How do you make a roux?
What's the difference between blonde and brown roux?

/topics
"""

# Write test script
with open('test_simple.txt', 'w') as f:
    f.write(test_script)

# Run it
result = subprocess.run([
    sys.executable, '-m', 'episodic'
], input="/script test_simple.txt\n/exit\n", text=True, capture_output=True)

# Check topics
print("=== STDERR (errors) ===")
print(result.stderr[-500:] if result.stderr else "No errors")

print("\n=== Topics Output ===")
lines = result.stdout.split('\n')
in_topics = False
for line in lines:
    if "Conversation Topics" in line:
        in_topics = True
    if in_topics:
        print(line)
        if "======" in line and in_topics and "Conversation Topics" not in line:
            break