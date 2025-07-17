#!/usr/bin/env python3
"""Quick test to verify the topic detection fix."""

import subprocess
import sys
import tempfile
import os

test_script = """
/init --erase
/set automatic_topic_detection true
/set min_messages_before_topic_change 4
/set show_topics true
/model chat gpt-3.5-turbo
/model detection gpt-3.5-turbo

# Start conversation
Hello, tell me about Mars
What's the atmosphere like?
How long to get there?
Is it cold?

# Topic change
How to make pasta carbonara?
What ingredients?

/topics list
/exit
"""

print("Testing topic detection fix...")
print("-" * 50)

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(test_script)
    script_path = f.name

try:
    result = subprocess.run(
        [sys.executable, "-m", "episodic", "--execute", script_path],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Save full output for debugging
    with open("topic_test_output.txt", "w") as f:
        f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
    
    if "Error: cannot access local variable 'get_ancestry'" in result.stdout:
        print("❌ FAILED: Topic detection error still present")
    elif "Topic change detected" in result.stdout:
        print("✅ PASSED: Topic detection working!")
    else:
        print("⚠️  WARNING: No topic change detected")
        
    # Also check stderr
    if "Error: cannot access local variable 'get_ancestry'" in result.stderr:
        print("❌ FAILED: Topic detection error in stderr")
    
    # Check for topics list
    if "Conversation Topics" in result.stdout:
        print("✅ PASSED: Topics list displayed")
        # Count topics
        topic_count = result.stdout.count("Range:")
        print(f"   Found {topic_count} topics")
    else:
        print("❌ FAILED: Topics list not displayed")
    
    if result.stderr:
        print("\nErrors:")
        print(result.stderr[:500])
        
finally:
    os.unlink(script_path)
    
print("\nDone!")