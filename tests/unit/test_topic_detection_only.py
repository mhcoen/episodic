#!/usr/bin/env python3
"""Test topic detection without actual LLM responses."""

import subprocess
import sys
import tempfile
import os

test_script = """
/init --erase
/set automatic_topic_detection true
/set min_messages_before_topic_change 4
/set show_topics true
/set skip_llm_response true
/model chat gpt-3.5-turbo
/model detection gpt-3.5-turbo

# Start conversation
Hello, tell me about Mars
What's the atmosphere like?
How long to get there?
Is it cold?

# Topic change - should be detected here
How to make pasta carbonara?
What ingredients?
What cheese to use?
How long to cook?

# Another topic change
What is machine learning?
How do neural networks work?
What is backpropagation?
What is deep learning?

/topics list
/exit
"""

print("Testing topic detection (without LLM responses)...")
print("-" * 50)

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(test_script)
    script_path = f.name

try:
    result = subprocess.run(
        [sys.executable, "-m", "episodic", "--execute", script_path],
        capture_output=True,
        text=True,
        timeout=30  # Should be fast without LLM calls
    )
    
    # Save full output for debugging
    with open("topic_detection_output.txt", "w") as f:
        f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
    
    # Check for the specific error
    if "Error: cannot access local variable 'get_ancestry'" in result.stdout or \
       "Error: cannot access local variable 'get_ancestry'" in result.stderr:
        print("❌ FAILED: Topic detection error still present!")
        print("   The get_ancestry import fix did not work")
    else:
        print("✅ PASSED: No get_ancestry error!")
    
    # Check if topic changes were detected
    topic_changes = result.stdout.count("Topic change detected")
    print(f"\n📌 Topic changes detected: {topic_changes}")
    if topic_changes >= 2:
        print("✅ PASSED: Multiple topic changes detected")
    elif topic_changes == 1:
        print("⚠️  WARNING: Only one topic change detected")
    else:
        print("❌ FAILED: No topic changes detected")
    
    # Check for topics list
    if "Conversation Topics" in result.stdout:
        print("\n✅ PASSED: Topics list displayed")
        # Count topics
        topic_count = result.stdout.count("Range:")
        print(f"   Found {topic_count} topics")
        
        # Look for specific topic names
        if "ongoing-" in result.stdout:
            ongoing_count = result.stdout.count("ongoing-")
            print(f"   {ongoing_count} topics still have 'ongoing-' names")
    else:
        print("\n❌ FAILED: Topics list not displayed")
    
    # Performance check
    if result.returncode == 0:
        print("\n✅ PASSED: Script completed successfully")
    else:
        print(f"\n❌ FAILED: Script failed with return code {result.returncode}")
        
    if result.stderr and len(result.stderr) > 10:
        print("\n⚠️  Errors detected:")
        print(result.stderr[:500])
        
finally:
    os.unlink(script_path)
    
print("\nFull output saved to: topic_detection_output.txt")
print("Done!")