#!/usr/bin/env python3
"""
Test script to reproduce the streaming duplication issue.
Specifically for headers surrounded by **'s
"""

import subprocess
import sys
import time

def test_streaming_with_headers():
    """Test streaming with markdown headers."""
    print("Testing streaming duplication with markdown headers...\n")
    
    # Start episodic in a subprocess
    cmd = [sys.executable, "-m", "episodic"]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # Unbuffered for real-time output
    )
    
    # Commands to test
    commands = [
        "/set stream on",
        "/set debug on",
        "What are the top 5 sci-fi movies of all time? Please format your response with **bold headers**.",
        "/exit"
    ]
    
    # Send commands one by one
    for cmd in commands:
        print(f"Sending: {cmd}")
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        time.sleep(2)  # Give time for response
    
    # Get output
    output, error = proc.communicate()
    
    print("\n=== OUTPUT ===")
    print(output)
    
    if error:
        print("\n=== ERROR ===")
        print(error)

if __name__ == "__main__":
    test_streaming_with_headers()