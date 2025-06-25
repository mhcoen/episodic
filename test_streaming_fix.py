#!/usr/bin/env python3
"""
Test script to verify streaming fixes.

This tests:
1. Double printing is fixed
2. Constant-rate streaming works
3. Configuration via /set commands
"""

import subprocess
import time
import sys

def run_episodic_command(command):
    """Run an episodic command and return the output."""
    cmd = [sys.executable, "-m", "episodic"]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, error = proc.communicate(input=command)
    return output, error

def test_streaming():
    """Test the streaming functionality."""
    print("Testing streaming fixes...\n")
    
    # Test 1: Check configuration display
    print("1. Testing /set command to view current settings:")
    output, _ = run_episodic_command("/set\n/exit\n")
    print("Output snippet:")
    for line in output.split('\n'):
        if 'stream' in line.lower():
            print(f"  {line}")
    print()
    
    # Test 2: Enable constant-rate streaming
    print("2. Testing stream_constant_rate toggle:")
    commands = """
/set stream_constant_rate on
/set stream_rate 10
/set
/exit
"""
    output, _ = run_episodic_command(commands)
    print("Output snippet:")
    for line in output.split('\n'):
        if 'constant-rate streaming' in line.lower() or 'stream_rate' in line.lower():
            print(f"  {line}")
    print()
    
    # Test 3: Simple chat test (manual verification needed)
    print("3. To manually test streaming:")
    print("   Run: python -m episodic")
    print("   Then:")
    print("   /set stream_constant_rate on")
    print("   /set stream_rate 5")
    print("   Type a message and observe the response streaming")
    print("   - Should stream at 5 words per second")
    print("   - Should NOT print text twice")
    print()
    print("   Compare with:")
    print("   /set stream_constant_rate off")
    print("   Type a message")
    print("   - Should stream immediately as chunks arrive")
    print("   - Should NOT print text twice")

if __name__ == "__main__":
    test_streaming()