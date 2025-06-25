#!/usr/bin/env python
"""Test to find source of NoneType: None output."""

import subprocess
import sys

# Run the CLI and capture stderr separately
proc = subprocess.Popen(
    [sys.executable, "-m", "episodic"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send summary command then exit
stdout, stderr = proc.communicate(input="/summary 5\n/exit\n")

print("=== STDOUT ===")
print(stdout[:500])  # First 500 chars

print("\n=== STDERR ===")
print(stderr)

# Check if NoneType appears in stderr
if "NoneType" in stderr:
    print("\nFound 'NoneType' in stderr!")
    # Print lines containing NoneType
    for line in stderr.split('\n'):
        if "NoneType" in line:
            print(f"  Line: {line}")