#!/usr/bin/env python3
"""Debug command handling."""

import sys
sys.path.insert(0, '.')

from episodic.cli import handle_command

# Test command handling
test_commands = ["/topics", "/list", "/help"]

for cmd in test_commands:
    print(f"\nTesting command: {cmd}")
    print("-" * 40)
    try:
        result = handle_command(cmd)
        print(f"Command returned: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()