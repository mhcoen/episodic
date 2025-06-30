#!/usr/bin/env python3
"""Test CLI topics command handling."""

import sys
sys.path.insert(0, '.')

from episodic.cli import handle_command
from episodic.config import config

print("Testing /topics command through CLI handler...")
print("=" * 60)

# Check config
print(f"Benchmark enabled: {config.get('benchmark', False)}")
print(f"Benchmark display: {config.get('benchmark_display', True)}")

# Try the command
print("\nExecuting: /topics")
result = handle_command("/topics")
print(f"\nCommand returned: {result}")

# Try with debug
print("\n\nTrying with debug enabled:")
config.set("debug", True)
result = handle_command("/topics")
print(f"\nCommand returned: {result}")