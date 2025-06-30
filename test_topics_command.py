#!/usr/bin/env python3
"""Test topics command directly."""

import sys
sys.path.insert(0, '.')

from episodic.commands.topics import topics

print("Testing /topics command directly...")
print("=" * 60)

# Call the topics function directly
topics(limit=10, all=False, verbose=False)

print("\nDirect call completed.")