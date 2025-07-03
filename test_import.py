#!/usr/bin/env python3
"""Test imports one by one"""

import sys
print("1. Starting test...", file=sys.stderr)

try:
    print("2. Importing episodic...", file=sys.stderr)
    import episodic
    print("3. Imported episodic", file=sys.stderr)
    
    print("4. Importing episodic.cli...", file=sys.stderr)
    import episodic.cli
    print("5. Imported episodic.cli", file=sys.stderr)
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()

print("6. Done", file=sys.stderr)