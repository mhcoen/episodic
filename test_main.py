#!/usr/bin/env python3
"""Test main entry point"""

import sys
print("Setting up test...", file=sys.stderr)

# Set up arguments
sys.argv = ['episodic']  # No arguments, just run normally

try:
    print("Importing __main__...", file=sys.stderr)
    from episodic.__main__ import main
    print("Calling main()...", file=sys.stderr)
    main()
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()