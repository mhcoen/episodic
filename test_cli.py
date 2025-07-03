#!/usr/bin/env python3
"""Test CLI startup"""

import sys
print("Starting test...", file=sys.stderr)

try:
    from episodic.cli import main
    print("Import successful", file=sys.stderr)
    
    # Try to run main with --help
    sys.argv = ['episodic', '--help']
    print("Calling main()...", file=sys.stderr)
    main()
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()