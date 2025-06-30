#!/usr/bin/env python3
"""Test interactive mode with explicit flushing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Import after setting unbuffered
from episodic.cli import app
import typer

# Run with explicit flush
if __name__ == "__main__":
    print("Starting Episodic in test mode...", flush=True)
    print("Try typing /topics when prompted", flush=True)
    app()