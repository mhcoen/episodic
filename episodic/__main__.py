
"""
Entry point for the Episodic CLI.
"""

import os
import sys

# Clear screen immediately before any imports
if os.name == 'nt':  # Windows
    os.system('cls')
else:  # Unix/Linux/MacOS
    os.system('clear')

# Disable ChromaDB telemetry before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"

if __name__ == "__main__":
    from episodic.cli import app
    app()
