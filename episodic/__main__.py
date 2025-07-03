
"""
Entry point for the Episodic CLI.
"""

import os
# Disable ChromaDB telemetry before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"

if __name__ == "__main__":
    from episodic.cli import app
    app()
