"""
Entry point for the Episodic CLI.
"""

import os
import sys

# Environment settings before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"


def main():
    """Main entry point for the CLI."""
    # Clear screen
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/MacOS
        os.system('clear')

    from episodic.cli import app
    app()


if __name__ == "__main__":
    main()
