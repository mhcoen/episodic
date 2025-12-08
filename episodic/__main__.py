"""
Entry point for the Episodic CLI.
"""

import os
import sys

# Environment settings before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Tell huggingface_hub to disable progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


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
