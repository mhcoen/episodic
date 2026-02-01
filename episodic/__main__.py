"""
Entry point for the Episodic CLI.
"""

import os
import sys
import logging
import warnings

# Environment settings before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress noisy transformer/safetensors model loading output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*not sharded.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*LOAD REPORT.*")


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
