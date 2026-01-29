"""
UI components for Episodic CLI.

This package contains UI formatting and input handling for
interactive prompts like topic disambiguation.
"""

from .disambiguation import (
    DisambiguationResult,
    format_disambiguation_options,
    handle_disambiguation_input,
)

__all__ = [
    "DisambiguationResult",
    "format_disambiguation_options",
    "handle_disambiguation_input",
]
