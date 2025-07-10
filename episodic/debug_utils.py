"""
Debug utilities for Episodic.

Provides common debugging functionality used across modules.
"""

import typer
from episodic.config import config
from episodic.color_utils import secho_color


def debug_print(message: str, indent: bool = False, style: str = "default") -> None:
    """
    Print debug message with consistent formatting.
    
    Args:
        message: The debug message to print
        indent: Whether to indent the message
        style: Style of debug output ('default', 'fancy', or 'minimal')
    """
    if not config.get("debug", False):
        return
        
    if style == "fancy":
        # Fancy style with emoji and colors
        if indent:
            secho_color(f"   {message}", fg='bright_cyan')
        else:
            secho_color(f"🔍 DEBUG: {message}", fg='yellow', bold=True)
    else:
        # Default/minimal style
        prefix = "  " if indent else ""
        typer.secho(f"{prefix}[DEBUG] {message}", fg="yellow", err=True)