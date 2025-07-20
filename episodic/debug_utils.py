"""
Debug utilities for Episodic.

Provides common debugging functionality used across modules.
Now integrated with the named debug system for category-based filtering.
"""

import typer
from typing import Optional
from episodic.config import config
from episodic.color_utils import secho_color


def debug_print(
    message: str, 
    indent: bool = False, 
    style: str = "default",
    category: Optional[str] = None
) -> None:
    """
    Print debug message with consistent formatting.
    
    Args:
        message: The debug message to print
        indent: Whether to indent the message
        style: Style of debug output ('default', 'fancy', or 'minimal')
        category: Debug category (e.g., 'memory', 'topic', 'drift')
    """
    # Use new debug system if available
    try:
        from episodic.debug_system import debug_enabled
        
        # If no category specified, check legacy debug flag
        if category is None:
            if not config.get("debug", False):
                return
        else:
            if not debug_enabled(category):
                return
    except ImportError:
        # Fallback to legacy debug flag
        if not config.get("debug", False):
            return
    
    # Add category prefix if specified
    if category:
        message = f"[{category.upper()}] {message}"
        
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