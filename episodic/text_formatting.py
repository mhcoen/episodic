"""
Text formatting utilities for Episodic.

This module provides text wrapping and formatting functionality
for terminal output.
"""

import shutil
import textwrap
import re
import typer
from episodic.color_utils import secho_color
from episodic.config import config


def get_wrap_width() -> int:
    """Get the appropriate text wrapping width for the terminal."""
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    margin = 4
    max_width = 100  # Maximum line length for readability
    # Use terminal width or 100, whichever is smaller (with minimum of 40)
    return min(max_width, max(40, terminal_width - margin))


def wrapped_text_print(text: str, **typer_kwargs) -> None:
    """Print text with automatic wrapping while preserving formatting."""
    # Check if wrapping is enabled (default to True)
    if config.get("text_wrap", True) == False:
        secho_color(str(text), **typer_kwargs)
        return
    
    wrap_width = get_wrap_width()
    
    # Process text to preserve formatting while wrapping long lines
    lines = str(text).split('\n')
    wrapped_lines = []
    
    for line in lines:
        if len(line) <= wrap_width:
            # Line is short enough, keep as-is
            wrapped_lines.append(line)
        else:
            # Line is too long, wrap it while preserving indentation
            # Detect indentation (spaces or tabs at start of line)
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            
            # Wrap the content while preserving indentation
            if stripped:  # Only wrap if there's actual content
                wrapped = textwrap.fill(
                    stripped, 
                    width=wrap_width,
                    initial_indent=indent,
                    subsequent_indent=indent  # Same indent as first line
                )
                wrapped_lines.append(wrapped)
            else:
                # Empty line - preserve as-is
                wrapped_lines.append(line)
    
    # Join the processed lines back together
    wrapped_text = '\n'.join(wrapped_lines)
    
    # Print with the specified formatting
    secho_color(wrapped_text, **typer_kwargs)


def wrapped_llm_print(text: str, **typer_kwargs) -> None:
    """Print LLM text with automatic wrapping while preserving formatting."""
    # First handle bold markers
    # Split text by bold markers
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            # This is bold text
            bold_text = part[2:-2]
            wrapped_text_print(bold_text, bold=True, **typer_kwargs)
        else:
            # Regular text
            wrapped_text_print(part, **typer_kwargs)


def debug_print(message: str, indent: bool = False) -> None:
    """Print debug message with consistent formatting."""
    if indent:
        secho_color(f"   {message}", fg='bright_cyan')
    else:
        secho_color(f"🔍 DEBUG: {message}", fg='yellow', bold=True)