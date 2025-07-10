"""
Color utilities for forcing color output in non-TTY environments.
"""

import os
import sys
from typing import Optional
import click


def force_color_output():
    """Force color output even in non-TTY environments."""
    # Set environment variables
    os.environ['FORCE_COLOR'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Disable click's ANSI stripping
    if hasattr(click.utils, '_ansi_colors_disabled'):
        click.utils._ansi_colors_disabled = False
    
    # Force click to think we support color
    if hasattr(click, '_compat'):
        # Monkey patch isatty to always return True for color support
        click._compat.isatty
        click._compat.isatty = lambda stream: True


def should_force_color() -> bool:
    """Determine if we should force color output."""
    # Check environment variables
    if os.environ.get('FORCE_COLOR', '').lower() in ('1', 'true', 'yes'):
        return True
    
    # Check if NO_COLOR is set (respect it)
    if os.environ.get('NO_COLOR'):
        return False
    
    # Check if we're in a known CI/development environment that supports color
    ci_env_vars = ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'CIRCLECI', 'TRAVIS']
    if any(os.environ.get(var) for var in ci_env_vars):
        return True
    
    # If stdout is not a TTY but stderr is, we might be piped but still want color
    if not sys.stdout.isatty() and sys.stderr.isatty():
        return True
    
    return False


def secho_color(text: str, fg: Optional[str] = None, bg: Optional[str] = None,
                bold: bool = False, dim: bool = False, underline: bool = False,
                blink: bool = False, reverse: bool = False, nl: bool = True):
    """
    Simple wrapper that forces color output.
    
    This is a drop-in replacement for typer.secho that ensures colors work
    even in non-TTY environments.
    """
    # Use click's style and echo with color=True to force colors
    styled = click.style(text, fg=fg, bg=bg, bold=bold, dim=dim,
                       underline=underline, blink=blink, reverse=reverse)
    click.echo(styled, nl=nl, color=True)


# Initialize color forcing on import if needed
if should_force_color():
    force_color_output()