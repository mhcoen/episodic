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
    # Build ANSI codes manually to ensure proper bold formatting
    codes = []
    
    # Bold must come first for proper rendering
    if bold:
        codes.append('\033[1m')
    if dim:
        codes.append('\033[2m')
    if underline:
        codes.append('\033[4m')
    if blink:
        codes.append('\033[5m')
    if reverse:
        codes.append('\033[7m')
    
    # Color codes
    if fg:
        color_map = {
            'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
            'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37,
            'bright_black': 90, 'bright_red': 91, 'bright_green': 92,
            'bright_yellow': 93, 'bright_blue': 94, 'bright_magenta': 95,
            'bright_cyan': 96, 'bright_white': 97
        }
        if fg in color_map:
            codes.append(f'\033[{color_map[fg]}m')
    
    if bg:
        bg_color_map = {
            'black': 40, 'red': 41, 'green': 42, 'yellow': 43,
            'blue': 44, 'magenta': 45, 'cyan': 46, 'white': 47
        }
        if bg in bg_color_map:
            codes.append(f'\033[{bg_color_map[bg]}m')
    
    # Output the formatted text
    prefix = ''.join(codes)
    suffix = '\033[0m' if codes else ''
    newline = '\n' if nl else ''
    
    sys.stdout.write(f"{prefix}{text}{suffix}{newline}")
    sys.stdout.flush()


# Initialize color forcing on import if needed
if should_force_color():
    force_color_output()