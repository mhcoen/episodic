"""
Unified text formatting and display module for Episodic.

This module provides consistent text formatting across the application,
including word wrapping, bold formatting, and colored values in lists.
Supports both instant display (for system outputs) and streaming (for LLM responses).
"""

import re
import textwrap
import shutil
from typing import Optional, Tuple, List

import typer
from episodic.configuration import get_llm_color, get_system_color, get_text_color, get_heading_color
from episodic.color_utils import secho_color, force_color_output

# Force color output if needed
force_color_output()


def get_wrap_width() -> int:
    """Get the appropriate text wrapping width for the terminal."""
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    margin = 4
    max_width = 100  # Maximum line length for readability
    return min(terminal_width - margin, max_width)


def format_and_display_text(text: str, base_color: Optional[str] = None, 
                           value_color: Optional[str] = None,
                           wrap: bool = True) -> None:
    """
    Format and display text with proper wrapping, bold formatting, and colored values.
    
    This handles:
    - Word wrapping based on terminal width
    - Bold text (marked with **)
    - Colored values after colons in bulleted lists
    - Markdown-style headers
    
    Args:
        text: The text to format and display
        base_color: Base color for regular text (defaults to text_color)
        value_color: Color for values after colons (defaults to system_color)
        wrap: Whether to apply word wrapping
    """
    if base_color is None:
        base_color = get_text_color()
    if value_color is None:
        value_color = get_system_color()
    
    # The color functions already return strings, but lowercase them for typer
    if isinstance(base_color, str):
        base_color = base_color.lower()
    if isinstance(value_color, str):
        value_color = value_color.lower()
    
    wrap_width = get_wrap_width() if wrap else None
    
    # Split into lines for processing
    lines = text.split('\n')
    
    for line in lines:
        # Check for headers
        if line.startswith('# '):
            heading_fg = get_heading_color()
            if isinstance(heading_fg, str):
                heading_fg = heading_fg.lower()
            secho_color(line[2:], fg=heading_fg, bold=True)
            continue
        elif line.startswith('## '):
            system_fg = get_system_color()
            if isinstance(system_fg, str):
                system_fg = system_fg.lower()
            secho_color(line[3:], fg=system_fg, bold=True)
            continue
        elif line.startswith('### '):
            system_fg = get_system_color()
            if isinstance(system_fg, str):
                system_fg = system_fg.lower()
            secho_color(line[4:], fg=system_fg, bold=True)
            continue
        elif line.startswith('#### '):
            # Level 4 headers - make them stand out with underline
            header_text = line[5:]
            header_color = get_heading_color()
            if isinstance(header_color, str):
                header_color = header_color.lower()
            secho_color(f"\n{header_text}", fg=header_color, bold=True)
            secho_color("─" * len(header_text), fg=header_color)
            continue
        
        # Check if this is a bulleted list item with a colon
        bullet_match = re.match(r'^(\s*)([-•*])\s+(.+)$', line)
        
        if bullet_match:
            indent = bullet_match.group(1)
            bullet = bullet_match.group(2)
            content = bullet_match.group(3)
            
            # Check if there's a bold section ending with colon
            # Pattern: **Key**: Value or **Key**:\tValue (with tab for alignment)
            colon_match = re.match(r'^\*\*([^*]+)\*\*:(\s*)(.*)$', content)
            
            if colon_match:
                # Bulleted list with key:value format
                key = colon_match.group(1)
                spacing = colon_match.group(2)  # Preserve original spacing (including tabs)
                value = colon_match.group(3)
                
                # Format the line, preserving the original spacing
                formatted_line = f"{indent}{bullet} {key}:{spacing}{value}"
                
                if wrap_width and len(formatted_line) > wrap_width:
                    # Wrap the line preserving structure
                    first_line = f"{indent}{bullet} {key}:{spacing}"
                    
                    # Calculate indent for wrapped lines (align with where the value starts)
                    # This should align with the start of the value text, not just after the bullet
                    wrap_indent = ' ' * len(first_line)
                    
                    # Wrap the value part
                    if value:
                        # Calculate available width for the first line
                        first_line_available = wrap_width - len(first_line)
                        
                        # If the value is long, we need to wrap it properly
                        # Use textwrap with the correct widths for first and subsequent lines
                        wrapped_value = textwrap.fill(
                            value,
                            width=wrap_width,
                            initial_indent='',
                            subsequent_indent=wrap_indent,
                            break_long_words=False,
                            break_on_hyphens=True
                        )
                        
                        # However, we need to adjust because the first line already has the prefix
                        # So we need to re-wrap considering the first line's prefix
                        value_lines = value.split()
                        current_line = ""
                        all_lines = []
                        first_line_done = False
                        
                        for word in value_lines:
                            if not first_line_done:
                                # First line - check against available space after prefix
                                if len(current_line) + len(word) + (1 if current_line else 0) <= first_line_available:
                                    current_line += (" " if current_line else "") + word
                                else:
                                    # First line is full
                                    all_lines.append(current_line)
                                    current_line = word
                                    first_line_done = True
                            else:
                                # Subsequent lines - check against full wrap width
                                if len(wrap_indent + current_line + " " + word) <= wrap_width:
                                    current_line += " " + word
                                else:
                                    # Line is full
                                    all_lines.append(current_line)
                                    current_line = word
                        
                        # Add any remaining text
                        if current_line:
                            all_lines.append(current_line)
                        
                        # Print first line with key
                        secho_color(first_line, fg=base_color, bold=True, nl=False)
                        
                        # Print value lines with color
                        if all_lines:
                            secho_color(all_lines[0], fg=value_color)
                            for vline in all_lines[1:]:
                                secho_color(wrap_indent + vline, fg=value_color)
                    else:
                        secho_color(first_line.rstrip(), fg=base_color, bold=True)
                else:
                    # No wrapping needed
                    secho_color(f"{indent}{bullet} ", fg=base_color, nl=False)
                    secho_color(f"{key}:", fg=base_color, bold=True, nl=False)
                    secho_color(spacing, fg=base_color, nl=False)
                    if value:
                        secho_color(value, fg=value_color)
                    else:
                        typer.echo()
            else:
                # Regular bulleted item without key:value format
                _display_formatted_line(line, base_color, wrap_width)
        else:
            # Regular line - check for inline formatting
            _display_formatted_line(line, base_color, wrap_width)


def _display_formatted_line(line: str, color: str, wrap_width: Optional[int]) -> None:
    """Display a single line with bold formatting and word wrapping."""
    if not line.strip():
        typer.echo()
        return
    
    # Handle lines with bold markers
    if '**' in line:
        _display_line_with_bold(line, color, wrap_width)
    else:
        # No special formatting
        if wrap_width and len(line) > wrap_width:
            wrapped = textwrap.fill(line, width=wrap_width)
            secho_color(wrapped, fg=color)
        else:
            secho_color(line, fg=color)


def _display_line_with_bold(line: str, color: str, wrap_width: Optional[int]) -> None:
    """Display a line that contains bold markers (**)."""
    # Split line into segments by ** markers
    parts = line.split('**')
    
    if wrap_width:
        # Need to handle wrapping while preserving bold
        current_line = ""
        current_length = 0
        
        for i, part in enumerate(parts):
            is_bold = i % 2 == 1  # Odd indices are bold
            
            if current_length + len(part) > wrap_width:
                # Need to wrap
                if current_line:
                    typer.echo()  # New line
                    current_line = ""
                    current_length = 0
                
                # Wrap this part
                wrapped = textwrap.fill(part, width=wrap_width)
                lines = wrapped.split('\n')
                
                for j, wline in enumerate(lines):
                    if j > 0:
                        typer.echo()  # New line
                    secho_color(wline, fg=color, bold=is_bold, nl=False)
                
                current_length = len(lines[-1])
            else:
                secho_color(part, fg=color, bold=is_bold, nl=False)
                current_length += len(part)
        
        typer.echo()  # Final newline
    else:
        # No wrapping - simple display
        for i, part in enumerate(parts):
            is_bold = i % 2 == 1
            secho_color(part, fg=color, bold=is_bold, nl=False)
        typer.echo()


def format_key_value_list(items: List[Tuple[str, str]], 
                         bullet: str = "•",
                         indent: str = "  ") -> str:
    """
    Format a list of key-value pairs for display.
    
    Args:
        items: List of (key, value) tuples
        bullet: Bullet character to use
        indent: Indentation for each item
        
    Returns:
        Formatted text ready for display
    """
    lines = []
    for key, value in items:
        lines.append(f"{indent}{bullet} **{key}**: {value}")
    return '\n'.join(lines)


def display_markdown(text: str, base_color: Optional[str] = None,
                    value_color: Optional[str] = None) -> None:
    """
    Display markdown-formatted text with proper formatting.
    
    This is a convenience wrapper around format_and_display_text
    that ensures markdown formatting is properly handled.
    """
    format_and_display_text(text, base_color, value_color)


def unified_format_and_display(text: str, base_color: Optional[str] = None, 
                               value_color: Optional[str] = None,
                               instant: bool = True) -> None:
    """
    Unified formatting function that respects text_wrap config and supports both
    instant display (for system outputs) and streaming (for LLM responses).
    
    Args:
        text: The text to format and display
        base_color: Base color for regular text (defaults to text_color)
        value_color: Color for values after colons (defaults to system_color)
        instant: If True, displays instantly. If False, could be extended for streaming.
    """
    from episodic.config import config
    
    # Check if wrapping is enabled in config
    wrap_enabled = config.get('text_wrap', True)
    
    # Use the existing formatting logic
    format_and_display_text(text, base_color, value_color, wrap=wrap_enabled)


def display_help_content(content: str, base_color: Optional[str] = None) -> None:
    """
    Display help content with unified formatting and wrapping.
    
    This is a convenience function specifically for help displays that
    automatically uses the correct colors and respects text_wrap settings.
    
    Args:
        content: The help content to display
        base_color: Override base color (defaults to text_color)
    """
    if base_color is None:
        base_color = get_text_color()
    
    unified_format_and_display(content, base_color=base_color, instant=True)



