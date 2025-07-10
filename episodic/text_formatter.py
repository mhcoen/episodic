"""
Unified text formatting and display module for Episodic.

This module provides consistent text formatting across the application,
including word wrapping, bold formatting, and colored values in lists.
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
            # Pattern: **Key**: Value
            colon_match = re.match(r'^\*\*([^*]+)\*\*:\s*(.*)$', content)
            
            if colon_match:
                # Bulleted list with key:value format
                key = colon_match.group(1)
                value = colon_match.group(2)
                
                # Format the line
                formatted_line = f"{indent}{bullet} {key}: {value}"
                
                if wrap_width and len(formatted_line) > wrap_width:
                    # Wrap the line preserving structure
                    first_line = f"{indent}{bullet} {key}: "
                    
                    # Calculate indent for wrapped lines (align with text after bullet)
                    wrap_indent = ' ' * (len(indent) + len(bullet) + 2)
                    
                    # Wrap the value part
                    if value:
                        wrapped_value = textwrap.fill(
                            value,
                            width=wrap_width,
                            initial_indent='',
                            subsequent_indent=wrap_indent
                        )
                        
                        # Print first line with key
                        secho_color(first_line, fg=base_color, bold=True, nl=False)
                        
                        # Print value lines with color
                        value_lines = wrapped_value.split('\n')
                        secho_color(value_lines[0], fg=value_color)
                        for vline in value_lines[1:]:
                            secho_color(vline, fg=value_color)
                    else:
                        secho_color(first_line.rstrip(), fg=base_color, bold=True)
                else:
                    # No wrapping needed
                    secho_color(f"{indent}{bullet} ", fg=base_color, nl=False)
                    secho_color(f"{key}: ", fg=base_color, bold=True, nl=False)
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


def stream_with_word_wrap(stream_generator, model: str, color: Optional[str] = None, 
                         wrap_width: Optional[int] = None, prefix: Optional[str] = None) -> str:
    """
    Stream output with word wrapping and bold support.
    
    Args:
        stream_generator: Generator yielding text chunks
        model: Model name (for process_stream_response)
        color: Color for output (defaults to LLM color)
        wrap_width: Width for wrapping (defaults to terminal width)
        
    Returns:
        Complete response text
    """
    from episodic.llm import process_stream_response
    from episodic.config import config
    
    if color is None:
        color = get_llm_color()
    if isinstance(color, str):
        color = color.lower()
        
    if wrap_width is None:
        wrap_width = get_wrap_width() if config.get("text_wrap", True) else None
    
    # State tracking
    full_response_parts = []
    current_word = ""
    current_position = 0
    in_bold = False
    bold_count = 0
    line_start = True
    in_numbered_list = False
    in_header = False
    header_level = 0
    
    for chunk_data in process_stream_response(stream_generator, model):
        chunk = chunk_data.get('content', '')
        full_response_parts.append(chunk)
        
        for char in chunk:
            # Check for header markers at line start
            if line_start and char == '#':
                if not in_header:
                    in_header = True
                    header_level = 1
                else:
                    header_level += 1
                continue
            elif in_header and header_level > 0 and char == ' ' and line_start:
                # Skip only the first space after header markers
                line_start = False
                continue
            
            if char == '*':
                bold_count += 1
                if bold_count == 2:
                    # Toggle bold state
                    in_bold = not in_bold
                    bold_count = 0
                continue
            elif bold_count == 1:
                # Single asterisk, add it to word
                current_word += '*'
                bold_count = 0
            
            if char in ' \n':
                # End of word
                if current_word:
                    # Check if this is a numbered list item at the start of a line
                    word_is_bold = in_bold or in_header
                    # Check for numbered list: "1." or "1" at start of line
                    if line_start and len(current_word) > 0:
                        # Strip trailing period and check if it's a digit
                        word_without_period = current_word.rstrip('.')
                        if word_without_period.isdigit() and len(word_without_period) <= 2:
                            in_numbered_list = True  # Start bolding for numbered list
                    
                    # Bold everything in numbered list until colon
                    if in_numbered_list:
                        word_is_bold = True
                    
                    # Check if this word ends with colon to stop bolding next words
                    if current_word.endswith(':') and in_numbered_list:
                        word_is_bold = True
                        # Will reset in_numbered_list after printing this word
                    
                    # Check wrap - account for the space we're about to print
                    space_len = 1 if current_position > 0 else 0
                    if wrap_width and current_position + space_len + len(current_word) > wrap_width:
                        secho_color('\n', nl=False)
                        current_position = 0
                        line_start = True
                        space_len = 0  # Don't print space at start of new line
                    
                    # Print word
                    secho_color(current_word, fg=color, nl=False, bold=word_is_bold)
                    current_position += len(current_word)
                    
                    # Reset numbered list flag after printing word with colon
                    if current_word.endswith(':') and in_numbered_list:
                        in_numbered_list = False
                    
                    current_word = ""
                    
                    if not char.isspace():
                        line_start = False
                
                # Print space or newline
                if char == '\n':
                    secho_color('\n', nl=False)
                    current_position = 0
                    line_start = True
                    in_numbered_list = False  # Reset after newline
                    in_header = False  # Reset header state
                else:
                    # Space
                    if current_position > 0:  # Don't print leading spaces
                        secho_color(' ', fg=color, nl=False)
                        current_position += 1
            else:
                # Accumulate character
                current_word += char
    
    # Print any remaining word
    if current_word:
        word_is_bold = in_bold or in_header or (in_numbered_list and not current_word.endswith(':'))
        
        # Check wrap
        if wrap_width and current_position + len(current_word) > wrap_width:
            secho_color('\n', nl=False)
        
        secho_color(current_word, fg=color, nl=False, bold=word_is_bold)
    
    # Ensure we end with a newline
    if current_position > 0:
        typer.echo()
    
    return ''.join(full_response_parts)