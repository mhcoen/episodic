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
from episodic.config import config
from episodic.configuration import get_llm_color, get_system_color, get_text_color, get_heading_color


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
            typer.secho(line[2:], fg=heading_fg, bold=True)
            continue
        elif line.startswith('## '):
            system_fg = get_system_color()
            if isinstance(system_fg, str):
                system_fg = system_fg.lower()
            typer.secho(line[3:], fg=system_fg, bold=True)
            continue
        elif line.startswith('### '):
            system_fg = get_system_color()
            if isinstance(system_fg, str):
                system_fg = system_fg.lower()
            typer.secho(line[4:], fg=system_fg, bold=True)
            continue
        elif line.startswith('#### '):
            # Level 4 headers - make them stand out with underline
            header_text = line[5:]
            header_color = get_heading_color()
            if isinstance(header_color, str):
                header_color = header_color.lower()
            typer.secho(f"\n{header_text}", fg=header_color, bold=True)
            typer.secho("─" * len(header_text), fg=header_color)
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
                        typer.secho(first_line, fg=base_color, bold=True, nl=False)
                        
                        # Print value lines with color
                        value_lines = wrapped_value.split('\n')
                        typer.secho(value_lines[0], fg=value_color)
                        for vline in value_lines[1:]:
                            typer.secho(vline, fg=value_color)
                    else:
                        typer.secho(first_line.rstrip(), fg=base_color, bold=True)
                else:
                    # No wrapping needed
                    typer.secho(f"{indent}{bullet} ", fg=base_color, nl=False)
                    typer.secho(f"{key}: ", fg=base_color, bold=True, nl=False)
                    if value:
                        typer.secho(value, fg=value_color)
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
            typer.secho(wrapped, fg=color)
        else:
            typer.secho(line, fg=color)


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
                    typer.secho(wline, fg=color, bold=is_bold, nl=False)
                
                current_length = len(lines[-1])
            else:
                typer.secho(part, fg=color, bold=is_bold, nl=False)
                current_length += len(part)
        
        typer.echo()  # Final newline
    else:
        # No wrapping - simple display
        for i, part in enumerate(parts):
            is_bold = i % 2 == 1
            typer.secho(part, fg=color, bold=is_bold, nl=False)
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


class StreamingFormatter:
    """
    Handles formatting for streaming text output.
    
    This class maintains state across chunks to properly handle
    formatting that spans multiple chunks.
    """
    
    def __init__(self, base_color: Optional[str] = None, 
                 value_color: Optional[str] = None,
                 wrap_width: Optional[int] = None):
        self.base_color = base_color or get_text_color()
        self.value_color = value_color or get_system_color()
        self.wrap_width = wrap_width if wrap_width is not None else get_wrap_width()
        
        # The color functions already return strings, but lowercase them for typer
        if isinstance(self.base_color, str):
            self.base_color = self.base_color.lower()
        if isinstance(self.value_color, str):
            self.value_color = self.value_color.lower()
        
        # State tracking
        self.current_line = ""
        self.line_position = 0
        self.in_bold = False
        self.bold_count = 0
        self.in_bullet_key = False  # Tracking if we're in a bullet list key
        self.after_colon = False    # Tracking if we're after a colon in a bullet
        self.is_bullet_line = False
        
    def process_chunk(self, chunk: str) -> None:
        """Process a chunk of streaming text."""
        for char in chunk:
            if char == '*':
                self.bold_count += 1
                if self.bold_count == 2:
                    # Toggle bold state
                    self.in_bold = not self.in_bold
                    self.bold_count = 0
                continue
            elif self.bold_count == 1:
                # Single asterisk, add it to output
                self._output_char('*')
                self.bold_count = 0
            
            if char == '\n':
                # End of line - flush current line
                self._flush_line()
                typer.echo()
                self.line_position = 0
                self.current_line = ""
                self.is_bullet_line = False
                self.in_bullet_key = False
                self.after_colon = False
            else:
                self.current_line += char
                
                # Check if we're starting a bullet line
                if self.line_position == 0 and not self.is_bullet_line:
                    # Check for bullet pattern at start of line
                    bullet_match = re.match(r'^(\s*)([-•*])\s+', self.current_line)
                    if bullet_match:
                        self.is_bullet_line = True
                        # If the next content starts with **, we're in a key
                        remaining = self.current_line[len(bullet_match.group(0)):]
                        if remaining.startswith('**'):
                            self.in_bullet_key = True
                
                # Check for colon in bullet line
                if self.is_bullet_line and not self.after_colon and char == ':':
                    # Check if we're at the end of a bold key
                    if self.in_bold and self.in_bullet_key:
                        self.after_colon = True
                
                self._output_char(char)
    
    def _output_char(self, char: str) -> None:
        """Output a single character with appropriate formatting."""
        # Determine color
        if self.after_colon and self.is_bullet_line:
            color = self.value_color
        else:
            color = self.base_color
        
        # Determine if bold
        is_bold = self.in_bold and not self.after_colon
        
        # Check for wrapping
        if self.wrap_width and self.line_position >= self.wrap_width and char == ' ':
            typer.echo()
            self.line_position = 0
            return
        
        typer.secho(char, fg=color, bold=is_bold, nl=False)
        self.line_position += 1
    
    def _flush_line(self) -> None:
        """Flush any remaining content in the current line."""
        # Nothing special needed - characters already output
        pass
    
    def finish(self) -> None:
        """Finish streaming and ensure proper line ending."""
        if self.current_line and not self.current_line.endswith('\n'):
            typer.echo()