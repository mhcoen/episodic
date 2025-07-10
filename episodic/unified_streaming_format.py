"""
Format-preserving streaming functions for unified_streaming module.
"""

import re
import sys
import typer
from typing import Generator, Optional, List
from episodic.config import config
from episodic.configuration import get_llm_color
from episodic.color_utils import secho_color
from episodic.llm import process_stream_response


# Import debug_print from common utilities
from episodic.debug_utils import debug_print


def stream_with_format_preservation(
    stream_generator: Generator,
    model: str,
    prefix: Optional[str],
    color: Optional[str],
    wrap_width: Optional[int]
) -> str:
    """
    Alternative streaming that preserves indentation and spacing.
    
    This function processes complete lines to maintain formatting while
    still providing a streaming experience.
    """
    if config.get("debug"):
        debug_print("Using format-preserving streaming")
    
    # Display prefix if provided
    if prefix:
        llm_color = color or get_llm_color()
        if isinstance(llm_color, str):
            llm_color = llm_color.lower()
        secho_color(prefix, fg=llm_color, nl=False)
    
    # Get color for streaming
    if not color:
        color = get_llm_color()
        if isinstance(color, str):
            color = color.lower()
    
    full_response = []
    buffer = ""
    line_position = 0
    
    # Process stream
    for chunk in process_stream_response(stream_generator, model):
        if chunk:
            full_response.append(chunk)
            buffer += chunk
            
            # Process complete lines to preserve formatting
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                
                # Check if line needs wrapping (while preserving indent)
                if wrap_width and len(line) > wrap_width:
                    wrapped_lines = _wrap_preserving_indent(line, wrap_width)
                    for wrapped_line in wrapped_lines:
                        _print_formatted_line(wrapped_line, color)
                else:
                    _print_formatted_line(line, color)
                
                line_position = 0
            
            # Handle partial line in buffer
            # Check if buffer needs wrapping
            if wrap_width and len(buffer) > wrap_width:
                # Buffer is too long, we need to wrap it
                # Find last space before wrap_width
                wrap_point = buffer[:wrap_width].rfind(' ')
                if wrap_point > 0:
                    # Print up to wrap point
                    _print_formatted_line(buffer[:wrap_point], color)
                    # Keep rest in buffer with proper indentation
                    remaining = buffer[wrap_point + 1:]
                    # Detect indentation from the line
                    indent_match = re.match(r'^(\s*)', buffer)
                    indent = indent_match.group(1) if indent_match else ''
                    buffer = indent + remaining
                    line_position = len(buffer)
                else:
                    # No good break point, flush what we have
                    _print_formatted_line(buffer, color, newline=False)
                    line_position += len(buffer)
                    buffer = ""
            elif len(buffer) > 80 or (buffer and not buffer[-1].isalnum()):
                # Only flush if it's getting long or looks complete
                _print_formatted_line(buffer, color, newline=False)
                line_position += len(buffer)
                buffer = ""
    
    # Final buffer
    if buffer:
        _print_formatted_line(buffer, color, newline=False)
    
    # Ensure final newline
    typer.echo()
    
    return ''.join(full_response)


def _wrap_preserving_indent(line: str, wrap_width: int) -> List[str]:
    """
    Wrap a line while preserving its indentation.
    
    Args:
        line: The line to wrap
        wrap_width: Maximum width for each line
        
    Returns:
        List of wrapped lines with preserved indentation
    """
    # Detect indentation
    indent_match = re.match(r'^(\s*)', line)
    indent = indent_match.group(1) if indent_match else ''
    indent_width = len(indent)
    
    # If line fits, return as-is
    if len(line) <= wrap_width:
        return [line]
    
    # Extract content after indentation
    content = line[indent_width:]
    wrapped_lines = []
    
    while content:
        # Calculate available width for content
        available_width = wrap_width - indent_width
        
        if len(content) <= available_width:
            # Remaining content fits
            wrapped_lines.append(indent + content)
            break
        
        # Find a good break point (prefer spaces)
        break_point = content[:available_width].rfind(' ')
        
        if break_point > 0:
            # Break at space
            wrapped_lines.append(indent + content[:break_point])
            content = content[break_point + 1:]  # Skip the space
        else:
            # No space found, break at width
            wrapped_lines.append(indent + content[:available_width])
            content = content[available_width:]
    
    return wrapped_lines


def _print_formatted_line(line: str, color: str, newline: bool = True):
    """
    Print a line with formatting detection and preservation.
    
    Handles bold text markers while preserving spacing.
    """
    # Check for bold markers
    if '**' not in line:
        # No bold markers, print as-is with color
        typer.secho(line, fg=color, nl=newline)
        return
    
    # Process line with bold markers - handle ** that might be split across words
    
    # First, let's handle the case where ** might not have content between them
    # or might be malformed (e.g., "**bold** text **more**")
    parts = []
    current_pos = 0
    
    # Find all ** positions
    while True:
        start = line.find('**', current_pos)
        if start == -1:
            # No more **, add rest of line
            if current_pos < len(line):
                parts.append(('text', line[current_pos:]))
            break
            
        # Add text before **
        if start > current_pos:
            parts.append(('text', line[current_pos:start]))
        
        # Find closing **
        end = line.find('**', start + 2)
        if end == -1:
            # No closing **, treat as regular text
            parts.append(('text', line[start:]))
            break
        
        # Add bold text
        bold_content = line[start + 2:end]
        if bold_content:  # Only add if there's actual content
            parts.append(('bold', bold_content))
        
        current_pos = end + 2
    
    # Now print the parts
    for part_type, content in parts:
        if part_type == 'bold':
            # Use raw ANSI codes for bold + color
            color_codes = {
                'cyan': '\033[36m',
                'green': '\033[32m',
                'yellow': '\033[33m',
                'blue': '\033[34m',
                'magenta': '\033[35m',
                'red': '\033[31m',
                'white': '\033[37m',
            }
            color_code = color_codes.get(color, '\033[37m')
            sys.stdout.write(f"{color_code}\033[1m{content}\033[0m")
            sys.stdout.flush()
        else:
            # Regular text
            typer.secho(content, fg=color, nl=False)
    
    # End of line
    if newline:
        typer.echo()