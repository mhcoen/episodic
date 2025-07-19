"""
Utility functions for drawing boxes in the terminal.
"""

import shutil
from typing import List, Optional
import typer
from episodic.configuration import get_text_color


def draw_input_box(text: str, width: Optional[int] = None, color: Optional[str] = None) -> None:
    """
    Draw a box around user input text.
    
    Args:
        text: The text to display in the box
        width: Optional fixed width for the box (defaults to terminal width - 4)
        color: Optional color for the box (defaults to dim text color)
    """
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    
    # Determine box width
    if width is None:
        # First, wrap text to a reasonable width for measurement
        text_lines = _wrap_text(text, min(terminal_width - 8, 76))
        max_line_length = max(len(line) for line in text_lines) if text_lines else 0
        
        # Calculate the width needed for the content with prefix
        # First line: "> " + text
        # Other lines: "  " + text  
        content_width = max_line_length + 2  # +2 for the prefix
        
        # Total box width = content + 2 spaces (padding) + 2 borders
        box_width = content_width + 4
        
        # Apply constraints
        box_width = min(box_width, terminal_width - 4, 80)
        box_width = max(box_width, 12)  # Minimum reasonable width
    else:
        box_width = min(width, terminal_width - 4)
    
    # Use a very faint gray color for the box
    if color is None:
        # Use ANSI 256-color mode for a specific gray
        color = "bright_black"  # This is a darker gray
    
    # Split text into lines if needed
    # Internal width = box_width - 4 (2 for borders, 2 for padding spaces)
    # But we also need to account for the "> " prefix, so -2 more
    lines = _wrap_text(text, box_width - 6)
    
    # Draw top border with explicit ANSI color for faint appearance
    print(f"\033[38;5;236m╭{'─' * (box_width - 2)}╮\033[0m")
    
    # Draw text lines with prefix
    for i, line in enumerate(lines):
        if i == 0:
            # First line gets the > prefix
            content = f"> {line}"
        else:
            # Continuation lines are indented
            content = f"  {line}"
        
        # When we print, we add a space before and after content: "│ > hi there │"
        # Total line length = 1 (│) + 1 (space) + content + padding + 1 (space) + 1 (│)
        # So: box_width = 2 + len(content) + padding + 2
        # Therefore: padding = box_width - len(content) - 4
        padding = box_width - len(content) - 4
        padded_content = content + " " * padding
        
        # Draw the line with faint gray borders
        print(f"\033[38;5;236m│\033[0m {padded_content} \033[38;5;236m│\033[0m")
    
    # Draw bottom border
    print(f"\033[38;5;236m╰{'─' * (box_width - 2)}╯\033[0m")


def draw_simple_input_box(text: str, width: Optional[int] = None, color: Optional[str] = None) -> None:
    """
    Draw a simple box using ASCII characters (fallback for terminals without Unicode).
    
    Args:
        text: The text to display in the box
        width: Optional fixed width for the box
        color: Optional color for the box
    """
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    
    # Determine box width
    if width is None:
        # First, wrap text to a reasonable width for measurement
        text_lines = _wrap_text(text, min(terminal_width - 8, 76))
        max_line_length = max(len(line) for line in text_lines) if text_lines else 0
        
        # Calculate the width needed for the content with prefix
        # First line: "> " + text
        # Other lines: "  " + text  
        content_width = max_line_length + 2  # +2 for the prefix
        
        # Total box width = content + 2 spaces (padding) + 2 borders
        box_width = content_width + 4
        
        # Apply constraints
        box_width = min(box_width, terminal_width - 4, 80)
        box_width = max(box_width, 12)  # Minimum reasonable width
    else:
        box_width = min(width, terminal_width - 4)
    
    # Use a very faint gray color for the box
    if color is None:
        color = "bright_black"
    
    # Split text into lines if needed
    # Same calculation as Unicode version
    lines = _wrap_text(text, box_width - 6)
    
    # Draw top border
    print(f"\033[38;5;236m+{'-' * (box_width - 2)}+\033[0m")
    
    # Draw text lines
    for i, line in enumerate(lines):
        if i == 0:
            content = f"> {line}"
        else:
            content = f"  {line}"
        
        padding = box_width - len(content) - 4
        padded_content = content + " " * padding
        
        print(f"\033[38;5;236m|\033[0m {padded_content} \033[38;5;236m|\033[0m")
    
    # Draw bottom border
    print(f"\033[38;5;236m+{'-' * (box_width - 2)}+\033[0m")


def _wrap_text(text: str, width: int) -> List[str]:
    """
    Wrap text to fit within specified width.
    
    Args:
        text: The text to wrap
        width: Maximum width for each line
        
    Returns:
        List of wrapped lines
    """
    if not text:
        return [""]
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        
        # If adding this word would exceed width, start new line
        if current_length > 0 and current_length + 1 + word_length > width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            if current_line:
                current_length += 1  # Space before word
            current_line.append(word)
            current_length += word_length
    
    # Add final line
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines if lines else [""]