"""
Unified streaming module that extracts the streaming logic from conversation.py
to be used by all streaming outputs (LLM, Muse, Summary, etc.)
"""

import time
import queue
import threading
import math
import random
from typing import Optional, List, Generator

import typer
from episodic.config import config
from episodic.configuration import get_llm_color
from episodic.color_utils import secho_color
from episodic.llm import process_stream_response


def debug_print(message: str, indent: bool = False) -> None:
    """Print debug messages if debug mode is enabled."""
    if config.get("debug", False):
        prefix = "  " if indent else ""
        typer.secho(f"{prefix}[DEBUG] {message}", fg="yellow", err=True)


def unified_stream_response(
    stream_generator: Generator,
    model: str,
    prefix: Optional[str] = None,
    color: Optional[str] = None,
    wrap_width: Optional[int] = None
) -> str:
    """
    Unified streaming function that handles all response streaming with consistent formatting.
    
    This is extracted from conversation.py to ensure all outputs have identical formatting,
    including numbered list bolding up to and including the colon.
    
    Args:
        stream_generator: The stream generator from the LLM
        model: The model name
        prefix: Optional prefix to display before streaming (e.g., "🤖 ", "✨ ")
        color: Optional color override (defaults to LLM color)
        wrap_width: Optional wrap width override
        
    Returns:
        The complete response text
    """
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
    
    # Process the stream and display it
    full_response_parts = []
    
    # Get streaming rate configuration
    stream_rate = config.get("stream_rate", 15)  # Default to 15 words per second
    use_constant_rate = config.get("stream_constant_rate", False)
    use_natural_rhythm = config.get("stream_natural_rhythm", False)
    use_char_streaming = config.get("stream_char_mode", False)
    
    if config.get("debug"):
        debug_print(f"Streaming modes - char: {use_char_streaming}, natural: {use_natural_rhythm}, constant: {use_constant_rate}")
    
    # Get wrap width if not provided
    if wrap_width is None and config.get("text_wrap", True):
        import shutil
        terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        margin = 4
        max_width = 100
        wrap_width = min(terminal_width - margin, max_width)
    
    if use_natural_rhythm and stream_rate > 0:
        # Use natural rhythm streaming with sinusoidal variation
        word_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Function to print words with natural rhythm
        def natural_rhythm_printer():
            import math
            import random
            
            current_line = ""
            line_position = 0
            in_bold = False
            in_list_item = False
            list_indent = 0
            words_per_second = stream_rate
            base_interval = 1.0 / words_per_second  # Base interval in seconds
            
            # Rhythm parameters
            amplitude = base_interval * 0.3  # 30% variation
            period = 4.0  # 4-second cycle (like breathing)
            start_time = time.time()
            
            # Punctuation delays (in seconds)
            punctuation_delays = {
                '.': 0.3,
                '!': 0.3,
                '?': 0.3,
                ':': 0.2,
                ';': 0.2,
                ',': 0.1,
                '\n': 0.4
            }
            
            while not stop_event.is_set() or not word_queue.empty():
                try:
                    word = word_queue.get(timeout=0.1)
                    
                    # Calculate delay with natural rhythm
                    elapsed = time.time() - start_time
                    rhythm_factor = math.sin(2 * math.pi * elapsed / period)
                    interval = base_interval + amplitude * rhythm_factor
                    
                    # Add slight random variation
                    interval = max(0.02, interval + random.uniform(-0.02, 0.02))
                    
                    # Determine if we're at the start of a line
                    line_start = (line_position == 0)
                    
                    # Handle newline-only words specially
                    if word == '\n':
                        typer.echo()
                        line_position = 0
                        in_list_item = False
                        continue
                    
                    # Check if starting a numbered list item
                    is_numbered_list_start = False
                    if line_start and len(word) > 0:
                        word_without_period = word.rstrip('.')
                        if word_without_period.isdigit() and len(word_without_period) <= 2:
                            is_numbered_list_start = True
                            in_list_item = True  # Start bolding
                    
                    # Determine if word should be bold
                    word_is_bold = in_bold or (in_list_item and not word.endswith(':'))
                    
                    # Handle list indentation for continuation lines
                    if line_start and in_list_item and not is_numbered_list_start:
                        # This is a continuation line in a list item
                        secho_color('   ', fg=color, nl=False)  # 3 spaces for indent
                        line_position = 3
                    
                    # Check if we need to wrap
                    if wrap_width and line_position > 0 and line_position + len(word) + 1 > wrap_width:
                        secho_color('\n', fg=color, nl=False)
                        line_position = 0
                        # If in list, add indent on new line
                        if in_list_item:
                            secho_color('   ', fg=color, nl=False)
                            line_position = 3
                    
                    # Add space before word if needed
                    if line_position > 0:
                        secho_color(' ', fg=color, nl=False)
                        line_position += 1
                    
                    # Handle bold markers
                    display_word = word
                    if '**' in word:
                        # Strip bold markers for display
                        parts = word.split('**')
                        display_word = ''
                        for i, part in enumerate(parts):
                            if i % 2 == 1:  # Odd parts are bold
                                in_bold = True
                            else:
                                in_bold = False
                            display_word += part
                    
                    # Strip any remaining ** in the middle
                    display_word = display_word.replace('**', '')
                    
                    # Print the word
                    secho_color(display_word, fg=color, nl=False, bold=word_is_bold)
                    line_position += len(display_word)
                    
                    # Check for line breaks in word
                    if '\n' in display_word:
                        last_newline = display_word.rfind('\n')
                        line_position = len(display_word) - last_newline - 1
                        in_list_item = False  # Reset list state on explicit newline
                    
                    # Check if word ends with colon to stop list bolding
                    if display_word.endswith(':') and in_list_item:
                        in_list_item = False
                    
                    # Apply delay
                    time.sleep(interval)
                    
                    # Check for punctuation delays
                    for punct, delay in punctuation_delays.items():
                        if display_word.rstrip().endswith(punct):
                            time.sleep(delay)
                            break
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    if config.get("debug"):
                        debug_print(f"Natural rhythm printer error: {e}")
                    break
            
            # Ensure we have a newline at the end if needed
            if line_position > 0:
                typer.echo("")
        
        # Start the printer thread
        printer_thread = threading.Thread(target=natural_rhythm_printer)
        printer_thread.daemon = True
        printer_thread.start()
        
        # Feed words to the queue
        try:
            accumulated_text = ""
            for chunk_content in process_stream_response(stream_generator, model):
                if chunk_content:
                    full_response_parts.append(chunk_content)
                    # Accumulate text to handle word boundaries properly
                    accumulated_text += chunk_content
                    
                    # Process accumulated text into words
                    # We want to send complete words to the printer, but keep partial words
                    # for the next iteration to avoid breaking words mid-stream
                    
                    # Find the last space or newline in the accumulated text
                    last_break = max(
                        accumulated_text.rfind(' '),
                        accumulated_text.rfind('\n')
                    )
                    
                    if last_break > 0:
                        # We have at least one complete word
                        complete_text = accumulated_text[:last_break + 1]
                        remaining_text = accumulated_text[last_break + 1:]
                        
                        # Split complete text into words, preserving structure
                        import re
                        # This regex better preserves the original structure
                        words = re.split(r'(\s+)', complete_text)
                        
                        for word in words:
                            if word:  # Skip empty strings from split
                                if word.isspace():
                                    # Handle whitespace specially
                                    if '\n' in word:
                                        # Send each newline separately
                                        for char in word:
                                            if char == '\n':
                                                word_queue.put('\n')
                                            # Ignore other whitespace between words
                                else:
                                    # Regular word
                                    word_queue.put(word)
                        
                        accumulated_text = remaining_text
                    elif '\n' in accumulated_text:
                        # Special case: even if we don't have a space, process newlines
                        parts = accumulated_text.split('\n', 1)
                        if parts[0]:
                            word_queue.put(parts[0])
                        word_queue.put('\n')
                        accumulated_text = parts[1] if len(parts) > 1 else ""
            
            # Process any remaining text
            if accumulated_text.strip():
                word_queue.put(accumulated_text.strip())
        finally:
            stop_event.set()
            printer_thread.join(timeout=1.0)
            
    elif use_constant_rate and stream_rate > 0:
        # Constant rate streaming
        word_buffer = []
        words_per_second = stream_rate
        interval = 1.0 / words_per_second
        
        current_position = 0
        line_start = True
        in_bold = False
        in_numbered_list = False
        
        accumulated_text = ""
        for chunk_content in process_stream_response(stream_generator, model):
            if chunk_content:
                full_response_parts.append(chunk_content)
                
                # Accumulate text to handle word boundaries properly
                accumulated_text += chunk_content
                
                # Process accumulated text into words
                # Find the last space or newline in the accumulated text
                last_break = max(
                    accumulated_text.rfind(' '),
                    accumulated_text.rfind('\n')
                )
                
                if last_break > 0:
                    # We have at least one complete word
                    complete_text = accumulated_text[:last_break + 1]
                    remaining_text = accumulated_text[last_break + 1:]
                    
                    # Split complete text into words, preserving structure
                    import re
                    words = re.split(r'(\s+)', complete_text)
                    
                    for word in words:
                        if word:  # Skip empty strings from split
                            if word.isspace():
                                # Handle whitespace specially
                                if '\n' in word:
                                    # Add each newline separately
                                    for char in word:
                                        if char == '\n':
                                            word_buffer.append('\n')
                                        # Ignore other whitespace between words
                            else:
                                # Regular word
                                word_buffer.append(word)
                    
                    accumulated_text = remaining_text
                elif '\n' in accumulated_text:
                    # Special case: process newlines even without space
                    parts = accumulated_text.split('\n', 1)
                    if parts[0]:
                        word_buffer.append(parts[0])
                    word_buffer.append('\n')
                    accumulated_text = parts[1] if len(parts) > 1 else ""
        
        # Process any remaining text
        if accumulated_text.strip():
            word_buffer.append(accumulated_text.strip())
        
        # Process buffered words
        while word_buffer:
            word = word_buffer.pop(0)
            
            # Handle newline-only words specially
            if word == '\n':
                typer.echo()
                current_position = 0
                line_start = True
                in_numbered_list = False
                time.sleep(interval)
                continue
            
            # Check if starting numbered list
            if line_start and len(word) > 0:
                word_without_period = word.rstrip('.')
                if word_without_period.isdigit() and len(word_without_period) <= 2:
                    in_numbered_list = True
            
            # Determine if bold
            word_is_bold = in_bold or in_numbered_list
            
            # Handle bold markers
            display_word = word
            if '**' in word:
                parts = word.split('**')
                display_word = ''
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        in_bold = True
                    else:
                        in_bold = False
                    display_word += part
            
            display_word = display_word.replace('**', '')
            
            # Check if we need to wrap
            if wrap_width and current_position > 0 and current_position + len(display_word) + 1 > wrap_width:
                typer.echo()
                current_position = 0
                line_start = True
            
            # Add space if needed
            if current_position > 0:
                secho_color(' ', fg=color, nl=False)
                current_position += 1
            
            # Print word
            secho_color(display_word, fg=color, nl=False, bold=word_is_bold)
            current_position += len(display_word)
            line_start = False
            
            # Check if word ends with colon
            if display_word.endswith(':') and in_numbered_list:
                in_numbered_list = False
            
            # Check for newlines
            if '\n' in display_word:
                current_position = 0
                line_start = True
                in_numbered_list = False
            
            time.sleep(interval)
        
        # Final newline if needed
        if current_position > 0:
            typer.echo("")
            
    else:
        # Immediate display (no rate limiting) - most detailed logic
        current_position = 0
        line_start = True
        in_bold = False
        in_numbered_list = False
        current_word = ""
        
        for chunk_content in process_stream_response(stream_generator, model):
            if chunk_content:
                full_response_parts.append(chunk_content)
                
                # Process character by character
                for char in chunk_content:
                    # Check for bold marker
                    if current_word.endswith('*') and char == '*':
                        # Remove the * from current word
                        current_word = current_word[:-1]
                        
                        # Print pending word if any
                        if current_word:
                            # Check if starting numbered list
                            if line_start and current_word.rstrip('.').isdigit():
                                in_numbered_list = True
                            
                            word_is_bold = in_bold or in_numbered_list
                            
                            # Wrap if needed
                            if wrap_width and current_position > 0 and current_position + len(current_word) + 1 > wrap_width:
                                typer.echo()
                                current_position = 0
                                line_start = True
                            
                            if current_position > 0:
                                secho_color(' ', fg=color, nl=False)
                                current_position += 1
                            
                            secho_color(current_word, fg=color, nl=False, bold=word_is_bold)
                            current_position += len(current_word)
                            
                            if current_word.endswith(':') and in_numbered_list:
                                in_numbered_list = False
                            
                            current_word = ""
                            line_start = False
                        
                        # Toggle bold
                        in_bold = not in_bold
                        continue
                    
                    # Check for word boundary
                    if char in ' \n\t':
                        # End of word
                        if current_word:
                            # Check if starting numbered list
                            if line_start and current_word.rstrip('.').isdigit():
                                in_numbered_list = True
                            
                            word_is_bold = in_bold or in_numbered_list
                            
                            # Wrap if needed
                            if wrap_width and current_position > 0 and current_position + len(current_word) + 1 > wrap_width:
                                typer.echo()
                                current_position = 0
                                line_start = True
                            
                            if current_position > 0:
                                secho_color(' ', fg=color, nl=False)
                                current_position += 1
                            
                            secho_color(current_word, fg=color, nl=False, bold=word_is_bold)
                            current_position += len(current_word)
                            
                            if current_word.endswith(':') and in_numbered_list:
                                in_numbered_list = False
                            
                            current_word = ""
                            line_start = False
                        
                        # Handle the whitespace
                        if char == '\n':
                            typer.echo()
                            current_position = 0
                            line_start = True
                            in_numbered_list = False
                    else:
                        # Accumulate character
                        current_word += char
        
        # Print any remaining word
        if current_word:
            if line_start and current_word.rstrip('.').isdigit():
                in_numbered_list = True
            
            word_is_bold = in_bold or in_numbered_list
            
            if wrap_width and current_position > 0 and current_position + len(current_word) + 1 > wrap_width:
                typer.echo()
            
            if current_position > 0:
                secho_color(' ', fg=color, nl=False)
            
            secho_color(current_word, fg=color, nl=False, bold=word_is_bold)
        
        # Final newline
        typer.echo("")
    
    # Join the full response
    full_response = ''.join(full_response_parts)
    
    # Post-process to ensure markdown headers have proper spacing
    # This ensures headers (###, ##, etc.) always have a blank line before them
    # unless they're at the start of the text or already have proper spacing
    import re
    
    # Replace any header that doesn't have double newline before it (except at start)
    # This matches: any character that's not a newline, followed by a single newline, followed by headers
    full_response = re.sub(r'([^\n\r])\n(#{1,6} )', r'\1\n\n\2', full_response)
    
    # Also handle case where header follows text with just a space
    full_response = re.sub(r'([^\n\r]) (#{1,6} )', r'\1\n\n\2', full_response)
    
    return full_response