"""
Unified streaming module that extracts the streaming logic from conversation.py
to be used by all streaming outputs (LLM, Muse, Summary, etc.)
"""

import time
import math
import random
from typing import Optional, Generator

import typer
from episodic.config import config
from episodic.configuration import get_llm_color
from episodic.color_utils import secho_color
from episodic.llm import process_stream_response


# Import debug_print from common utilities
from episodic.debug_utils import debug_print


def unified_stream_response(
    stream_generator: Generator,
    model: str,
    prefix: Optional[str] = None,
    color: Optional[str] = None,
    wrap_width: Optional[int] = None,
    preserve_formatting: Optional[bool] = None
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
        preserve_formatting: If True, preserves indentation and multiple spaces. 
                           If None, auto-detects based on content.
        
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
    
    # Auto-detect if format preservation is needed
    if preserve_formatting is None:
        # Peek at the first chunk to decide
        import itertools
        stream_generator, peek_generator = itertools.tee(stream_generator)
        first_chunk = ""
        try:
            for chunk in process_stream_response(peek_generator, model):
                first_chunk += chunk
                if len(first_chunk) > 200:  # Check first 200 chars
                    break
        except:
            pass
        
        # Check for formatting indicators
        formatting_indicators = [
            '  /',      # Indented commands
            '\n  ',     # Multi-space indentation
            '    ',     # 4-space indentation
            '\t',       # Tabs
            '```',      # Code blocks
            '|',        # Tables
        ]
        preserve_formatting = any(indicator in first_chunk for indicator in formatting_indicators)
        
        if config.get("debug_streaming_verbose", False):
            debug_print(f"Format preservation auto-detected: {preserve_formatting}")
    
    # Route to appropriate streaming function
    if preserve_formatting:
        # Import the format-preserving function
        from episodic.unified_streaming_format import stream_with_format_preservation
        return stream_with_format_preservation(
            stream_generator, model, prefix, color, wrap_width
        )
    
    # Continue with standard streaming
    # Process the stream and display it
    full_response_parts = []
    
    # Get streaming rate configuration
    stream_rate = config.get("stream_rate", 15)  # Default to 15 words per second
    use_constant_rate = config.get("stream_constant_rate", False)
    use_natural_rhythm = config.get("stream_natural_rhythm", False)
    use_char_streaming = config.get("stream_char_mode", False)
    
    if config.get("debug_streaming_verbose", False):
        debug_print(f"Streaming modes - char: {use_char_streaming}, natural: {use_natural_rhythm}, constant: {use_constant_rate}")
    
    # Get wrap width if not provided
    if wrap_width is None and config.get("text_wrap", True):
        import shutil
        terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        margin = 4
        max_width = 100
        wrap_width = min(terminal_width - margin, max_width)
        if config.get("debug_streaming_verbose", False):
            debug_print(f"Terminal width: {terminal_width}, wrap width: {wrap_width}")
    
    # Determine delay calculation method
    if stream_rate <= 0:
        # Immediate mode - no delays
        calculate_delay = lambda elapsed, word: 0
    elif use_natural_rhythm:
        # Natural rhythm with sinusoidal variation
        base_interval = 1.0 / stream_rate
        amplitude = base_interval * 0.3  # 30% variation
        period = 4.0  # 4-second cycle (like breathing)
        
        def calculate_delay(elapsed: float, word: str) -> float:
            rhythm_factor = math.sin(2 * math.pi * elapsed / period)
            interval = base_interval + amplitude * rhythm_factor
            # Add slight random variation
            interval = max(0.02, interval + random.uniform(-0.02, 0.02))
            
            # Add punctuation delays
            punctuation_delays = {
                '.': 0.1, '!': 0.1, '?': 0.1,
                ':': 0.1, ';': 0.1, ',': 0.1, '\n': 0.1
            }
            for punct, delay in punctuation_delays.items():
                if word.rstrip().endswith(punct):
                    interval += delay
                    break
            return interval
    else:
        # Constant rate streaming
        interval = 1.0 / stream_rate
        calculate_delay = lambda elapsed, word: interval
    
    # State tracking
    current_position = 0
    line_start = True
    in_bold = False
    in_numbered_list = False
    in_list_item = False
    in_header = False
    accumulated_text = ""
    start_time = time.time()
    
    # Process stream
    try:
        for chunk_content in process_stream_response(stream_generator, model):
            if chunk_content:
                full_response_parts.append(chunk_content)
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
                                # Process each newline
                                for char in word:
                                    if char == '\n':
                                        _print_word('\n', color, wrap_width, 
                                                   current_position, line_start,
                                                   in_bold, in_numbered_list, in_list_item, in_header)
                                        current_position = 0
                                        line_start = True
                                        in_list_item = False
                                        in_numbered_list = False
                                        in_header = False
                                # Ignore other whitespace between words
                        else:
                            # Regular word - print it
                            current_position, line_start, in_bold, in_numbered_list, in_list_item, in_header = \
                                _print_word(word, color, wrap_width, 
                                          current_position, line_start,
                                          in_bold, in_numbered_list, in_list_item, in_header)
                            
                            # Apply delay
                            elapsed = time.time() - start_time
                            delay = calculate_delay(elapsed, word)
                            if delay > 0:
                                time.sleep(delay)
                
                accumulated_text = remaining_text
            elif '\n' in accumulated_text:
                # Special case: process newlines even without space
                parts = accumulated_text.split('\n', 1)
                if parts[0]:
                    current_position, line_start, in_bold, in_numbered_list, in_list_item, in_header = \
                        _print_word(parts[0], color, wrap_width,
                                  current_position, line_start,
                                  in_bold, in_numbered_list, in_list_item, in_header)
                    # Apply delay
                    elapsed = time.time() - start_time
                    delay = calculate_delay(elapsed, parts[0])
                    if delay > 0:
                        time.sleep(delay)
                
                # Print newline
                _print_word('\n', color, wrap_width,
                           current_position, line_start,
                           in_bold, in_numbered_list, in_list_item, in_header)
                current_position = 0
                line_start = True
                # Reset all formatting states on newline
                in_list_item = False
                in_numbered_list = False
                in_header = False
                accumulated_text = parts[1] if len(parts) > 1 else ""
    
    except KeyboardInterrupt:
        # User pressed Ctrl-C, clean up and return what we have
        typer.echo()  # New line after interrupted output
        typer.secho("\n⚡ Response interrupted", fg="yellow")
        
        # Join what we have so far
        full_response = ''.join(full_response_parts)
        return full_response + "\n[Response interrupted by user]"
    
    # Process any remaining text
    if accumulated_text.strip():
        # Split remaining text into words and process with wrapping
        import re
        remaining_words = re.split(r'(\s+)', accumulated_text)
        for word in remaining_words:
            if word and not word.isspace():  # Skip empty strings and whitespace
                current_position, line_start, in_bold, in_numbered_list, in_list_item, in_header = \
                    _print_word(word, color, wrap_width,
                              current_position, line_start,
                              in_bold, in_numbered_list, in_list_item, in_header)
    
    # Final newline if needed
    if current_position > 0:
        typer.echo("")
    
    # Add extra blank line after muse responses
    if config.get("muse_mode", False):
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


def _print_word(word: str, color: str, wrap_width: Optional[int],
                current_position: int, line_start: bool,
                in_bold: bool, in_numbered_list: bool, in_list_item: bool, in_header: bool) -> tuple:
    """
    Print a single word with appropriate formatting.
    
    Returns:
        tuple: (new_position, new_line_start, new_in_bold, new_in_numbered_list, new_in_list_item, new_in_header)
    """
    # Debug output - only if verbose streaming debug is enabled
    if config.get("debug_streaming_verbose", False) and word != '\n' and not word.isspace():
        debug_print(f"Word: '{word}', line_start={line_start}, in_header={in_header}, in_numbered={in_numbered_list}, in_list={in_list_item}", indent=True)
    
    # Handle newline-only words specially
    if word == '\n':
        typer.echo()
        return 0, True, in_bold, False, False, False
    
    # Check if starting a numbered list item
    if line_start and len(word) > 0:
        word_without_period = word.rstrip('.')
        if word_without_period.isdigit() and len(word_without_period) <= 2:
            in_numbered_list = True
        # Also check for patterns like "1. Advanced" where number and text come together
        elif '.' in word and len(word) > 2:
            prefix = word.split('.')[0]
            if prefix.isdigit() and len(prefix) <= 2:
                in_numbered_list = True
    
    # Check if this is a markdown header and remove the ### prefix
    is_header_start = line_start and word.startswith('#')
    if is_header_start:
        in_header = True
        # Remove the ### prefix from the word
        word = word.lstrip('#').lstrip()  # Remove # symbols and any following space
        if not word:  # If word was just ###, skip it
            return current_position, line_start, in_bold, in_numbered_list, in_list_item, in_header
        if config.get("debug", False):
            debug_print(f"Detected header start, cleaned word: '{word}'", indent=True)
    
    # Check if starting a bulleted list
    is_bullet = line_start and word == '-'
    if is_bullet:
        in_list_item = True
    
    # Strip bold markers to check for colon  
    display_word = word.replace('**', '')
    word_ends_with_colon = display_word.endswith(':')
    
    # Determine if word should be bold (including the colon word itself)
    word_is_bold = in_bold or in_list_item or in_numbered_list or in_header
    
    # Debug: Show why a word is bold when debug is enabled
    if config.get("debug", False) and word_is_bold:
        debug_reasons = []
        if in_bold: debug_reasons.append("in_bold")
        if in_list_item: debug_reasons.append("in_list_item") 
        if in_numbered_list: debug_reasons.append("in_numbered_list")
        if in_header: debug_reasons.append("in_header")
        debug_print(f"'{word}' is bold because: {', '.join(debug_reasons)}", indent=True)
    
    # Debug: Print headers and lists with a marker when debug is enabled
    is_numbered_pattern = word.rstrip('.').isdigit() or ('.' in word and len(word) > 2 and word.split('.')[0].isdigit())
    has_italic_markers = '*' in word and not word.startswith('**')
    debug_enabled = config.get("debug", False)
    
    # Always show debug for formatting patterns when debug is on
    if debug_enabled and (line_start and (word.startswith('#') or is_numbered_pattern or word == '-') or has_italic_markers):
        print(f"\n[DEBUG FORMATTING] Word '{word}' line_start={line_start} header={in_header} num={in_numbered_list} list={in_list_item} bold={word_is_bold} italic={has_italic_markers}", flush=True)
    
    # Check if we need to wrap
    if wrap_width and current_position > 0 and current_position + len(word) + 1 > wrap_width:
        if config.get("debug", False):
            debug_print(f"Wrapping: pos={current_position}, word_len={len(word)}, wrap_width={wrap_width}", indent=True)
        secho_color('\n', fg=color, nl=False)
        current_position = 0
        line_start = True
    
    # Add space before word if needed
    if current_position > 0:
        secho_color(' ', fg=color, nl=False)
        current_position += 1
    
    # Handle bold markers (**) and italic markers (*) - render both as bold
    display_word = word
    has_inline_bold = False
    
    # Handle ** bold markers
    if '**' in word:
        marker_count = word.count('**')
        if marker_count > 0:
            if word.startswith('**'):
                has_inline_bold = True
                in_bold = True
            elif in_bold and word.endswith('**'):
                has_inline_bold = True
                in_bold = False
            elif in_bold:
                has_inline_bold = True
        display_word = word.replace('**', '')
    # Handle * italic markers (render as bold in terminal)
    elif '*' in word and not word.startswith('**'):
        # Handle single * markers like ** markers but for italics
        marker_count = word.count('*')
        if marker_count > 0:
            if word.startswith('*'):
                has_inline_bold = True
                in_bold = True
            elif in_bold and (word.endswith('*') or (word_ends_with_colon and word.rstrip(':').endswith('*'))):
                has_inline_bold = True
                in_bold = False
            elif in_bold:
                has_inline_bold = True
        display_word = word.replace('*', '')
    elif in_bold:
        # We're in the middle of bold text
        has_inline_bold = True
    
    # Determine if word should be bold (from any source)
    original_word_is_bold = word_is_bold
    word_is_bold = word_is_bold or has_inline_bold
    
    
    # Print the word
    if config.get("debug", False) and word_is_bold:
        debug_print(f"Printing '{display_word}' with bold=True", indent=True)
    
    # Use secho_color for proper terminal output with bold support
    secho_color(display_word, fg=color, nl=False, bold=word_is_bold)
    
    current_position += len(display_word)
    line_start = False
    
    # Check for line breaks in word
    if '\n' in display_word:
        last_newline = display_word.rfind('\n')
        current_position = len(display_word) - last_newline - 1
        line_start = True
        in_list_item = False
        in_numbered_list = False
        in_header = False
    
    # Reset bold states AFTER printing a word that ends with colon in a list
    if word_ends_with_colon and (in_list_item or in_numbered_list):
        in_list_item = False
        in_numbered_list = False
        in_bold = False  # Reset markdown bold state too
        in_header = False
    
    return current_position, line_start, in_bold, in_numbered_list, in_list_item, in_header
