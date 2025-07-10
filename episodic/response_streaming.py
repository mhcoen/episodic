"""
Response streaming utilities for Episodic.

This module provides different streaming implementations for LLM responses,
including constant-rate and natural rhythm streaming.
"""

import time
import threading
import queue
import re
import math
import random
import textwrap
from typing import Generator, List, Optional

import typer
from episodic.color_utils import secho_color
from episodic.config import config
from episodic.configuration import get_llm_color
from episodic.text_formatting import get_wrap_width


def process_stream_response(stream_generator: Generator, model: str) -> Generator[str, None, None]:
    """Process streaming response from LLM."""
    # This is imported here to avoid circular imports
    from episodic.llm import process_stream_response as llm_process_stream
    return llm_process_stream(stream_generator, model)


class ResponseStreamer:
    """Handles different modes of response streaming."""
    
    def __init__(self):
        self.word_queue = queue.Queue()
        self.stop_event = threading.Event()
        
    def stream_response(
        self, 
        stream_generator: Generator,
        model: str,
        stream_rate: float = 15.0,
        use_constant_rate: bool = False,
        use_natural_rhythm: bool = False
    ) -> str:
        """
        Stream response with specified mode.
        
        Returns the complete response text.
        """
        full_response_parts = []
        
        if use_natural_rhythm and stream_rate > 0:
            self._natural_rhythm_stream(stream_generator, model, stream_rate, full_response_parts)
        elif use_constant_rate and stream_rate > 0:
            self._constant_rate_stream(stream_generator, model, stream_rate, full_response_parts)
        else:
            self._default_stream(stream_generator, model, full_response_parts)
            
        return ''.join(full_response_parts)
    
    def _natural_rhythm_stream(
        self,
        stream_generator: Generator,
        model: str,
        stream_rate: float,
        full_response_parts: List[str]
    ) -> None:
        """Stream with natural rhythm using sinusoidal variation."""
        # Start the printer thread
        printer_thread = threading.Thread(
            target=self._natural_rhythm_printer,
            args=(stream_rate,)
        )
        printer_thread.start()
        
        # Process chunks and split into words
        self._process_chunks_to_words(stream_generator, model, full_response_parts)
        
        # Signal completion and wait for printer to finish
        self.word_queue.put(None)  # Sentinel value
        self.stop_event.set()
        printer_thread.join()
    
    def _constant_rate_stream(
        self,
        stream_generator: Generator,
        model: str,
        stream_rate: float,
        full_response_parts: List[str]
    ) -> None:
        """Stream at a constant rate."""
        # Start the printer thread
        printer_thread = threading.Thread(
            target=self._constant_rate_printer,
            args=(stream_rate,)
        )
        printer_thread.start()
        
        # Process chunks and split into words
        self._process_chunks_to_words(stream_generator, model, full_response_parts)
        
        # Signal completion and wait for printer to finish
        self.word_queue.put(None)  # Sentinel value
        self.stop_event.set()
        printer_thread.join()
    
    def _default_stream(
        self,
        stream_generator: Generator,
        model: str,
        full_response_parts: List[str]
    ) -> None:
        """Default streaming with word wrap and bold support."""
        current_word = ""
        current_position = 0
        wrap_width = get_wrap_width() if config.get("text_wrap") else None
        in_bold = False
        bold_count = 0
        line_start = True
        in_numbered_list = False
        
        for chunk in process_stream_response(stream_generator, model):
            full_response_parts.append(chunk)
            
            for char in chunk:
                if char == '*':
                    bold_count += 1
                    if bold_count == 2:
                        # Toggle bold state
                        in_bold = not in_bold
                        bold_count = 0
                    continue
                elif bold_count == 1:
                    # Single asterisk, print it
                    current_word += '*'
                    bold_count = 0
                
                if char in ' \n':
                    # End of word
                    if current_word:
                        # Check if this is a numbered list item at the start of a line
                        word_is_bold = in_bold
                        if line_start and current_word.rstrip('.').isdigit():
                            in_numbered_list = True  # Start bolding for numbered list
                        
                        # Bold everything in numbered list until colon
                        if in_numbered_list:
                            word_is_bold = True
                        
                        # Check if this word ends with colon to stop bolding next words
                        if current_word.endswith(':') and in_numbered_list:
                            # This word should still be bold, but stop for next words
                            word_is_bold = True
                            # Will reset in_numbered_list after printing this word
                        
                        # Check wrap
                        if wrap_width and current_position + len(current_word) > wrap_width:
                            secho_color('\n', nl=False)
                            current_position = 0
                            line_start = True
                        
                        # Print word
                        llm_color = get_llm_color()
                        if isinstance(llm_color, str):
                            llm_color = llm_color.lower()
                        secho_color(current_word, fg=llm_color, nl=False, bold=word_is_bold)
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
                    else:
                        secho_color(' ', nl=False)
                        current_position += 1
                else:
                    current_word += char
        
        # Print any remaining word
        if current_word:
            llm_color = get_llm_color()
            if isinstance(llm_color, str):
                llm_color = llm_color.lower()
            secho_color(current_word, fg=llm_color, nl=False, bold=in_bold)
        
        # Ensure newline at end
        typer.echo("")
    
    def _process_chunks_to_words(
        self,
        stream_generator: Generator,
        model: str,
        full_response_parts: List[str]
    ) -> None:
        """Process streaming chunks and split into words."""
        accumulated_text = ""
        for chunk in process_stream_response(stream_generator, model):
            full_response_parts.append(chunk)
            accumulated_text += chunk
            
            # Split accumulated text into words while preserving whitespace
            # Use regex to split on word boundaries but keep the whitespace
            words = re.findall(r'[^ \t\r\n\f]+[ \t\r\f]*|\n', accumulated_text)
            
            # Add complete words to the queue
            for word in words[:-1]:  # Keep last potentially incomplete word
                self.word_queue.put(word)
            
            # Keep the last potentially incomplete word
            if words:
                accumulated_text = words[-1] if not words[-1].endswith((' ', '\n')) else ''
                if not accumulated_text:
                    self.word_queue.put(words[-1])
            else:
                accumulated_text = ''
        
        # Add any remaining text
        if accumulated_text:
            self.word_queue.put(accumulated_text)
    
    def _natural_rhythm_printer(self, words_per_second: float) -> None:
        """Print words with natural rhythm variation."""
        current_line = ""
        line_position = 0
        in_bold = False
        wrap_width = get_wrap_width() if config.get("text_wrap", True) else None
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
            ',': 0.15,
            ';': 0.2,
            ':': 0.2,
            ')': 0.1,
            '"': 0.1,
        }
        
        while not self.stop_event.is_set() or not self.word_queue.empty():
            try:
                word = self.word_queue.get(timeout=0.1)
                if word is None:  # Sentinel value
                    break
                
                # Calculate current phase in rhythm cycle
                elapsed = time.time() - start_time
                phase = (elapsed % period) / period
                
                # Calculate interval with sinusoidal variation
                rhythm_factor = math.sin(2 * math.pi * phase)
                interval = base_interval + amplitude * rhythm_factor
                
                # Add small random jitter (±10ms)
                jitter = random.uniform(-0.01, 0.01)
                interval += jitter
                
                # Ensure interval is positive
                interval = max(0.02, interval)
                
                # Add word to current line
                current_line += word
                
                # Process and print the word
                self._process_word_for_printing(
                    word, current_line, line_position, in_bold, wrap_width
                )
                
                # Check for punctuation at end of word and add delay
                if word.strip():
                    last_char = word.strip()[-1]
                    if last_char in punctuation_delays:
                        interval += punctuation_delays[last_char]
                
                # Sleep for calculated interval
                time.sleep(interval)
            except queue.Empty:
                continue
        
        # Print any remaining content
        self._print_remaining_content(current_line)
    
    def _constant_rate_printer(self, words_per_second: float) -> None:
        """Print words at a constant rate."""
        current_line = ""
        line_position = 0
        in_bold = False
        wrap_width = get_wrap_width() if config.get("text_wrap", True) else None
        delay = 1.0 / words_per_second
        
        while not self.stop_event.is_set() or not self.word_queue.empty():
            try:
                word = self.word_queue.get(timeout=0.1)
                if word is None:  # Sentinel value
                    break
                
                # Add word to current line
                current_line += word
                
                # Process and print the word
                self._process_word_for_printing(
                    word, current_line, line_position, in_bold, wrap_width
                )
                
                # Sleep for constant rate
                time.sleep(delay)
            except queue.Empty:
                continue
        
        # Print any remaining content
        self._print_remaining_content(current_line)
    
    def _process_word_for_printing(
        self,
        word: str,
        current_line: str,
        line_position: int,
        in_bold: bool,
        wrap_width: Optional[int]
    ) -> None:
        """Process a word for printing with formatting."""
        # Check if we have complete lines to print
        if '\n' in current_line:
            lines = current_line.split('\n')
            # Print all complete lines
            for line in lines[:-1]:
                if config.get("text_wrap") and wrap_width and len(line) > wrap_width:
                    wrapped = textwrap.fill(
                        line,
                        width=wrap_width,
                        initial_indent="",
                        subsequent_indent=""
                    )
                    llm_color = get_llm_color()
                    if isinstance(llm_color, str):
                        llm_color = llm_color.lower()
                    secho_color(wrapped, fg=llm_color, bold=False)
                else:
                    llm_color = get_llm_color()
                    if isinstance(llm_color, str):
                        llm_color = llm_color.lower()
                    secho_color(line, fg=llm_color, bold=False)
            # Keep the incomplete last line
            current_line = lines[-1]
            line_position = 0  # Reset position after newline
            in_bold = False  # Reset bold state after newline
        else:
            # Process word for bold markers and print with wrapping
            display_word = word
            
            # Check for bold markers
            should_be_bold = in_bold  # Current bold state for this word
            
            if display_word.startswith('**'):
                should_be_bold = True
                in_bold = True
                display_word = display_word[2:]
            
            if display_word.endswith('**'):
                display_word = display_word[:-2]
                in_bold = False  # Turn off bold after this word
            elif should_be_bold and (display_word.endswith(':') or display_word.endswith('-')):
                # Stop bold after colon or dash in lists
                in_bold = False
            
            # Strip any remaining ** in the middle (shouldn't happen but just in case)
            display_word = display_word.replace('**', '')
            
            # Check if we need to wrap
            if wrap_width and line_position > 0 and line_position + len(display_word) > wrap_width:
                typer.echo()  # New line
                line_position = 0
            
            # Print the word
            llm_color = get_llm_color()
            if isinstance(llm_color, str):
                llm_color = llm_color.lower()
            secho_color(display_word, fg=llm_color, bold=should_be_bold, nl=False)
            line_position += len(display_word)
            
            # Don't accumulate printed words
            current_line = ""
    
    def _print_remaining_content(self, current_line: str) -> None:
        """Print any remaining content at the end of streaming."""
        if current_line.strip():
            llm_color = get_llm_color()
            if isinstance(llm_color, str):
                llm_color = llm_color.lower()
            secho_color(current_line, fg=llm_color)
        else:
            # Ensure we have a newline at the end
            typer.echo("")