"""
Unified response streaming for Episodic.

This module provides a single, configurable streaming implementation
that handles all streaming modes (immediate, constant rate, natural rhythm).
"""

import time
import threading
import queue
import re
import math
import random
from typing import Generator, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import typer
from episodic.color_utils import secho_color
from episodic.config import config
from episodic.configuration import get_llm_color
from episodic.text_formatting import get_wrap_width


class StreamingMode(Enum):
    """Streaming rate modes."""
    IMMEDIATE = "immediate"
    CONSTANT = "constant"
    NATURAL = "natural"


@dataclass
class StreamConfig:
    """Configuration for streaming behavior."""
    mode: StreamingMode = StreamingMode.IMMEDIATE
    words_per_second: float = 15.0
    enable_wrapping: bool = True
    enable_bold: bool = True
    natural_variation: float = 0.3  # 30% variation for natural mode
    natural_period: float = 4.0  # 4-second cycle for natural rhythm
    punctuation_delays: dict = None
    
    def __post_init__(self):
        if self.punctuation_delays is None:
            self.punctuation_delays = {
                '.': 0.3,
                '!': 0.3,
                '?': 0.3,
                ',': 0.15,
                ';': 0.2,
                ':': 0.2,
                ')': 0.1,
                '"': 0.1,
            }


class UnifiedResponseStreamer:
    """Unified streaming handler with configurable behavior."""
    
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
        Stream response with configured behavior.
        
        This is the main entry point that maintains backward compatibility.
        """
        # Determine mode from parameters
        if use_natural_rhythm and stream_rate > 0:
            mode = StreamingMode.NATURAL
        elif use_constant_rate and stream_rate > 0:
            mode = StreamingMode.CONSTANT
        else:
            mode = StreamingMode.IMMEDIATE
        
        # Create config
        stream_config = StreamConfig(
            mode=mode,
            words_per_second=stream_rate,
            enable_wrapping=config.get("text_wrap", True),
            enable_bold=True
        )
        
        # Process the stream
        return self._unified_stream(stream_generator, model, stream_config)
    
    def _unified_stream(
        self,
        stream_generator: Generator,
        model: str,
        stream_config: StreamConfig
    ) -> str:
        """Unified streaming implementation."""
        full_response_parts = []
        
        # Import here to avoid circular dependency
        from episodic.llm import process_stream_response as llm_process_stream
        
        
        if stream_config.mode == StreamingMode.IMMEDIATE:
            # Direct streaming without delay
            self._process_immediate(
                llm_process_stream(stream_generator, model),
                full_response_parts,
                stream_config
            )
        else:
            # Queued streaming with timing control
            self._process_queued(
                llm_process_stream(stream_generator, model),
                full_response_parts,
                stream_config
            )
        
        return ''.join(full_response_parts)
    
    def _process_immediate(
        self,
        chunk_generator: Generator,
        full_response_parts: List[str],
        stream_config: StreamConfig
    ) -> None:
        """Process chunks immediately without delay."""
        processor = WordProcessor(stream_config)
        
        for chunk in chunk_generator:
            full_response_parts.append(chunk)
            processor.process_chunk(chunk)
        
        # Handle any remaining content
        processor.finish()
    
    def _process_queued(
        self,
        chunk_generator: Generator,
        full_response_parts: List[str],
        stream_config: StreamConfig
    ) -> None:
        """Process chunks through a queue with timing control."""
        # Reset queue and event
        self.word_queue = queue.Queue()
        self.stop_event.clear()
        
        # Start printer thread
        printer_thread = threading.Thread(
            target=self._queued_printer,
            args=(stream_config,)
        )
        printer_thread.start()
        
        # Process chunks into words
        accumulated_text = ""
        for chunk in chunk_generator:
            full_response_parts.append(chunk)
            accumulated_text += chunk
            
            # Split into words while preserving whitespace
            words = re.findall(r'[^ \t\r\n\f]+[ \t\r\f]*|\n', accumulated_text)
            
            # Queue complete words
            for word in words[:-1]:
                self.word_queue.put(word)
            
            # Keep last potentially incomplete word
            if words:
                if words[-1].endswith((' ', '\n')):
                    # Last word is complete, queue it
                    self.word_queue.put(words[-1])
                    accumulated_text = ''
                else:
                    # Last word might be incomplete, keep it
                    accumulated_text = words[-1]
            else:
                accumulated_text = ''
        
        # Queue any remaining text
        if accumulated_text:
            self.word_queue.put(accumulated_text)
        
        # Signal completion
        self.word_queue.put(None)
        self.stop_event.set()
        printer_thread.join()
    
    def _queued_printer(self, stream_config: StreamConfig) -> None:
        """Print words from queue with configured timing."""
        processor = WordProcessor(stream_config)
        timing_calculator = TimingCalculator(stream_config)
        
        while not self.stop_event.is_set() or not self.word_queue.empty():
            try:
                word = self.word_queue.get(timeout=0.1)
                if word is None:  # Sentinel
                    break
                
                # Process and print word
                processor.process_word(word)
                
                # Calculate and apply delay
                delay = timing_calculator.get_delay(word)
                if delay > 0:
                    time.sleep(delay)
                    
            except queue.Empty:
                continue
        
        # Ensure final newline
        typer.echo("")


class WordProcessor:
    """Handles word processing and printing logic."""
    
    def __init__(self, stream_config: StreamConfig):
        self.config = stream_config
        self.current_word = ""
        self.current_position = 0
        self.line_start = True
        self.in_bold = False
        self.bold_count = 0
        self.in_numbered_list = False
        self.wrap_width = get_wrap_width() if stream_config.enable_wrapping else None
    
    def process_chunk(self, chunk: str) -> None:
        """Process a chunk character by character."""
        for char in chunk:
            if char == '*' and self.config.enable_bold:
                self.bold_count += 1
                if self.bold_count == 2:
                    self.in_bold = not self.in_bold
                    self.bold_count = 0
                continue
            elif self.bold_count == 1:
                self.current_word += '*'
                self.bold_count = 0
            
            if char in ' \n':
                # End of word
                if self.current_word:
                    self._print_word()
                    self.current_word = ""
                
                # Handle space/newline
                if char == '\n':
                    secho_color('\n', nl=False)
                    self.current_position = 0
                    self.line_start = True
                    self.in_numbered_list = False
                else:
                    secho_color(' ', nl=False)
                    self.current_position += 1
                    if self.current_position > 0:
                        self.line_start = False
            else:
                self.current_word += char
    
    def process_word(self, word: str) -> None:
        """Process a complete word (for queued mode)."""
        if '\n' in word:
            # Handle words with newlines
            parts = word.split('\n')
            for i, part in enumerate(parts):
                if i > 0:
                    secho_color('\n', nl=False)
                    self.current_position = 0
                    self.line_start = True
                    self.in_numbered_list = False
                if part:
                    self.current_word = part.rstrip()
                    self._print_word()
                    # Print trailing spaces if any
                    trailing_spaces = len(part) - len(part.rstrip())
                    if trailing_spaces > 0:
                        secho_color(' ' * trailing_spaces, nl=False)
                        self.current_position += trailing_spaces
        else:
            # Regular word
            self.current_word = word.rstrip()
            self._print_word()
            # Print trailing spaces if any
            trailing_spaces = len(word) - len(word.rstrip())
            if trailing_spaces > 0:
                secho_color(' ' * trailing_spaces, nl=False)
                self.current_position += trailing_spaces
    
    def _print_word(self) -> None:
        """Print the current word with appropriate formatting."""
        if not self.current_word:
            return
        
        
        # Determine if word should be bold
        word_is_bold = self.in_bold
        
        # Check for numbered list at line start
        if self.line_start and self.current_word.rstrip('.').isdigit():
            self.in_numbered_list = True
        
        # Bold numbered list items until colon
        if self.in_numbered_list:
            word_is_bold = True
            if self.current_word.endswith(':'):
                self.in_numbered_list = False
        
        # Check for line wrap
        if self.wrap_width and self.current_position > 0:
            if self.current_position + len(self.current_word) + 1 > self.wrap_width:
                secho_color('\n', nl=False)
                self.current_position = 0
                self.line_start = True
        
        # Print word
        llm_color = get_llm_color()
        if isinstance(llm_color, str):
            llm_color = llm_color.lower()
        secho_color(self.current_word, fg=llm_color, nl=False, bold=word_is_bold)
        self.current_position += len(self.current_word)
        
        if not self.current_word.isspace():
            self.line_start = False
    
    def finish(self) -> None:
        """Handle any remaining content and ensure proper ending."""
        if self.current_word:
            self._print_word()
            self.current_word = ""
        
        # Ensure newline at end
        typer.echo("")


class TimingCalculator:
    """Calculates timing delays based on streaming mode."""
    
    def __init__(self, stream_config: StreamConfig):
        self.config = stream_config
        self.start_time = time.time()
        self.base_interval = 1.0 / stream_config.words_per_second
    
    def get_delay(self, word: str) -> float:
        """Calculate delay for the given word."""
        if self.config.mode == StreamingMode.CONSTANT:
            delay = self.base_interval
        elif self.config.mode == StreamingMode.NATURAL:
            delay = self._natural_delay()
        else:
            delay = 0
        
        # Add punctuation delays if applicable
        if word.strip() and self.config.punctuation_delays:
            last_char = word.strip()[-1]
            if last_char in self.config.punctuation_delays:
                delay += self.config.punctuation_delays[last_char]
        
        return max(0.02, delay)  # Minimum 20ms
    
    def _natural_delay(self) -> float:
        """Calculate natural rhythm delay with sinusoidal variation."""
        elapsed = time.time() - self.start_time
        phase = (elapsed % self.config.natural_period) / self.config.natural_period
        
        # Sinusoidal variation
        rhythm_factor = math.sin(2 * math.pi * phase)
        interval = self.base_interval + (self.base_interval * self.config.natural_variation * rhythm_factor)
        
        # Add small random jitter
        jitter = random.uniform(-0.01, 0.01)
        
        return interval + jitter


# For backward compatibility, keep the old class name
ResponseStreamer = UnifiedResponseStreamer