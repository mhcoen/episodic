"""
Conversation management functionality for Episodic.

This module handles all conversation-related operations including:
- Managing chat messages and responses
- Building conversation context
- Semantic drift detection
- Session cost tracking
- Text formatting and wrapping
"""

import os
import shutil
import textwrap
import logging
import time
import threading
import queue
import re
from typing import Optional, List, Dict, Any, Tuple

import typer

from episodic.db import (
    insert_node, get_node, get_ancestry, get_head, set_head,
    get_recent_nodes, get_recent_topics, update_topic_end_node,
    store_topic, update_topic_name
)
from episodic.llm import query_with_context
from episodic.llm_config import get_current_provider
from episodic.configuration import get_model_context_limit
from episodic.config import config
from episodic.configuration import (
    get_llm_color, get_system_color, DEFAULT_CONTEXT_DEPTH,
    COST_PRECISION
)
from episodic.ml import ConversationalDrift
from episodic.compression import queue_topic_for_compression
from episodic.topics import (
    detect_topic_change_separately, extract_topic_ollama, 
    should_create_first_topic, build_conversation_segment,
    _display_topic_evolution
)
from episodic.benchmark import benchmark_operation, benchmark_resource, display_pending_benchmark

# Set up logging
logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation flow, responses, and related state."""
    
    def __init__(self):
        """Initialize the ConversationManager."""
        self.current_node_id = None
        self.current_topic = None  # Track current topic (name, start_node_id)
        self.drift_calculator = None
        self.session_costs = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,  
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }
    
    def get_session_costs(self) -> Dict[str, Any]:
        """Get the current session costs."""
        return self.session_costs.copy()
    
    def reset_session_costs(self) -> None:
        """Reset session costs to zero."""
        self.session_costs = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }
    
    def get_current_node_id(self) -> Optional[str]:
        """Get the current node ID."""
        return self.current_node_id
    
    def set_current_node_id(self, node_id: str) -> None:
        """Set the current node ID."""
        self.current_node_id = node_id
    
    def set_current_topic(self, topic_name: str, start_node_id: str) -> None:
        """Set the current topic."""
        self.current_topic = (topic_name, start_node_id)
    
    def get_current_topic(self) -> Optional[Tuple[str, str]]:
        """Get the current topic (name, start_node_id) or None."""
        return self.current_topic
    
    def initialize_conversation(self) -> None:
        """Initialize the conversation state from the database."""
        self.current_node_id = get_head()
        
        # Initialize current topic from database
        if self.current_node_id:
            # Find the topic that contains the current head node
            recent_topics = get_recent_topics(limit=10)
            for topic in recent_topics:
                # Check if current node is within this topic's range
                # For now, just use the most recent topic
                if topic.get('end_node_id'):
                    self.set_current_topic(topic['name'], topic['start_node_id'])
                    break
    
    def get_wrap_width(self) -> int:
        """Get the appropriate text wrapping width for the terminal."""
        terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        margin = 4
        max_width = 100  # Maximum line length for readability
        # Use terminal width or 100, whichever is smaller (with minimum of 40)
        return min(max_width, max(40, terminal_width - margin))
    
    def wrapped_text_print(self, text: str, **typer_kwargs) -> None:
        """Print text with automatic wrapping while preserving formatting."""
        # Check if wrapping is enabled
        if not config.get("text_wrap", True):
            typer.secho(str(text), **typer_kwargs)
            return
        
        wrap_width = self.get_wrap_width()
        
        # Process text to preserve formatting while wrapping long lines
        lines = str(text).split('\n')
        wrapped_lines = []
        
        for line in lines:
            if len(line) <= wrap_width:
                # Line is short enough, keep as-is
                wrapped_lines.append(line)
            else:
                # Line is too long, wrap it while preserving indentation
                # Detect indentation (spaces or tabs at start of line)
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]
                
                # Wrap the content while preserving indentation
                if stripped:  # Only wrap if there's actual content
                    wrapped = textwrap.fill(
                        stripped, 
                        width=wrap_width,
                        initial_indent=indent,
                        subsequent_indent=indent  # Same indent as first line
                    )
                    wrapped_lines.append(wrapped)
                else:
                    # Empty line - preserve as-is
                    wrapped_lines.append(line)
        
        # Join the processed lines back together
        wrapped_text = '\n'.join(wrapped_lines)
        
        # Print with the specified formatting
        typer.secho(wrapped_text, **typer_kwargs)
    
    def wrapped_llm_print(self, text: str, **typer_kwargs) -> None:
        """Print LLM text with automatic wrapping while preserving formatting."""
        # First handle bold markers
        import re
        
        # Split text by bold markers
        parts = re.split(r'(\*\*[^*]+\*\*)', text)
        
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                # This is bold text
                bold_text = part[2:-2]
                self.wrapped_text_print(bold_text, bold=True, **typer_kwargs)
            else:
                # Regular text
                self.wrapped_text_print(part, **typer_kwargs)
    
    def get_drift_calculator(self) -> Optional[ConversationalDrift]:
        """Get or create the drift calculator instance."""
        # Check if drift detection is disabled in config
        if not config.get("show_drift", True):
            return None
            
        if self.drift_calculator is None:
            try:
                self.drift_calculator = ConversationalDrift()
            except Exception as e:
                # If drift calculator fails to initialize (e.g., missing dependencies),
                # disable drift detection for this session
                if config.get("debug", False):
                    typer.echo(f"⚠️  Drift detection disabled: {e}")
                self.drift_calculator = False  # Mark as disabled
        return self.drift_calculator if self.drift_calculator is not False else None
    
    def build_semdepth_context(
        self, 
        nodes: List[Dict[str, Any]], 
        semdepth: int, 
        text_field: str = "content"
    ) -> str:
        """
        Build combined context from the last N nodes for semantic analysis.
        
        Args:
            nodes: List of conversation nodes in chronological order
            semdepth: Number of most recent nodes to combine
            text_field: Field containing text content
            
        Returns:
            Combined text from the last semdepth nodes
        """
        if not nodes or semdepth < 1:
            return ""
        
        # Get the last semdepth nodes
        context_nodes = nodes[-semdepth:] if len(nodes) >= semdepth else nodes
        
        # Combine their content
        combined_text = []
        for node in context_nodes:
            content = node.get(text_field, "").strip()
            if content:
                combined_text.append(content)
        
        return "\n".join(combined_text)
    
    def display_semantic_drift(self, current_user_node_id: str) -> None:
        """
        Calculate and display semantic drift between consecutive user messages.
        
        Only compares user inputs to detect when the user changes topics,
        ignoring assistant responses which just follow the user's lead.
        
        Args:
            current_user_node_id: ID of the current user message node
        """
        calc = self.get_drift_calculator()
        if not calc:
            return  # Drift detection disabled
        
        try:
            # Get conversation history from root to current node
            conversation_chain = get_ancestry(current_user_node_id)
            
            # Filter to user messages only
            user_messages = [node for node in conversation_chain 
                            if node.get("role") == "user" and node.get("content", "").strip()]
            
            # Need at least 2 user messages for comparison
            if len(user_messages) < 2:
                if config.get("debug", False):
                    typer.echo(f"   (Need 2 user messages for drift, have {len(user_messages)})")
                return
            
            # Compare current user message to previous user message
            current_user = user_messages[-1]
            previous_user = user_messages[-2]
            
            # Calculate semantic drift between consecutive user inputs
            drift_score = calc.calculate_drift(previous_user, current_user, text_field="content")
            
            # Format drift display based on score level
            if drift_score >= 0.8:
                drift_emoji = "🔄"
                drift_desc = "High topic shift"
            elif drift_score >= 0.6:
                drift_emoji = "📈"
                drift_desc = "Moderate drift"
            elif drift_score >= 0.3:
                drift_emoji = "➡️"
                drift_desc = "Low drift"
            else:
                drift_emoji = "🎯"
                drift_desc = "Minimal drift"
            
            # Display drift information
            prev_short_id = previous_user.get("short_id", "??")
            typer.secho(f"\n{drift_emoji} Semantic drift: {drift_score:.3f} ({drift_desc}) from user message {prev_short_id}", fg=get_system_color())
            
            # Show additional context if debug mode is enabled
            if config.get("debug", False):
                prev_content = previous_user.get("content", "")[:80]
                curr_content = current_user.get("content", "")[:80]
                typer.echo(f"   Previous: {prev_content}{'...' if len(previous_user.get('content', '')) > 80 else ''}")
                typer.echo(f"   Current:  {curr_content}{'...' if len(current_user.get('content', '')) > 80 else ''}")
                
                # Show embedding cache efficiency
                cache_size = calc.get_cache_size()
                typer.echo(f"   Embedding cache: {cache_size} entries")
            
        except Exception as e:
            # If drift calculation fails, silently continue (don't disrupt conversation flow)
            if config.get("debug", False):
                typer.echo(f"⚠️  Drift calculation error: {e}")
    
    def handle_chat_message(
        self, 
        user_input: str,
        model: str,
        system_message: str,
        context_depth: int = DEFAULT_CONTEXT_DEPTH
    ) -> Tuple[str, str]:
        """
        Handle a chat message (non-command input).
        
        Args:
            user_input: The user's chat message
            model: The LLM model to use
            system_message: The system prompt
            context_depth: Number of messages to include in context
            
        Returns:
            Tuple of (assistant_node_id, display_response)
            
        Processes the message through the LLM and updates the conversation DAG.
        """
        with benchmark_operation("Message Processing"):
            # Get recent messages for context BEFORE adding the new message
            with benchmark_resource("Database", "get recent nodes"):
                recent_nodes = get_recent_nodes(limit=10)  # Get last 10 nodes for context
            
            # Add the user message to the database
            with benchmark_resource("Database", "insert user node"):
                user_node_id, user_short_id = insert_node(user_input, self.current_node_id, role="user")
        
            # Detect topic change BEFORE querying the main LLM
            topic_changed = False
            new_topic_name = None
            topic_cost_info = None
            
            if config.get("debug", False):
                typer.echo(f"\n🔍 DEBUG: Topic detection check")
                typer.echo(f"   Recent nodes count: {len(recent_nodes) if recent_nodes else 0}")
                typer.echo(f"   Current topic: {self.current_topic}")
                typer.echo(f"   Min messages before topic change: {config.get('min_messages_before_topic_change', 4)}")
            
            if recent_nodes and len(recent_nodes) >= 2:  # Need at least some history
                try:
                    with benchmark_operation("Topic Detection"):
                        topic_changed, new_topic_name, topic_cost_info = detect_topic_change_separately(recent_nodes, user_input)
                        if config.get("debug", False):
                            typer.echo(f"   Topic change detected: {topic_changed}")
                            if topic_changed:
                                typer.echo(f"   New topic: {new_topic_name}")
                except Exception as e:
                    if config.get("debug", False):
                        typer.echo(f"   ❌ Topic detection error: {e}")
                    # Continue without topic detection on error
                    topic_changed = False
            else:
                if config.get("debug", False):
                    typer.echo("   ⚠️  Not enough history for topic detection")
            
            # Add topic detection costs to session
            if topic_cost_info:
                self.session_costs["total_input_tokens"] += topic_cost_info.get("input_tokens", 0)
                self.session_costs["total_output_tokens"] += topic_cost_info.get("output_tokens", 0)
                self.session_costs["total_tokens"] += topic_cost_info.get("total_tokens", 0)
                self.session_costs["total_cost_usd"] += topic_cost_info.get("cost_usd", 0.0)
            
            # Store debug info to display later
            debug_topic_info = None
            if config.get("debug", False) and topic_changed:
                debug_topic_info = (new_topic_name, topic_cost_info)

            # Query the LLM with context
            try:
                # Check if streaming is enabled (default to True)
                use_streaming = config.get("stream_responses", True)
                
                # Query with context
                with benchmark_resource("LLM Call", model):
                    if use_streaming:
                        # Get streaming response
                        stream_generator, _ = query_with_context(
                            user_node_id, 
                            model=model,
                            system_message=system_message,
                            context_depth=context_depth,
                            stream=True
                        )
                        
                        # Calculate and display semantic drift if enabled (before streaming)
                        if config.get("show_drift", True):
                            self.display_semantic_drift(user_node_id)
                        
                        # Display debug topic info if it was stored
                        if debug_topic_info:
                            new_topic_name, topic_cost_info = debug_topic_info
                            typer.echo(f"\n🔍 DEBUG: Topic change detected")
                            typer.echo(f"   New topic: {new_topic_name}")
                            if topic_cost_info:
                                typer.echo(f"   Detection cost: ${topic_cost_info.get('cost_usd', 0.0):.{COST_PRECISION}f}")
                        
                        # Display blank line before response
                        typer.echo("")
                        
                        # Stream the response with proper formatting
                        typer.secho("🤖 ", fg=get_llm_color(), nl=False)  # Robot emoji without newline
                        
                        # Process the stream and display it
                        from episodic.llm import process_stream_response
                        full_response_parts = []
                        
                        # Get streaming rate configuration
                        stream_rate = config.get("stream_rate", 15)  # Default to 15 words per second
                        use_constant_rate = config.get("stream_constant_rate", False)
                        use_natural_rhythm = config.get("stream_natural_rhythm", False)
                        use_char_streaming = config.get("stream_char_mode", False)
                        
                        if config.get("debug", False):
                            typer.echo(f"DEBUG: Streaming modes - char: {use_char_streaming}, natural: {use_natural_rhythm}, constant: {use_constant_rate}")
                        
                        if False:  # Disabled character streaming
                            pass
                            
                        elif use_natural_rhythm and stream_rate > 0:
                            # Use natural rhythm streaming with sinusoidal variation
                            word_queue = queue.Queue()
                            stop_event = threading.Event()
                            
                            # Function to print words with natural rhythm
                            def natural_rhythm_printer():
                                import math
                                import random
                                
                                current_line = ""
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
                                    ',': 0.15,
                                    ';': 0.2,
                                    ':': 0.2,
                                    ')': 0.1,
                                    '"': 0.1,
                                }
                                
                                while not stop_event.is_set() or not word_queue.empty():
                                    try:
                                        word = word_queue.get(timeout=0.1)
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
                                        
                                        # Check if we have complete lines to print
                                        if '\n' in current_line:
                                            lines = current_line.split('\n')
                                            # Print all complete lines
                                            for line in lines[:-1]:
                                                if config.get("text_wrap", True):
                                                    wrap_width = self.get_wrap_width()
                                                    if len(line) > wrap_width:
                                                        wrapped = textwrap.fill(
                                                            line,
                                                            width=wrap_width,
                                                            initial_indent="",
                                                            subsequent_indent=""
                                                        )
                                                        typer.secho(wrapped, fg=get_llm_color())
                                                    else:
                                                        typer.secho(line, fg=get_llm_color())
                                                else:
                                                    typer.secho(line, fg=get_llm_color())
                                            # Keep the incomplete last line
                                            current_line = lines[-1]
                                        else:
                                            # Print the word without newline
                                            typer.secho(word, fg=get_llm_color(), nl=False)
                                            # Don't accumulate printed words
                                            current_line = ""
                                        
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
                                if current_line.strip():
                                    typer.secho(current_line, fg=get_llm_color())
                                else:
                                    # Ensure we have a newline at the end
                                    typer.echo("")
                            
                            # Start the printer thread
                            printer_thread = threading.Thread(target=natural_rhythm_printer)
                            printer_thread.start()
                            
                            # Process chunks and split into words
                            accumulated_text = ""
                            for chunk in process_stream_response(stream_generator, model):
                                full_response_parts.append(chunk)
                                accumulated_text += chunk
                                
                                # Split accumulated text into words while preserving whitespace
                                # Use regex to split on word boundaries but keep the whitespace
                                words = re.findall(r'\S+\s*|\n', accumulated_text)
                                
                                # Add complete words to the queue
                                for word in words[:-1]:  # Keep last potentially incomplete word
                                    word_queue.put(word)
                                
                                # Keep the last potentially incomplete word
                                if words:
                                    accumulated_text = words[-1] if not words[-1].endswith((' ', '\n')) else ''
                                    if not accumulated_text:
                                        word_queue.put(words[-1])
                                else:
                                    accumulated_text = ''
                            
                            # Add any remaining text
                            if accumulated_text:
                                word_queue.put(accumulated_text)
                            
                            # Signal completion and wait for printer to finish
                            word_queue.put(None)  # Sentinel value
                            stop_event.set()
                            printer_thread.join()
                            
                        elif use_constant_rate and stream_rate > 0:
                            # Use constant-rate streaming with word queue
                            word_queue = queue.Queue()
                            stop_event = threading.Event()
                            
                            # Function to print words at a constant rate
                            def constant_rate_printer():
                                current_line = ""
                                words_per_second = stream_rate
                                delay = 1.0 / words_per_second
                                
                                while not stop_event.is_set() or not word_queue.empty():
                                    try:
                                        word = word_queue.get(timeout=0.1)
                                        if word is None:  # Sentinel value
                                            break
                                        
                                        # Add word to current line
                                        current_line += word
                                        
                                        # Check if we have complete lines to print
                                        if '\n' in current_line:
                                            lines = current_line.split('\n')
                                            # Print all complete lines
                                            for line in lines[:-1]:
                                                if config.get("text_wrap", True):
                                                    wrap_width = self.get_wrap_width()
                                                    if len(line) > wrap_width:
                                                        wrapped = textwrap.fill(
                                                            line,
                                                            width=wrap_width,
                                                            initial_indent="",
                                                            subsequent_indent=""
                                                        )
                                                        typer.secho(wrapped, fg=get_llm_color())
                                                    else:
                                                        typer.secho(line, fg=get_llm_color())
                                                else:
                                                    typer.secho(line, fg=get_llm_color())
                                            # Keep the incomplete last line
                                            current_line = lines[-1]
                                        else:
                                            # Print the word without newline
                                            typer.secho(word, fg=get_llm_color(), nl=False)
                                            # Don't accumulate printed words
                                            current_line = ""
                                        
                                        # Sleep for constant rate
                                        time.sleep(delay)
                                    except queue.Empty:
                                        continue
                                
                                # Print any remaining content
                                if current_line.strip():
                                    typer.secho(current_line, fg=get_llm_color())
                                else:
                                    # Ensure we have a newline at the end
                                    typer.echo("")
                            
                            # Start the printer thread
                            printer_thread = threading.Thread(target=constant_rate_printer)
                            printer_thread.start()
                            
                            # Process chunks and split into words
                            accumulated_text = ""
                            for chunk in process_stream_response(stream_generator, model):
                                full_response_parts.append(chunk)
                                accumulated_text += chunk
                                
                                # Split accumulated text into words while preserving whitespace
                                # Use regex to split on word boundaries but keep the whitespace
                                words = re.findall(r'\S+\s*|\n', accumulated_text)
                                
                                # Add complete words to the queue
                                for word in words[:-1]:  # Keep last potentially incomplete word
                                    word_queue.put(word)
                                
                                # Keep the last potentially incomplete word
                                if words:
                                    accumulated_text = words[-1] if not words[-1].endswith((' ', '\n')) else ''
                                    if not accumulated_text:
                                        word_queue.put(words[-1])
                                else:
                                    accumulated_text = ''
                            
                            # Add any remaining text
                            if accumulated_text:
                                word_queue.put(accumulated_text)
                            
                            # Signal completion and wait for printer to finish
                            word_queue.put(None)  # Sentinel value
                            stop_event.set()
                            printer_thread.join()
                            
                        else:
                            # Default streaming with word wrap and bold support
                            current_word = ""
                            current_position = 0
                            wrap_width = self.get_wrap_width() if config.get("text_wrap", True) else None
                            in_bold = False
                            bold_count = 0
                            
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
                                            # Check wrap
                                            if wrap_width and current_position + len(current_word) > wrap_width:
                                                typer.secho('\n', nl=False)
                                                current_position = 0
                                            
                                            # Print word
                                            typer.secho(current_word, fg=get_llm_color(), nl=False, bold=in_bold)
                                            current_position += len(current_word)
                                            current_word = ""
                                        
                                        # Print space or newline
                                        if char == '\n':
                                            typer.secho('\n', nl=False)
                                            current_position = 0
                                        else:
                                            typer.secho(' ', fg=get_llm_color(), nl=False)
                                            current_position += 1
                                    else:
                                        # Accumulate character
                                        current_word += char
                            
                            # Print remaining word
                            if current_word:
                                if wrap_width and current_position + len(current_word) > wrap_width:
                                    typer.secho('\n', nl=False)
                                typer.secho(current_word, fg=get_llm_color(), nl=False, bold=in_bold)
                        
                        # Get the full response
                        display_response = ''.join(full_response_parts)
                        
                        # Add newline after streaming
                        typer.echo("")
                        
                        # Add blank line after response (as requested by user)
                        typer.echo("")
                        
                        # For streaming, estimate token counts from the response we have
                        # This is approximate but avoids making a duplicate API call
                        from litellm import token_counter
                        
                        # Count tokens in the response
                        output_tokens = token_counter(model=model, text=display_response)
                        
                        # Estimate input tokens from context (rough approximation)
                        # We'd need to reconstruct the full prompt to get exact count
                        input_tokens = 100  # Rough estimate, could be improved
                        
                        cost_info = {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'cost_usd': 0.0  # Would need pricing info to calculate
                        }
                        
                        # Display cost info after streaming if enabled (currently not available for streaming)
                        if config.get("show_cost", False) and cost_info:
                            # Calculate context usage
                            current_tokens = cost_info.get('input_tokens', 0)
                            context_limit = get_model_context_limit(model)
                            context_percentage = (current_tokens / context_limit) * 100
                            
                            # Format context percentage with appropriate precision
                            if context_percentage < 1.0:
                                context_display = f"{context_percentage:.1f}%"
                            else:
                                context_display = f"{int(context_percentage)}%"
                            
                            cost_msg = f"Tokens: {cost_info.get('total_tokens', 0)} | Cost: ${cost_info.get('cost_usd', 0.0):.{COST_PRECISION}f} USD | Context: {context_display} full"
                            typer.secho(cost_msg, fg=get_system_color())
                        
                    else:
                        # Non-streaming response
                        response, cost_info = query_with_context(
                            user_node_id, 
                            model=model,
                            system_message=system_message,
                            context_depth=context_depth,
                            stream=False
                        )
                        display_response = response
                        
                        # Calculate and display semantic drift if enabled
                        if config.get("show_drift", True):
                            self.display_semantic_drift(user_node_id)
                        
                        # Display debug topic info if it was stored
                        if debug_topic_info:
                            new_topic_name, topic_cost_info = debug_topic_info
                            typer.echo(f"\n🔍 DEBUG: Topic change detected")
                            typer.echo(f"   New topic: {new_topic_name}")
                            if topic_cost_info:
                                typer.echo(f"   Detection cost: ${topic_cost_info.get('cost_usd', 0.0):.{COST_PRECISION}f}")
                        
                        # Collect all status messages to display in one block
                        status_messages = []
                        

                        # Add cost information if enabled
                        if config.get("show_cost", False) and cost_info:
                            # Calculate context usage
                            current_tokens = cost_info.get('input_tokens', 0)  # Use input tokens for context calculation
                            context_limit = get_model_context_limit(model)
                            context_percentage = (current_tokens / context_limit) * 100
                            
                            # Format context percentage with appropriate precision
                            if context_percentage < 1.0:
                                context_display = f"{context_percentage:.1f}%"
                            else:
                                context_display = f"{int(context_percentage)}%"
                            
                            status_messages.append(f"Tokens: {cost_info.get('total_tokens', 0)} | Cost: ${cost_info.get('cost_usd', 0.0):.{COST_PRECISION}f} USD | Context: {context_display} full")

                        # Display the response block with proper spacing
                        if status_messages:
                            # Show blank line, then status messages, then LLM response
                            typer.echo("")
                            for msg in status_messages:
                                typer.secho(msg, fg=get_system_color())
                            typer.secho("🤖 ", fg=get_llm_color(), nl=False)
                            self.wrapped_llm_print(display_response, fg=get_llm_color())
                        else:
                            # No status messages, just show blank line then LLM response
                            typer.echo("")  # Blank line
                            typer.secho("🤖 ", fg=get_llm_color(), nl=False)
                            self.wrapped_llm_print(display_response, fg=get_llm_color())

                # Update session costs
                if cost_info:
                    self.session_costs["total_input_tokens"] += cost_info.get("input_tokens", 0)
                    self.session_costs["total_output_tokens"] += cost_info.get("output_tokens", 0)
                    self.session_costs["total_tokens"] += cost_info.get("total_tokens", 0)
                    self.session_costs["total_cost_usd"] += cost_info.get("cost_usd", 0.0)

                # Add the assistant's response to the database with provider and model information
                provider = get_current_provider()
                with benchmark_resource("Database", "insert assistant node"):
                    assistant_node_id, assistant_short_id = insert_node(
                        display_response, 
                        user_node_id, 
                        role="assistant",
                        provider=provider,
                        model=model
                    )

                # Update the current node to the assistant's response
                self.current_node_id = assistant_node_id
                with benchmark_resource("Database", "set head"):
                    set_head(assistant_node_id)
                
                # Process topic changes based on our earlier detection
                if topic_changed:
                    # Topic change detected - close previous topic and start new one
                    recent_topics = get_recent_topics(limit=1)
                    if recent_topics:
                        previous_topic = recent_topics[0]
                        # Find the parent of the user node (should be the last assistant message)
                        user_node = get_node(user_node_id)
                        if user_node and user_node.get('parent_id'):
                            parent_node_id = user_node['parent_id']
                            
                            # Extract the topic name from the previous topic's content
                            topic_nodes = []
                            ancestry = get_ancestry(parent_node_id)
                            
                            # Collect nodes from the previous topic
                            found_start = False
                            for i, node in enumerate(ancestry):
                                if node['id'] == previous_topic['start_node_id']:
                                    found_start = True
                                    # Collect all nodes from start to end (parent_node_id)
                                    for j in range(i, len(ancestry)):
                                        topic_nodes.append(ancestry[j])
                                        if ancestry[j]['id'] == parent_node_id:
                                            break
                                    break
                            
                            if config.get("debug", False) and not found_start:
                                typer.echo(f"   WARNING: Start node {previous_topic['start_node_id']} not found in ancestry")
                            
                            # Build conversation segment from the previous topic
                            if topic_nodes:
                                segment = build_conversation_segment(topic_nodes, max_length=2000)
                                
                                if config.get("debug", False):
                                    typer.echo(f"\n🔍 DEBUG: Extracting name for previous topic '{previous_topic['name']}'")
                                    typer.echo(f"   Topic has {len(topic_nodes)} nodes")
                                    typer.echo(f"   Segment preview: {segment[:200]}...")
                                
                                # Extract a proper name for the previous topic
                                with benchmark_operation("Topic Name Extraction"):
                                    topic_name, extract_cost_info = extract_topic_ollama(segment)
                                
                                # Add extraction costs to session
                                if extract_cost_info:
                                    self.session_costs["total_input_tokens"] += extract_cost_info.get("input_tokens", 0)
                                    self.session_costs["total_output_tokens"] += extract_cost_info.get("output_tokens", 0)
                                    self.session_costs["total_tokens"] += extract_cost_info.get("total_tokens", 0)
                                    self.session_costs["total_cost_usd"] += extract_cost_info.get("cost_usd", 0.0)
                                
                                if config.get("debug", False):
                                    typer.echo(f"   Extracted topic name: {topic_name if topic_name else 'None (extraction failed)'}")
                                
                                final_topic_name = topic_name if topic_name else previous_topic['name']
                            else:
                                if config.get("debug", False):
                                    typer.echo(f"   WARNING: No topic nodes found for '{previous_topic['name']}'")
                                final_topic_name = previous_topic['name']
                            
                            # Update the topic name if it changed
                            if final_topic_name != previous_topic['name']:
                                rows_updated = update_topic_name(previous_topic['name'], previous_topic['start_node_id'], final_topic_name)
                                if config.get("debug", False):
                                    typer.echo(f"   ✅ Updated topic name: '{previous_topic['name']}' → '{final_topic_name}' ({rows_updated} rows)")
                            
                            # Update the previous topic's end node
                            update_topic_end_node(final_topic_name, previous_topic['start_node_id'], parent_node_id)
                            
                            # Queue the old topic for compression
                            queue_topic_for_compression(previous_topic['start_node_id'], parent_node_id, final_topic_name)
                            if config.get("debug", False):
                                typer.echo(f"   📦 Queued topic '{final_topic_name}' for compression")
                    else:
                        # No previous topics exist - this is the first topic change
                        # Create a topic for the initial conversation before this point
                        user_node = get_node(user_node_id)
                        if user_node and user_node.get('parent_id'):
                            parent_node_id = user_node['parent_id']
                            
                            # Get all nodes from the beginning up to the parent of the current user node
                            conversation_chain = get_ancestry(parent_node_id)
                            
                            # Find the very first user node in the database
                            # Don't rely on ancestry chain which might be broken
                            from episodic.db import get_connection
                            with get_connection() as conn:
                                c = conn.cursor()
                                c.execute("""
                                    SELECT id, short_id FROM nodes 
                                    WHERE role = 'user' 
                                    ORDER BY ROWID 
                                    LIMIT 1
                                """)
                                row = c.fetchone()
                            
                            if row:  # Found first user node
                                first_user_node_id, first_user_short_id = row
                                
                                # Get all nodes from start to parent of current user node
                                # Don't rely on ancestry chain which might be broken
                                with get_connection() as conn2:
                                    c2 = conn2.cursor()
                                    # Get the parent node's ROWID
                                    c2.execute("SELECT ROWID FROM nodes WHERE id = ?", (parent_node_id,))
                                    parent_row = c2.fetchone()
                                    
                                    if parent_row and parent_row[0] >= 3:  # Need at least a few nodes
                                        # Get all nodes from beginning up to parent
                                        c2.execute('''
                                            SELECT id, short_id, role, content 
                                            FROM nodes 
                                            WHERE ROWID <= ?
                                            ORDER BY ROWID
                                        ''', (parent_row[0],))
                                        
                                        nodes = []
                                        for node_row in c2.fetchall():
                                            nodes.append({
                                                'id': node_row[0],
                                                'short_id': node_row[1],
                                                'role': node_row[2],
                                                'content': node_row[3]
                                            })
                                        
                                        # Build segment from the initial conversation
                                        segment = build_conversation_segment(nodes, max_length=2000)
                                        
                                        if config.get("debug", False):
                                            typer.echo(f"\n🔍 DEBUG: Creating topic for initial conversation:")
                                            typer.echo(f"   From node {first_user_short_id} to {parent_node_id}")
                                            typer.echo(f"   Conversation preview: {segment[:200]}...")
                                
                                        # Extract topic name
                                        with benchmark_operation("Topic Name Extraction"):
                                            topic_name, extract_cost_info = extract_topic_ollama(segment)
                                        
                                        # Add extraction costs to session
                                        if extract_cost_info:
                                            self.session_costs["total_input_tokens"] += extract_cost_info.get("input_tokens", 0)
                                            self.session_costs["total_output_tokens"] += extract_cost_info.get("output_tokens", 0)
                                            self.session_costs["total_tokens"] += extract_cost_info.get("total_tokens", 0)
                                            self.session_costs["total_cost_usd"] += extract_cost_info.get("cost_usd", 0.0)
                                        
                                        # Use fallback if extraction failed
                                        if not topic_name:
                                            topic_name = "initial-conversation"
                                        
                                        # Store the initial topic
                                        store_topic(topic_name, first_user_node_id, parent_node_id, 'initial')
                                        typer.echo("")
                                        typer.secho(f"📌 Created topic for initial conversation: {topic_name}", fg=get_system_color())
                                        
                                        # Queue for compression
                                        queue_topic_for_compression(first_user_node_id, parent_node_id, topic_name)
                
                    # Create a new topic starting from this user message
                    # Use a placeholder name that will be updated when this topic ends
                    timestamp = int(time.time())
                    placeholder_topic_name = f"ongoing-{timestamp}"
                    
                    # Create the topic with placeholder name
                    store_topic(placeholder_topic_name, user_node_id, assistant_node_id, 'detected')
                    
                    # Set as current topic
                    self.set_current_topic(placeholder_topic_name, user_node_id)
                    
                    typer.echo("")
                    typer.secho(f"🔄 Topic changed", fg=get_system_color())
                else:
                    # No topic change - extend the current topic if one exists
                    current_topic = self.get_current_topic()
                    if current_topic:
                        topic_name, start_node_id = current_topic
                        # Update it to include the new assistant response
                        update_topic_end_node(topic_name, start_node_id, assistant_node_id)
                        if config.get("debug", False):
                            typer.echo(f"🔍 DEBUG: Extended topic '{topic_name}' to include new response")
                    else:
                        # No topics exist yet and no topic change detected
                        # Check if ANY topics exist in the database
                        from episodic.db import get_connection
                        with get_connection() as conn:
                            c = conn.cursor()
                            c.execute("SELECT COUNT(*) FROM topics")
                            topic_count = c.fetchone()[0]
                        
                        # If no topics exist at all, create the first one
                        if topic_count == 0:
                            # Extract topic from this exchange
                            segment = f"User: {user_input}\n\nAssistant: {display_response[:500]}..."
                            
                            with benchmark_operation("Topic Name Extraction"):
                                topic_name, extract_cost_info = extract_topic_ollama(segment)
                            
                            if not topic_name:
                                topic_name = "conversation"
                            
                            # Create the initial topic for JUST this exchange
                            # It starts at the current user node and ends at the current assistant node
                            store_topic(topic_name, user_node_id, assistant_node_id, 'initial')
                            # Set as current topic
                            self.set_current_topic(topic_name, user_node_id)
                            typer.echo("")
                            typer.secho(f"📌 Created initial topic: {topic_name}", fg=get_system_color())
                            
                            # Add extraction costs if any
                            if extract_cost_info:
                                self.session_costs["total_input_tokens"] += extract_cost_info.get("input_tokens", 0)
                                self.session_costs["total_output_tokens"] += extract_cost_info.get("output_tokens", 0)
                                self.session_costs["total_tokens"] += extract_cost_info.get("total_tokens", 0)
                                self.session_costs["total_cost_usd"] += extract_cost_info.get("cost_usd", 0.0)
                        else:
                            # Topics exist but no change detected - should have extended existing topic above
                            if config.get("debug", False):
                                typer.echo("🔍 DEBUG: No topic change detected, continuing conversation")
            
                # Show topic evolution if enabled (after topic detection)
                if config.get("show_topics", False):
                    _display_topic_evolution(assistant_node_id)
                
                # Add blank line after LLM response
                typer.echo("")
                
                return assistant_node_id, display_response

            except Exception as e:
                # Clear any pending benchmarks on error to avoid accumulation
                from episodic.benchmark import benchmark_manager
                benchmark_manager.pending_displays.clear()
                typer.echo(f"Error: {e}")
                raise


# Create a global instance for convenience
conversation_manager = ConversationManager()


# Expose the main functions at module level for backward compatibility
def handle_chat_message(
    user_input: str,
    model: str,
    system_message: str,
    context_depth: int = DEFAULT_CONTEXT_DEPTH
) -> Tuple[str, str]:
    """See ConversationManager.handle_chat_message for documentation."""
    result = conversation_manager.handle_chat_message(user_input, model, system_message, context_depth)
    # Display any pending benchmarks after the entire message processing is complete
    display_pending_benchmark()
    return result


def get_session_costs() -> Dict[str, Any]:
    """Get the current session costs."""
    return conversation_manager.get_session_costs()


def wrapped_text_print(text: str, **typer_kwargs) -> None:
    """Print text with automatic wrapping while preserving formatting."""
    conversation_manager.wrapped_text_print(text, **typer_kwargs)


def wrapped_llm_print(text: str, **typer_kwargs) -> None:
    """Print LLM text with automatic wrapping while preserving formatting."""
    conversation_manager.wrapped_llm_print(text, **typer_kwargs)