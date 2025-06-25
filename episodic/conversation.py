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
    
    def initialize_conversation(self) -> None:
        """Initialize the conversation state from the database."""
        self.current_node_id = get_head()
    
    def get_wrap_width(self) -> int:
        """Get the appropriate text wrapping width for the terminal."""
        terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        margin = 4
        max_width = 100  # Maximum line length for readability
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
                        subsequent_indent=indent + "  "  # Add slight extra indent for continuation
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
        self.wrapped_text_print(text, **typer_kwargs)
    
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
            
            if recent_nodes and len(recent_nodes) >= 2:  # Need at least some history
                with benchmark_operation("Topic Detection"):
                    topic_changed, new_topic_name, topic_cost_info = detect_topic_change_separately(recent_nodes, user_input)
            
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
                        typer.secho("🤖 ", fg=get_llm_color(), nl=False)
                        
                        # Process the stream and display it
                        from episodic.llm import process_stream_response
                        full_response_parts = []
                        
                        # Get streaming rate configuration
                        stream_rate = config.get("stream_rate", 15)  # Default to 15 words per second
                        use_constant_rate = config.get("stream_constant_rate", False)
                        
                        if use_constant_rate and stream_rate > 0:
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
                                                            subsequent_indent="   "
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
                            # Immediate streaming with proper word wrapping
                            current_line = ""
                            line_length = 0
                            wrap_width = self.get_wrap_width() if config.get("text_wrap", True) else None
                            
                            for chunk in process_stream_response(stream_generator, model):
                                full_response_parts.append(chunk)
                                
                                # If wrapping is disabled, just print the chunk
                                if not wrap_width:
                                    typer.secho(chunk, fg=get_llm_color(), nl=False)
                                    continue
                                
                                # Process each character in the chunk for wrapping
                                for char in chunk:
                                    if char == '\n':
                                        # Print the current line and start a new one
                                        typer.secho(current_line, fg=get_llm_color())
                                        current_line = ""
                                        line_length = 0
                                    else:
                                        # Add character to current line
                                        current_line += char
                                        line_length += 1
                                        
                                        # Check if we need to wrap
                                        if line_length >= wrap_width:
                                            # Find last space for word boundary
                                            last_space = current_line.rfind(' ')
                                            if last_space > 0 and last_space < len(current_line) - 1:
                                                # Print up to the last space
                                                typer.secho(current_line[:last_space], fg=get_llm_color())
                                                # Continue with the rest, indented
                                                current_line = "   " + current_line[last_space+1:]
                                                line_length = len(current_line)
                                            else:
                                                # No good break point, print as is and continue
                                                typer.secho(current_line, fg=get_llm_color(), nl=False)
                                                current_line = ""
                                                line_length = 0
                            
                            # Print any remaining content
                            if current_line:
                                typer.secho(current_line, fg=get_llm_color(), nl=False)
                        
                        # Get the full response and cost info
                        display_response = ''.join(full_response_parts)
                        
                        # Since we can't get accurate cost info from streaming yet,
                        # make a non-streaming call to get accurate costs
                        # This is a temporary workaround until litellm provides better streaming cost info
                        _, cost_info = query_with_context(
                            user_node_id, 
                            model=model,
                            system_message=system_message,
                            context_depth=context_depth,
                            stream=False
                        )
                        
                        # Add newline after streaming
                        typer.echo("")
                        
                        # Display cost info after streaming if enabled
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
                            self.wrapped_llm_print(f"🤖 {display_response}", fg=get_llm_color())
                        else:
                            # No status messages, just show blank line then LLM response
                            typer.echo("")  # Blank line
                            self.wrapped_llm_print(f"🤖 {display_response}", fg=get_llm_color())

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
                            for node in ancestry:
                                if node['id'] == previous_topic['start_node_id']:
                                    topic_nodes.append(node)
                                    # Continue collecting until we reach the end
                                    for subsequent_node in ancestry[ancestry.index(node):]:
                                        topic_nodes.append(subsequent_node)
                                        if subsequent_node['id'] == parent_node_id:
                                            break
                                    break
                            
                            # Build conversation segment from the previous topic
                            if topic_nodes:
                                segment = build_conversation_segment(topic_nodes, max_length=2000)
                                
                                # Extract a proper name for the previous topic
                                topic_name, extract_cost_info = extract_topic_ollama(segment)
                                
                                # Add extraction costs to session
                                if extract_cost_info:
                                    self.session_costs["total_input_tokens"] += extract_cost_info.get("input_tokens", 0)
                                    self.session_costs["total_output_tokens"] += extract_cost_info.get("output_tokens", 0)
                                    self.session_costs["total_tokens"] += extract_cost_info.get("total_tokens", 0)
                                    self.session_costs["total_cost_usd"] += extract_cost_info.get("cost_usd", 0.0)
                                
                                final_topic_name = topic_name if topic_name else previous_topic['name']
                            else:
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
                
                    # Create a new topic starting from this user message
                    # Generate a unique placeholder name that will be updated when the topic is closed
                    timestamp = int(time.time())
                    placeholder_topic_name = f"ongoing-discussion-{timestamp}"
                    store_topic(placeholder_topic_name, user_node_id, assistant_node_id, 'detected')
                    typer.echo("")
                    typer.secho(f"🔄 Topic changed", fg=get_system_color())
                else:
                    # No topic change - extend the current topic if one exists
                    recent_topics = get_recent_topics(limit=1)
                    if recent_topics:
                        current_topic = recent_topics[0]
                        # If no topic change was detected, we're continuing the current topic
                        # Update it to include the new assistant response
                        update_topic_end_node(current_topic['name'], current_topic['start_node_id'], assistant_node_id)
                    else:
                        # No topics exist yet - check if we should create the first topic
                        if should_create_first_topic(user_node_id):
                            # Get conversation for topic extraction
                            conversation_chain = get_ancestry(assistant_node_id)
                            
                            # Build segment from the entire conversation so far
                            segment = build_conversation_segment(conversation_chain, max_length=2000)
                            
                            if config.get("debug", False):
                                typer.echo(f"\n🔍 DEBUG: Extracting first topic from conversation:")
                                typer.echo(f"   Conversation preview: {segment[:200]}...")
                                typer.echo(f"   Total length: {len(segment)} chars")
                                typer.echo(f"   Number of nodes: {len(conversation_chain)}")
                            
                            # Only extract topic if we have actual content
                            if segment and segment.strip():
                                topic_name, extract_cost_info = extract_topic_ollama(segment)
                            else:
                                topic_name = None
                                extract_cost_info = None
                                if config.get("debug", False):
                                    typer.echo("   ⚠️  No conversation content found for topic extraction")
                            
                            # Add extraction costs to session
                            if extract_cost_info:
                                self.session_costs["total_input_tokens"] += extract_cost_info.get("input_tokens", 0)
                                self.session_costs["total_output_tokens"] += extract_cost_info.get("output_tokens", 0)
                                self.session_costs["total_tokens"] += extract_cost_info.get("total_tokens", 0)
                                self.session_costs["total_cost_usd"] += extract_cost_info.get("cost_usd", 0.0)
                            
                            # Use a generic fallback if extraction failed
                            if not topic_name:
                                topic_name = "conversation"
                                if config.get("debug", False):
                                    typer.echo("   Using fallback topic name: 'conversation'")
                            
                            if topic_name:
                                # Find the first user node to use as the start of the topic
                                first_user_node = None
                                for node in reversed(conversation_chain):
                                    if node.get('role') == 'user':
                                        first_user_node = node
                                
                                if first_user_node:
                                    # Store the topic spanning from first user message to current assistant message
                                    store_topic(topic_name, first_user_node['id'], assistant_node_id, 'initial')
                                    typer.echo("")
                                    typer.secho(f"📌 Created first topic: {topic_name}", fg=get_system_color())
            
                # Show topic evolution if enabled (after topic detection)
                if config.get("show_topics", False):
                    _display_topic_evolution(assistant_node_id)
                
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