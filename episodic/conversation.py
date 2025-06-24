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
from typing import Optional, List, Dict, Any, Tuple

import typer

from episodic.db import (
    insert_node, get_node, get_ancestry, get_head, set_head,
    get_recent_nodes, get_recent_topics, update_topic_end_node,
    store_topic
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
        # Add the user message to the database
        user_node_id, user_short_id = insert_node(user_input, self.current_node_id, role="user")
        
        # Detect topic change BEFORE querying the main LLM
        topic_changed = False
        new_topic_name = None
        
        # Get recent messages for context
        recent_nodes = get_recent_nodes(limit=10)  # Get last 10 nodes for context
        if recent_nodes and len(recent_nodes) >= 2:  # Need at least some history
            topic_changed, new_topic_name = detect_topic_change_separately(recent_nodes, user_input)
            
            if config.get("debug", False) and topic_changed:
                typer.echo(f"\n🔍 DEBUG: Topic change detected before LLM query")
                typer.echo(f"   New topic: {new_topic_name}")

        # Query the LLM with context
        try:
            # Query with context
            response, cost_info = query_with_context(
                user_node_id, 
                model=model,
                system_message=system_message,
                context_depth=context_depth
            )

            # Update session costs
            if cost_info:
                self.session_costs["total_input_tokens"] += cost_info.get("input_tokens", 0)
                self.session_costs["total_output_tokens"] += cost_info.get("output_tokens", 0)
                self.session_costs["total_tokens"] += cost_info.get("total_tokens", 0)
                self.session_costs["total_cost_usd"] += cost_info.get("cost_usd", 0.0)

            # Calculate and display semantic drift if enabled
            if config.get("show_drift", True):
                self.display_semantic_drift(user_node_id)

            # Use the response directly
            display_response = response
            
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

            # Add the assistant's response to the database with provider and model information
            provider = get_current_provider()
            assistant_node_id, assistant_short_id = insert_node(
                display_response, 
                user_node_id, 
                role="assistant",
                provider=provider,
                model=model
            )

            # Update the current node to the assistant's response
            self.current_node_id = assistant_node_id
            set_head(assistant_node_id)
            
            # Process topic changes based on our earlier detection
            if topic_changed and new_topic_name:
                # End the previous topic if one exists
                recent_topics = get_recent_topics(limit=1)
                if recent_topics:
                    previous_topic = recent_topics[0]
                    # Update the previous topic to end at the last assistant message before this user message
                    # Find the parent of the user node (should be the last assistant message)
                    parent_node = get_node(user_node_id).get('parent_id')
                    if parent_node:
                        update_topic_end_node(previous_topic['name'], previous_topic['start_node_id'], parent_node)
                        # Queue the old topic for compression
                        queue_topic_for_compression(previous_topic['start_node_id'], parent_node, previous_topic['name'])
                        if config.get("debug", False):
                            typer.echo(f"   📦 Queued topic '{previous_topic['name']}' for compression")
                
                # Create the new topic starting from this user message
                store_topic(new_topic_name, user_node_id, assistant_node_id, 'detected')
                typer.echo("")
                typer.secho(f"🔄 Topic changed to: {new_topic_name}", fg=get_system_color())
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
                        
                        topic_name = extract_topic_ollama(segment)
                        
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
    return conversation_manager.handle_chat_message(user_input, model, system_message, context_depth)


def get_session_costs() -> Dict[str, Any]:
    """Get the current session costs."""
    return conversation_manager.get_session_costs()


def wrapped_text_print(text: str, **typer_kwargs) -> None:
    """Print text with automatic wrapping while preserving formatting."""
    conversation_manager.wrapped_text_print(text, **typer_kwargs)


def wrapped_llm_print(text: str, **typer_kwargs) -> None:
    """Print LLM text with automatic wrapping while preserving formatting."""
    conversation_manager.wrapped_llm_print(text, **typer_kwargs)