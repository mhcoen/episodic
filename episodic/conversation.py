"""
Conversation management functionality for Episodic.

This module handles core conversation flow and delegates specialized
functionality to dedicated modules.
"""

from typing import Optional, Dict, Any, Tuple

import typer
from episodic.color_utils import secho_color, force_color_output

# Force color output if needed
force_color_output()

from episodic.db import (
    insert_node, get_ancestry, get_head, get_recent_nodes,
    get_recent_topics, update_topic_name
)
from episodic.llm import _execute_llm_query
from episodic.configuration import (
    get_llm_color, get_system_color,
    DEFAULT_CONTEXT_DEPTH
)
from episodic.config import config
from episodic.ml import ConversationalDrift
from episodic.topics import (
    build_conversation_segment, extract_topic_ollama,
    _display_topic_evolution
)
from episodic.benchmark import benchmark_operation, benchmark_resource

# Import specialized modules
from episodic.text_formatting import (
    wrapped_text_print, wrapped_llm_print
)
from episodic.debug_utils import debug_print
from episodic.topic_management import TopicHandler
from episodic.context_builder import ContextBuilder
from episodic.response_streaming import ResponseStreamer


class ConversationManager:
    """Manages conversation flow and coordinates specialized components."""
    
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
        
        # Initialize handlers
        self.topic_handler = TopicHandler(self)
        self.context_builder = ContextBuilder()
        self.response_streamer = ResponseStreamer()
    
    def get_session_costs(self) -> Dict[str, Any]:
        """Get the current session costs from the centralized LLM manager."""
        from episodic.llm_manager import llm_manager
        return llm_manager.get_session_costs()
    
    def reset_session_costs(self) -> None:
        """Reset session costs in the centralized LLM manager."""
        from episodic.llm_manager import llm_manager
        llm_manager.reset_stats()
        # Keep local tracking for backward compatibility but it won't be used
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
        old_topic = self.current_topic
        self.current_topic = (topic_name, start_node_id)
        if config.get("debug"):
            if old_topic:
                debug_print(f"Current topic changed from '{old_topic[0]}' to '{topic_name}'")
            else:
                debug_print(f"Current topic set to '{topic_name}'")
    
    def get_current_topic(self) -> Optional[Tuple[str, str]]:
        """Get the current topic (name, start_node_id) or None."""
        return self.current_topic
    
    def finalize_current_topic(self) -> None:
        """
        Finalize the current topic by giving it a proper name if it has a placeholder name.
        This is called when the conversation ends or when explicitly requested.
        """
        # Only finalize topics if automatic topic detection is enabled
        if not config.get("automatic_topic_detection"):
            return
            
        # If current_node_id is not set, try to get it from the database
        if not self.current_node_id:
            self.current_node_id = get_head()
            
        # Get all topics to find the actual last one (get_recent_topics returns oldest first)
        all_topics = get_recent_topics(limit=100)
        if not all_topics:
            return
            
        # The last topic in the list is the most recent one
        current_topic = all_topics[-1]
        
        # Check if it has a placeholder name
        if not current_topic['name'].startswith('ongoing-'):
            return  # Already has a proper name
            
        # Extract topic name from the conversation
        if config.get("debug"):
            typer.echo("")
            debug_print(f"Finalizing topic '{current_topic['name']}'")
            
        # Get nodes in the topic
        topic_nodes = []
        # For ongoing topics (end_node_id is NULL), use the current head
        if current_topic['end_node_id']:
            ancestry = get_ancestry(current_topic['end_node_id'])
        else:
            # Use current head for ongoing topics
            current_head = get_head()
            if current_head:
                ancestry = get_ancestry(current_head)
            else:
                ancestry = []
        
        if ancestry:
            # Collect nodes from topic start to end (or current for ongoing)
            found_start = False
            for node in ancestry:
                if node['id'] == current_topic['start_node_id']:
                    found_start = True
                if found_start:
                    topic_nodes.append(node)
                # For topics with an end, stop at the end node
                if current_topic['end_node_id'] and node['id'] == current_topic['end_node_id']:
                    break
                    
        if topic_nodes:
            # Build conversation segment
            segment = build_conversation_segment(topic_nodes, max_length=2000)
            
            # Extract topic name
            topic_name, _ = extract_topic_ollama(segment)
            
            if topic_name and topic_name != current_topic['name']:
                # Update the topic name
                rows_updated = update_topic_name(
                    current_topic['name'], 
                    current_topic['start_node_id'], 
                    topic_name
                )
                
                if config.get("debug"):
                    secho_color(f"   ✅ Finalized topic: '{current_topic['name']}' → '{topic_name}' ({rows_updated} rows)", fg='green', bold=True)
                    
                # Update current topic reference
                if self.current_topic and self.current_topic[0] == current_topic['name']:
                    self.set_current_topic(topic_name, self.current_topic[1])
    
    def initialize_conversation(self) -> None:
        """Initialize the conversation state from the database."""
        self.current_node_id = get_head()
        
        # Initialize current topic from database
        if self.current_node_id:
            # Find the topic that contains the current head node
            recent_topics = get_recent_topics(limit=100)  # Get more topics to search through
            
            # First, look for a topic that hasn't ended yet (ongoing topic)
            for topic in recent_topics:
                if not topic.get('end_node_id'):
                    # This topic is still ongoing
                    self.set_current_topic(topic['name'], topic['start_node_id'])
                    if config.get("debug"):
                        debug_print(f"Resuming ongoing topic '{topic['name']}'")
                    return
            
            # If no ongoing topic, find which topic contains the current head node
            if self.current_node_id:
                # Get the ancestry of the current node to check topic boundaries
                ancestry = get_ancestry(self.current_node_id)
                node_ids_in_chain = {node['id'] for node in ancestry}
                
                # Check each topic to see if current node falls within its range
                for topic in recent_topics:
                    start_id = topic['start_node_id']
                    end_id = topic.get('end_node_id')
                    
                    # If topic has both start and end, check if current node is between them
                    if start_id in node_ids_in_chain:
                        if not end_id or end_id in node_ids_in_chain:
                            # Current node is within this topic's range
                            # Check if current node comes after start but before end
                            start_found = False
                            current_found = False
                            end_found = False
                            
                            for node in ancestry:
                                if node['id'] == start_id:
                                    start_found = True
                                if node['id'] == self.current_node_id and start_found:
                                    current_found = True
                                if end_id and node['id'] == end_id:
                                    end_found = True
                                    break
                            
                            # If we found current between start and end (or no end), this is our topic
                            if current_found and (not end_id or not end_found):
                                self.set_current_topic(topic['name'], topic['start_node_id'])
                                if config.get("debug"):
                                    debug_print(f"Current node is in topic '{topic['name']}'")
                                return
            
            # No active topic found
            if config.get("debug"):
                debug_print("No active topic found for current head node")
    
    def get_drift_calculator(self) -> Optional[ConversationalDrift]:
        """Get or create the drift calculator instance."""
        # Check if drift detection is disabled in config
        if not config.get("show_drift"):
            return None
            
        if self.drift_calculator is None:
            try:
                # Get embedding settings from config
                embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
                embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")
                
                self.drift_calculator = ConversationalDrift(
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model
                )
                if config.get("debug"):
                    typer.echo(f"✅ Initialized drift calculator with {embedding_provider}/{embedding_model}")
            except Exception as e:
                # If drift calculator fails to initialize (e.g., missing dependencies),
                # disable drift detection for this session
                if config.get("debug"):
                    typer.echo(f"⚠️  Drift detection disabled: {e}")
                self.drift_calculator = False  # Mark as disabled
        return self.drift_calculator if self.drift_calculator is not False else None
    
    def display_semantic_drift(self, current_user_node_id: str) -> None:
        """
        Calculate and display semantic drift between consecutive user messages.
        
        Only compares user inputs to detect when the user changes topics,
        ignoring assistant responses which just follow the user's lead.
        """
        calc = self.get_drift_calculator()
        if not calc:
            return  # Drift detection disabled
        
        if config.get("debug"):
            typer.echo(f"   [drift] Using calculator instance: {id(calc)}")
        
        try:
            # Get conversation history from root to current node
            conversation_chain = get_ancestry(current_user_node_id)
            
            # Filter to user messages only
            user_messages = [node for node in conversation_chain 
                            if node.get("role") == "user" and node.get("content", "").strip()]
            
            # Need at least 2 user messages for comparison
            if len(user_messages) < 2:
                if config.get("debug"):
                    debug_print(f"(Need 2 user messages for drift, have {len(user_messages)})", indent=True)
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
            secho_color(f"\n{drift_emoji} Semantic drift: {drift_score:.3f} ({drift_desc}) from user message {prev_short_id}", fg=get_system_color())
            
            # Show additional context if debug mode is enabled
            if config.get("debug"):
                prev_content = previous_user.get("content", "")[:80]
                curr_content = current_user.get("content", "")[:80]
                debug_print(f"Previous: {prev_content}{'...' if len(previous_user.get('content', '')) > 80 else ''}", indent=True)
                debug_print(f"Current:  {curr_content}{'...' if len(current_user.get('content', '')) > 80 else ''}", indent=True)
                
                # Show embedding cache efficiency
                cache_size = calc.get_cache_size()
                debug_print(f"Embedding cache: {cache_size} entries", indent=True)
            
        except Exception as e:
            # If drift calculation fails, silently continue (don't disrupt conversation flow)
            if config.get("debug"):
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
        """
        # Import db functions locally to avoid Python scoping issues
        
        with benchmark_operation("Message Processing"):
            # Get recent messages for context BEFORE adding the new message
            with benchmark_resource("Database", "get recent nodes"):
                # For sliding window detection, we need more history
                detection_history_limit = 50 if config.get("use_sliding_window_detection") else 10
                recent_nodes = get_recent_nodes(limit=detection_history_limit)
            
            # Add the user message to the database
            with benchmark_resource("Database", "insert user node"):
                user_node_id, user_short_id = insert_node(user_input, self.current_node_id, role="user")
        
            # Detect topic change BEFORE querying the main LLM
            topic_changed, new_topic_name, topic_cost_info, topic_change_info = \
                self.topic_handler.detect_and_handle_topic_change(
                    recent_nodes, user_input, user_node_id
                )
            
            # Store topic detection scores for debugging
            self.topic_handler.store_topic_detection_scores(
                recent_nodes, user_node_id, topic_cost_info, topic_changed
            )
            
            # Update node ID
            self.set_current_node_id(user_node_id)
            
            # Check if skip_llm_response is enabled
            if config.get("skip_llm_response", False):
                # Create a placeholder response
                display_response = "[LLM response skipped]"
                assistant_node_id, assistant_short_id = insert_node(
                    display_response, 
                    user_node_id, 
                    role="assistant"
                )
                
                # Display drift if enabled
                if config.get("show_drift"):
                    self.display_semantic_drift(user_node_id)
                
                # Display the skipped response message
                typer.echo("")
                secho_color(f"🤖 {display_response}", fg=get_system_color())
                
                # Update current node
                self.set_current_node_id(assistant_node_id)
                
                # Handle topic boundaries
                self.topic_handler.handle_topic_boundaries(
                    topic_changed, user_node_id, assistant_node_id, topic_change_info
                )
                
                # Check for first topic creation
                if not topic_changed and not self.current_topic and config.get("automatic_topic_detection"):
                    self.topic_handler.check_and_create_first_topic(user_node_id, assistant_node_id)
                
                return assistant_node_id, display_response
            
            # Build conversation context
            messages, raw_messages, rag_context, web_context = \
                self.context_builder.build_conversation_context(
                    user_node_id, user_input, context_depth, model
                )
            
            # Display topic evolution if requested
            if config.get("show_topics") and raw_messages:
                _display_topic_evolution(user_node_id)
            
            # Display drift if enabled
            if config.get("show_drift"):
                self.display_semantic_drift(user_node_id)
            
            # Prepare for LLM query
            from episodic.web_synthesis import synthesize_web_response
            
            # Check if we're in muse mode and have web context
            if config.get("muse_mode") and web_context:
                # Use web synthesis for muse mode
                full_response = synthesize_web_response(
                    query=user_input,
                    search_results=web_context,
                    conversation_history=messages,
                    model=model
                )
                display_response = full_response
            else:
                # Regular LLM query
                with benchmark_resource("LLM Call", f"main query - {model}"):
                    # Query the LLM with streaming
                    stream_enabled = config.get("stream_responses", True)
                    
                    if stream_enabled:
                        # Get streaming configuration
                        stream_rate = float(config.get("stream_rate", 15.0))
                        use_constant_rate = config.get("stream_constant_rate", False)
                        use_natural_rhythm = config.get("stream_natural_rhythm", False)
                        
                        # Get the stream generator
                        with benchmark_resource("LLM", f"query stream - {model}"):
                            stream_generator, _ = _execute_llm_query(
                                messages=messages,
                                model=model,
                                stream=True
                            )
                        
                        # Stream the response
                        typer.echo("")  # Newline before streaming
                        full_response = self.response_streamer.stream_response(
                            stream_generator=stream_generator,
                            model=model,
                            stream_rate=stream_rate,
                            use_constant_rate=use_constant_rate,
                            use_natural_rhythm=use_natural_rhythm
                        )
                        display_response = full_response
                    else:
                        # Non-streaming response
                        with benchmark_resource("LLM", f"query - {model}"):
                            response, cost_info = _execute_llm_query(
                                messages=messages,
                                model=model,
                                stream=False
                            )
                        
                        # Display the response
                        if response:
                            typer.echo("")
                            wrapped_llm_print(response, fg=get_llm_color())
                            display_response = response
                        else:
                            display_response = "[No response from LLM]"
                            typer.echo("")
                            secho_color(display_response, fg='red')
            
            # Store the assistant's response
            with benchmark_resource("Database", "insert assistant node"):
                assistant_node_id, assistant_short_id = insert_node(
                    display_response, 
                    user_node_id, 
                    role="assistant"
                )
            
            # Update current node
            self.set_current_node_id(assistant_node_id)
            
            # Track RAG usage if applicable
            if rag_context:
                self.context_builder.track_rag_usage(assistant_node_id)
            
            # Handle topic boundaries
            self.topic_handler.handle_topic_boundaries(
                topic_changed, user_node_id, assistant_node_id, topic_change_info
            )
            
            # Check for first topic creation or update ongoing topic
            if config.get("automatic_topic_detection"):
                if not topic_changed and not self.current_topic:
                    self.topic_handler.check_and_create_first_topic(user_node_id, assistant_node_id)
                elif self.current_topic:
                    # Update ongoing topic name if needed
                    self.topic_handler.update_ongoing_topic_name(assistant_node_id)
            
            return assistant_node_id, display_response


# Create a module-level instance for backward compatibility
conversation_manager = ConversationManager()


# Module-level functions for backward compatibility
def handle_chat_message(
    user_input: str,
    model: str,
    system_message: str,
    context_depth: int = DEFAULT_CONTEXT_DEPTH,
    conversation_manager: Optional[ConversationManager] = None
) -> Tuple[str, str]:
    """Module-level wrapper for ConversationManager.handle_chat_message()."""
    if conversation_manager is None:
        # Use the module-level instance
        conversation_manager = globals()['conversation_manager']
    return conversation_manager.handle_chat_message(user_input, model, system_message, context_depth)


def get_session_costs() -> Dict[str, Any]:
    """Get session costs from centralized LLM manager."""
    from episodic.llm_manager import llm_manager
    return llm_manager.get_session_costs()


# Re-export text formatting functions for backward compatibility
wrapped_text_print = wrapped_text_print
wrapped_llm_print = wrapped_llm_print