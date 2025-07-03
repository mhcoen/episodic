"""
Conversation management functionality for Episodic.

This module handles all conversation-related operations including:
- Managing chat messages and responses
- Building conversation context
- Semantic drift detection
- Session cost tracking
- Text formatting and wrapping
"""

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
    insert_node, get_ancestry, get_head, set_head, get_recent_nodes,
    get_recent_topics, update_topic_end_node, store_topic,
    update_topic_name, get_node
)
from episodic.llm import query_with_context
from episodic.llm_config import get_current_provider
from episodic.configuration import get_model_context_limit
from episodic.config import config
from episodic.configuration import (
    get_llm_color, get_system_color, DEFAULT_CONTEXT_DEPTH,
    COST_PRECISION, format_cost
)
from episodic.ml import ConversationalDrift
from episodic.compression import queue_topic_for_compression
from episodic.topics import (
    detect_topic_change_separately, extract_topic_ollama, 
    should_create_first_topic, build_conversation_segment,
    _display_topic_evolution
)
from episodic.topic_boundary_analyzer import (
    analyze_topic_boundary, find_transition_point_heuristic
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
        old_topic = self.current_topic
        self.current_topic = (topic_name, start_node_id)
        if config.get("debug"):
            if old_topic:
                typer.echo(f"🔄 DEBUG: Current topic changed from '{old_topic[0]}' to '{topic_name}'")
            else:
                typer.echo(f"🔄 DEBUG: Current topic set to '{topic_name}'")
    
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
            typer.echo(f"\n🔍 DEBUG: Finalizing topic '{current_topic['name']}'")
            
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
                    typer.echo(f"   ✅ Finalized topic: '{current_topic['name']}' → '{topic_name}' ({rows_updated} rows)")
                    
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
                        typer.echo(f"🔍 DEBUG: Resuming ongoing topic '{topic['name']}'")
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
                                    typer.echo(f"🔍 DEBUG: Current node is in topic '{topic['name']}'")
                                return
            
            # No active topic found
            if config.get("debug"):
                typer.echo("🔍 DEBUG: No active topic found for current head node")
    
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
        if not config.get("text_wrap"):
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
        if not config.get("show_drift"):
            return None
            
        if self.drift_calculator is None:
            try:
                self.drift_calculator = ConversationalDrift()
            except Exception as e:
                # If drift calculator fails to initialize (e.g., missing dependencies),
                # disable drift detection for this session
                if config.get("debug"):
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
                if config.get("debug"):
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
            if config.get("debug"):
                prev_content = previous_user.get("content", "")[:80]
                curr_content = current_user.get("content", "")[:80]
                typer.echo(f"   Previous: {prev_content}{'...' if len(previous_user.get('content', '')) > 80 else ''}")
                typer.echo(f"   Current:  {curr_content}{'...' if len(current_user.get('content', '')) > 80 else ''}")
                
                # Show embedding cache efficiency
                cache_size = calc.get_cache_size()
                typer.echo(f"   Embedding cache: {cache_size} entries")
            
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
            
        Processes the message through the LLM and updates the conversation DAG.
        """
        with benchmark_operation("Message Processing"):
            # Get recent messages for context BEFORE adding the new message
            with benchmark_resource("Database", "get recent nodes"):
                # For sliding window detection, we need more history to properly slide the window
                detection_history_limit = 50 if config.get("use_sliding_window_detection") else 10
                recent_nodes = get_recent_nodes(limit=detection_history_limit)  # Get more nodes for sliding window
            
            # Add the user message to the database
            with benchmark_resource("Database", "insert user node"):
                user_node_id, user_short_id = insert_node(user_input, self.current_node_id, role="user")
        
            # Detect topic change BEFORE querying the main LLM
            topic_changed = False
            new_topic_name = None
            topic_cost_info = None
            
            # Check if automatic topic detection is enabled
            if config.get("automatic_topic_detection"):
                if config.get("debug"):
                    typer.echo(f"\n🔍 DEBUG: Topic detection check")
                    typer.echo(f"   Recent nodes count: {len(recent_nodes) if recent_nodes else 0}")
                    typer.echo(f"   Current topic: {self.current_topic}")
                    typer.echo(f"   Min messages before topic change: {config.get('min_messages_before_topic_change')}")
                
                if recent_nodes and len(recent_nodes) >= 2:  # Need at least some history
                    try:
                        with benchmark_operation("Topic Detection"):
                            # Debug: show which detector will be used
                            if config.get("debug"):
                                typer.echo(f"   Detection config: hybrid={config.get('use_hybrid_topic_detection')}, sliding={config.get('use_sliding_window_detection')}")
                            
                            # Use hybrid detection if enabled
                            if config.get("use_hybrid_topic_detection"):
                                if config.get("debug"):
                                    typer.echo("   Using HYBRID detection")
                                from episodic.topics.hybrid import HybridTopicDetector
                                detector = HybridTopicDetector()
                                topic_changed, new_topic_name, topic_cost_info = detector.detect_topic_change(
                                    recent_nodes,
                                    user_input,
                                    current_topic=self.current_topic
                                )
                            elif config.get("use_sliding_window_detection"):
                                if config.get("debug"):
                                    typer.echo("   Using SLIDING WINDOW detection")
                                # Use sliding window detection (3-3 windows)
                                from episodic.topics.realtime_windows import RealtimeWindowDetector
                                window_size = config.get("sliding_window_size", 3)
                                detector = RealtimeWindowDetector(window_size=window_size)
                                topic_changed, new_topic_name, topic_cost_info = detector.detect_topic_change(
                                    recent_nodes,
                                    user_input,
                                    current_topic=self.current_topic
                                )
                            else:
                                # Use standard LLM-based detection
                                from episodic.topics.detector import topic_manager
                                topic_changed, new_topic_name, topic_cost_info = topic_manager.detect_topic_change_separately(
                                    recent_nodes, 
                                    user_input,
                                    current_topic=self.current_topic
                                )
                            if config.get("debug"):
                                typer.echo(f"   Topic change detected: {topic_changed}")
                                if topic_changed:
                                    typer.echo(f"   New topic: {new_topic_name}")
                    except Exception as e:
                        if config.get("debug"):
                            typer.echo(f"   ❌ Topic detection error: {e}")
                        # Continue without topic detection on error
                        topic_changed = False
                else:
                    if config.get("debug"):
                        typer.echo("   ⚠️  Not enough history for topic detection")
            else:
                # Automatic topic detection is disabled
                if config.get("debug"):
                    typer.echo("\n🔍 DEBUG: Automatic topic detection is disabled")
            
            # Store topic detection scores for debugging (only if automatic detection is enabled)
            if config.get("automatic_topic_detection") and recent_nodes and len(recent_nodes) >= 2:
                # Store window-based detection scores in the window detection table
                if config.get("use_sliding_window_detection") and topic_cost_info and topic_cost_info.get("method") == "sliding_window":
                    try:
                        # Import get_node locally to avoid scope issues
                        from episodic.db import get_node as get_node_info
                        
                        # Get user node info
                        user_node = get_node_info(user_node_id)
                        if user_node and user_node.get('short_id'):
                            window_a_messages = topic_cost_info.get("window_a_messages", [])
                            
                            # Only store if we have window messages
                            if window_a_messages:
                                # Debug: Check what we're storing
                                if config.get("debug"):
                                    typer.echo(f"   DEBUG: Storing window for {user_node['short_id']}")
                                    typer.echo(f"   Window A messages: {[m.get('short_id', '?') for m in window_a_messages]}")
                                    typer.echo(f"   Start: {window_a_messages[0].get('short_id', '?')}, End: {window_a_messages[-1].get('short_id', '?')}")
                                
                                # Store directly to manual_index_scores table
                                from episodic.db import get_connection
                                
                                with get_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute("""
                                        INSERT INTO manual_index_scores (
                                            user_node_short_id, window_size,
                                            window_a_start_short_id, window_a_end_short_id, window_a_size,
                                            window_b_start_short_id, window_b_end_short_id, window_b_size,
                                            drift_score, keyword_score, combined_score,
                                            is_boundary, transition_phrase, threshold_used
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        user_node['short_id'],
                                        topic_cost_info.get("window_size", 3),
                                        window_a_messages[0].get('short_id', 'unknown') if window_a_messages else 'unknown',
                                        window_a_messages[-1].get('short_id', 'unknown') if window_a_messages else 'unknown',
                                        len(window_a_messages),
                                        user_node['short_id'],
                                        user_node['short_id'],
                                        1,
                                        topic_cost_info.get("drift_score", 0.0),
                                        topic_cost_info.get("keyword_score", 0.0),
                                        topic_cost_info.get("combined_score", topic_cost_info.get("drift_score", 0.0)),
                                        topic_cost_info.get("is_boundary", False),
                                        topic_cost_info.get("transition_phrase"),
                                        topic_cost_info.get("threshold_used")
                                    ))
                                if config.get("debug"):
                                    typer.echo(f"   ✅ Stored window detection score for {user_node['short_id']}")
                    except Exception as e:
                        if config.get("debug"):
                            typer.echo(f"   ⚠️ Failed to store window detection score: {e}")
                
                # Also store general detection scores
                from episodic.db import store_topic_detection_scores
                from episodic.topics import topic_manager
                import json
                
                # Get context information
                topics = get_recent_topics(limit=100)
                total_topics = len(topics)
                
                # Count user messages in current topic
                user_messages_in_topic = 0
                if self.current_topic:
                    user_messages_in_topic = topic_manager.count_user_messages_in_topic(
                        self.current_topic[1], 
                        None  # None means count to current head
                    )
                
                # Calculate effective threshold
                min_messages = config.get('min_messages_before_topic_change', 8)
                if total_topics <= 2:
                    effective_threshold = max(4, min_messages // 2)
                else:
                    effective_threshold = min_messages
                
                # Extract scores based on detection method
                detection_method = topic_cost_info.get("method", "unknown") if topic_cost_info else "no_detection"
                
                # Default values
                scores_data = {
                    "user_node_id": user_node_id,
                    "topic_changed": topic_changed,
                    "detection_method": detection_method,
                    "user_messages_in_topic": user_messages_in_topic,
                    "total_topics_count": total_topics,
                    "effective_threshold": effective_threshold
                }
                
                # Add sliding window detection scores if available
                if detection_method == "sliding_window" and topic_cost_info:
                    scores_data.update({
                        "semantic_drift_score": topic_cost_info.get("drift_score"),
                        "final_score": topic_cost_info.get("combined_score", topic_cost_info.get("drift_score")),
                        "transition_phrase": topic_cost_info.get("transition_phrase")
                    })
                # Add hybrid detection scores if available
                elif topic_cost_info and "signals" in topic_cost_info:
                    signals = topic_cost_info["signals"]
                    scores_data.update({
                        "final_score": topic_cost_info.get("score"),
                        "semantic_drift_score": signals.get("semantic_drift"),
                        "keyword_explicit_score": signals.get("keyword_explicit"),
                        "keyword_domain_score": signals.get("keyword_domain"),
                        "message_gap_score": signals.get("message_gap"),
                        "conversation_flow_score": signals.get("conversation_flow"),
                        "transition_phrase": topic_cost_info.get("transition_phrase")
                    })
                    
                    # Add domain information if available
                    if "detected_domains" in topic_cost_info:
                        scores_data["detected_domains"] = json.dumps(topic_cost_info["detected_domains"])
                    if "dominant_domain" in topic_cost_info:
                        scores_data["dominant_domain"] = topic_cost_info["dominant_domain"]
                    if "previous_domain" in topic_cost_info:
                        scores_data["previous_domain"] = topic_cost_info["previous_domain"]
                
                # Add LLM response if available
                if topic_cost_info and "llm_response" in topic_cost_info:
                    scores_data["llm_response"] = topic_cost_info["llm_response"]
                if topic_cost_info and "llm_confidence" in topic_cost_info:
                    scores_data["llm_confidence"] = topic_cost_info["llm_confidence"]
                
                # Store the scores
                try:
                    store_topic_detection_scores(**scores_data)
                except Exception as e:
                    if config.get("debug"):
                        typer.echo(f"   ⚠️  Failed to store topic detection scores: {e}")
            
            # Add topic detection costs to session
            if topic_cost_info:
                self.session_costs["total_input_tokens"] += topic_cost_info.get("input_tokens", 0)
                self.session_costs["total_output_tokens"] += topic_cost_info.get("output_tokens", 0)
                self.session_costs["total_tokens"] += topic_cost_info.get("total_tokens", 0)
                self.session_costs["total_cost_usd"] += topic_cost_info.get("cost_usd", 0.0)
            
            # Store debug info to display later
            debug_topic_info = None
            if config.get("debug") and topic_changed:
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
                        if config.get("show_drift"):
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
                        
                        if config.get("debug"):
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
                                                if config.get("text_wrap"):
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
                                                if config.get("text_wrap"):
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
                            wrap_width = self.get_wrap_width() if config.get("text_wrap") else None
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
                                                typer.secho('\n', nl=False)
                                                current_position = 0
                                                line_start = True
                                            
                                            # Print word
                                            typer.secho(current_word, fg=get_llm_color(), nl=False, bold=word_is_bold)
                                            current_position += len(current_word)
                                            
                                            # Reset numbered list flag after printing word with colon
                                            if current_word.endswith(':') and in_numbered_list:
                                                in_numbered_list = False
                                            
                                            current_word = ""
                                            
                                            if not char.isspace():
                                                line_start = False
                                        
                                        # Print space or newline
                                        if char == '\n':
                                            typer.secho('\n', nl=False)
                                            current_position = 0
                                            line_start = True
                                            in_numbered_list = False  # Reset after newline
                                        else:
                                            typer.secho(' ', fg=get_llm_color(), nl=False)
                                            current_position += 1
                                    else:
                                        # Accumulate character
                                        current_word += char
                            
                            # Print remaining word
                            if current_word:
                                word_is_bold = in_bold
                                if line_start and current_word.rstrip('.').isdigit():
                                    in_numbered_list = True
                                
                                # Bold everything in numbered list item line
                                if in_numbered_list:
                                    word_is_bold = True
                                    
                                if wrap_width and current_position + len(current_word) > wrap_width:
                                    typer.secho('\n', nl=False)
                                typer.secho(current_word, fg=get_llm_color(), nl=False, bold=word_is_bold)
                        
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
                        
                        # Count input tokens by reconstructing the context
                        # Rebuild the same messages that were sent to the LLM
                        from episodic.db import get_node, get_ancestry
                        
                        # Get the user node
                        user_node = get_node(user_node_id)
                        if not user_node:
                            input_tokens = 100  # Fallback estimate
                        else:
                            # Get ancestry for context
                            ancestry = get_ancestry(user_node_id)
                            if context_depth > 0:
                                ancestry = ancestry[-context_depth:]
                            
                            # Build messages list same as query_with_context
                            messages = [{"role": "system", "content": system_message}]
                            
                            # Add context from ancestry
                            for ancestor in ancestry[:-1]:  # Exclude current node
                                messages.append({
                                    "role": ancestor.get("role", "user"),
                                    "content": ancestor.get("content", "")
                                })
                            
                            # Add the user's message
                            messages.append({"role": "user", "content": user_node["content"]})
                            
                            # Count tokens in all messages
                            input_tokens = 0
                            for msg in messages:
                                input_tokens += token_counter(model=model, text=msg.get('content', ''))
                        
                        # Calculate cost using litellm
                        try:
                            from litellm import completion_cost
                            # Build prompt text for cost calculation
                            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                            
                            # Calculate cost using prompt and completion text
                            cost_usd = completion_cost(
                                model=model,
                                prompt=prompt_text,
                                completion=display_response
                            )
                        except Exception as e:
                            if config.get('debug', False):
                                typer.echo(f"DEBUG: Cost calculation failed: {e}")
                            cost_usd = 0.0
                        
                        cost_info = {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'cost_usd': cost_usd
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
                            
                            cost_msg = f"Tokens: {cost_info.get('total_tokens', 0)} | Cost: {format_cost(cost_info.get('cost_usd', 0.0))} USD | Context: {context_display} full"
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
                        if config.get("show_drift"):
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
                            
                            status_messages.append(f"Tokens: {cost_info.get('total_tokens', 0)} | Cost: {format_cost(cost_info.get('cost_usd', 0.0))} USD | Context: {context_display} full")

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
                
                # Process topic changes based on our earlier detection (only if automatic detection is enabled)
                if config.get("automatic_topic_detection") and topic_changed:
                    # Topic change detected - close previous topic and start new one
                    # Use current_topic to identify which topic to close
                    if self.current_topic:
                        topic_name, start_node_id = self.current_topic
                        previous_topic = {
                            'name': topic_name,
                            'start_node_id': start_node_id
                        }
                        # Find the parent of the user node (should be the last assistant message)
                        user_node = get_node(user_node_id)
                        if user_node and user_node.get('parent_id'):
                            parent_node_id = user_node['parent_id']
                            
                            if config.get("debug"):
                                typer.echo(f"   DEBUG: Topic boundary - user_node_id={user_node_id[:8]}, parent_node_id={parent_node_id[:8]}")
                                parent_node = get_node(parent_node_id)
                                if parent_node:
                                    typer.echo(f"   Parent node: {parent_node.get('short_id')} ({parent_node.get('role')})")
                            
                            # Before closing the previous topic, check if it has enough user messages
                            # Count user messages in the previous topic
                            from episodic.topics import TopicManager
                            tm = TopicManager()
                            user_messages_in_prev = tm.count_user_messages_in_topic(
                                previous_topic['start_node_id'],
                                parent_node_id  # End at parent of current user node
                            )
                            
                            # Let topics change based on drift - no message count restrictions
                            if True:  # Always proceed with topic change when drift is high
                                # Previous topic has enough messages, proceed with topic change
                                
                                # Analyze where the topic actually changed
                                actual_boundary = parent_node_id  # Default to current boundary
                                
                                # Check if boundary analysis is enabled (default: True)
                                if config.get("analyze_topic_boundaries", True):
                                    if config.get("debug"):
                                        typer.echo(f"\n🔍 DEBUG: Analyzing topic boundary...")
                                    
                                    # Get recent conversation history for analysis
                                    # We need more context than just recent_nodes
                                    full_ancestry = get_ancestry(user_node_id)
                                    
                                    # Use boundary analyzer to find actual transition
                                    if config.get("use_llm_boundary_analysis", True):
                                        # Use LLM-based analysis
                                        boundary_result = analyze_topic_boundary(
                                            full_ancestry[-20:] if len(full_ancestry) > 20 else full_ancestry,
                                            user_node_id,
                                            config.get("topic_detection_model", "ollama/llama3")
                                        )
                                        transition_type = "llm"
                                        boundary_cost = None
                                        
                                        # Add boundary analysis costs
                                        if boundary_cost:
                                            self.session_costs["total_input_tokens"] += boundary_cost.get("input_tokens", 0)
                                            self.session_costs["total_output_tokens"] += boundary_cost.get("output_tokens", 0)
                                            self.session_costs["total_tokens"] += boundary_cost.get("total_tokens", 0)
                                            self.session_costs["total_cost_usd"] += boundary_cost.get("cost_usd", 0.0)
                                        
                                        if boundary_result:
                                            actual_boundary = boundary_result
                                            if config.get("debug"):
                                                typer.echo(f"   Found actual boundary: {actual_boundary} (type: {transition_type})")
                                    else:
                                        # Use heuristic-based analysis (no LLM)
                                        heuristic_boundary = find_transition_point_heuristic(
                                            full_ancestry[-20:] if len(full_ancestry) > 20 else full_ancestry
                                        )
                                        if heuristic_boundary:
                                            actual_boundary = heuristic_boundary
                                            if config.get("debug"):
                                                typer.echo(f"   Found heuristic boundary: {actual_boundary}")
                                
                                # Extract the topic name from the previous topic's content
                                topic_nodes = []
                                ancestry = get_ancestry(actual_boundary)
                                
                                # Collect nodes from the previous topic
                                found_start = False
                                for i, node in enumerate(ancestry):
                                    if node['id'] == previous_topic['start_node_id']:
                                        found_start = True
                                        # Collect all nodes from start to end (parent_node_id)
                                        for j in range(i, len(ancestry)):
                                            topic_nodes.append(ancestry[j])
                                            if ancestry[j]['id'] == actual_boundary:
                                                break
                                        break
                            
                                if config.get("debug") and not found_start:
                                    typer.echo(f"   WARNING: Start node {previous_topic['start_node_id']} not found in ancestry")
                                
                                # Build conversation segment from the previous topic
                                if topic_nodes:
                                    segment = build_conversation_segment(topic_nodes, max_length=2000)
                                
                                    
                                    if config.get("debug"):
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
                                    
                                    if config.get("debug"):
                                        typer.echo(f"   Extracted topic name: {topic_name if topic_name else 'None (extraction failed)'}")
                                    
                                    final_topic_name = topic_name if topic_name else previous_topic['name']
                                else:
                                    if config.get("debug"):
                                        typer.echo(f"   WARNING: No topic nodes found for '{previous_topic['name']}'")
                                    final_topic_name = previous_topic['name']
                                
                                # Update the topic name if it changed
                                if final_topic_name != previous_topic['name']:
                                    rows_updated = update_topic_name(previous_topic['name'], previous_topic['start_node_id'], final_topic_name)
                                    if config.get("debug"):
                                        typer.echo(f"   ✅ Updated topic name: '{previous_topic['name']}' → '{final_topic_name}' ({rows_updated} rows)")
                                
                                # Update the previous topic's end node
                                if config.get("debug"):
                                    typer.echo(f"   DEBUG: Closing topic '{final_topic_name}' at boundary {actual_boundary[:8]}")
                                update_topic_end_node(final_topic_name, previous_topic['start_node_id'], actual_boundary)
                                
                                # Queue the old topic for compression
                                queue_topic_for_compression(previous_topic['start_node_id'], actual_boundary, final_topic_name)
                                if config.get("debug"):
                                    typer.echo(f"   📦 Queued topic '{final_topic_name}' for compression")
                    else:
                        # No previous topics exist - this is the first topic change
                        # Create a topic for the initial conversation before this point
                        user_node = get_node(user_node_id)
                        if user_node and user_node.get('parent_id'):
                            parent_node_id = user_node['parent_id']
                            
                            # Analyze where the topic actually changed for the initial topic too
                            actual_boundary = parent_node_id  # Default to current boundary
                            
                            # Check if boundary analysis is enabled (default: True)
                            if config.get("analyze_topic_boundaries", True):
                                if config.get("debug"):
                                    typer.echo(f"\n🔍 DEBUG: Analyzing initial topic boundary...")
                                
                                # Get recent conversation history for analysis
                                full_ancestry = get_ancestry(user_node_id)
                                
                                # Use boundary analyzer to find actual transition
                                if config.get("use_llm_boundary_analysis", True):
                                    # Use LLM-based analysis
                                    boundary_result = analyze_topic_boundary(
                                        full_ancestry[-20:] if len(full_ancestry) > 20 else full_ancestry,
                                        user_node_id,
                                        config.get("topic_detection_model", "ollama/llama3")
                                    )
                                    
                                    if boundary_result:
                                        actual_boundary = boundary_result
                                        if config.get("debug"):
                                            typer.echo(f"   Found initial topic boundary: {actual_boundary}")
                                else:
                                    # Use heuristic-based analysis (no LLM)
                                    heuristic_boundary = find_transition_point_heuristic(
                                        full_ancestry[-20:] if len(full_ancestry) > 20 else full_ancestry
                                    )
                                    if heuristic_boundary:
                                        actual_boundary = heuristic_boundary
                                        if config.get("debug"):
                                            typer.echo(f"   Found heuristic boundary: {actual_boundary}")
                            
                            # Get all nodes from the beginning up to the actual boundary
                            get_ancestry(actual_boundary)
                            
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
                                    
                                    # Get the actual boundary node's ROWID
                                    c2.execute("SELECT ROWID FROM nodes WHERE id = ?", (actual_boundary,))
                                    boundary_row = c2.fetchone()
                                    
                                    if boundary_row and boundary_row[0] >= 3:  # Need at least a few nodes
                                        # Get all nodes from beginning up to actual boundary
                                        c2.execute('''
                                            SELECT id, short_id, role, content 
                                            FROM nodes 
                                            WHERE ROWID <= ?
                                            ORDER BY ROWID
                                        ''', (boundary_row[0],))
                                        
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
                                        
                                        if config.get("debug"):
                                            typer.echo(f"\n🔍 DEBUG: Creating topic for initial conversation:")
                                            typer.echo(f"   From node {first_user_short_id} to {actual_boundary}")
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
                                        
                                        # Store the initial topic - don't set end_node_id yet!
                                        store_topic(topic_name, first_user_node_id, None, 'initial')
                                        # Set as current topic
                                        self.set_current_topic(topic_name, first_user_node_id)
                                        typer.echo("")
                                        typer.secho(f"📌 Created topic for initial conversation: {topic_name}", fg=get_system_color())
                                        
                                        # Now close the initial topic at the actual boundary
                                        update_topic_end_node(topic_name, first_user_node_id, actual_boundary)
                                        
                                        # Queue for compression
                                        queue_topic_for_compression(first_user_node_id, actual_boundary, topic_name)
                
                # Always create new topic if topic_changed is True (moved outside the else block)
                if topic_changed:
                    # Create a new topic starting from this user message
                    # Use a placeholder name that will be updated when this topic ends
                    timestamp = int(time.time())
                    placeholder_topic_name = f"ongoing-{timestamp}"
                    
                    # Create the topic with placeholder name - keep it open!
                    store_topic(placeholder_topic_name, user_node_id, None, 'detected')
                    
                    # Set as current topic
                    self.set_current_topic(placeholder_topic_name, user_node_id)
                    
                    typer.echo("")
                    typer.secho(f"🔄 Topic changed", fg=get_system_color())
                    
                    # Note: We'll extract a proper name for this topic after a few messages
                    # This is tracked in self.current_topic
                elif config.get("automatic_topic_detection"):
                    # No topic change - extend the current topic if one exists (only if automatic detection is enabled)
                    current_topic = self.get_current_topic()
                    if current_topic:
                        topic_name, start_node_id = current_topic
                        # Verify this topic still exists and hasn't been closed
                        existing_topics = get_recent_topics(limit=100)
                        topic_exists = False
                        for t in existing_topics:
                            if t['name'] == topic_name and t['start_node_id'] == start_node_id:
                                topic_exists = True
                                # Check if this topic was already closed
                                if t.get('end_node_id') and t['end_node_id'] != assistant_node_id:
                                    if config.get("debug"):
                                        typer.echo(f"🔍 DEBUG: Current topic '{topic_name}' was already closed at {t['end_node_id']}, cannot extend")
                                    # Clear the stale current topic
                                    self.current_topic = None
                                    topic_exists = False
                                break
                        
                        if topic_exists:
                            # For ongoing topics, we don't update end_node_id - it should stay NULL
                            # The topic automatically includes all nodes from start until it's closed
                            if config.get("debug"):
                                typer.echo(f"🔍 DEBUG: Topic '{topic_name}' continues (ongoing)")
                            
                            # Check if this topic needs renaming (if it has a placeholder name)
                            if topic_name.startswith('ongoing-'):
                                # Count messages in this topic
                                from episodic.topics import TopicManager
                                tm = TopicManager()
                                user_messages = tm.count_user_messages_in_topic(start_node_id, None)
                                
                                # If we have enough messages, extract a proper name
                                if user_messages >= 2:  # Extract name after 2 user messages
                                    # Get the topic content
                                    topic_nodes = []
                                    ancestry = get_ancestry(assistant_node_id)
                                    
                                    # Collect nodes from topic start to current
                                    found_start = False
                                    for node in ancestry:
                                        if node['id'] == start_node_id:
                                            found_start = True
                                        if found_start:
                                            topic_nodes.append(node)
                                            if node['id'] == assistant_node_id:
                                                break
                                    
                                    if topic_nodes and len(topic_nodes) >= 4:  # At least 2 exchanges
                                        # Build segment and extract name
                                        segment = build_conversation_segment(topic_nodes, max_length=1500)
                                        
                                        if config.get("debug"):
                                            typer.echo(f"\n🔍 DEBUG: Auto-extracting name for topic '{topic_name}'")
                                            typer.echo(f"   Messages in topic: {user_messages}")
                                        
                                        topic_extracted, _ = extract_topic_ollama(segment)
                                        
                                        if topic_extracted and topic_extracted != topic_name:
                                            # Update the topic name
                                            rows = update_topic_name(topic_name, start_node_id, topic_extracted)
                                            if rows > 0:
                                                # Update our current topic reference
                                                self.set_current_topic(topic_extracted, start_node_id)
                                                if config.get("debug"):
                                                    typer.echo(f"   ✅ Auto-renamed topic: '{topic_name}' → '{topic_extracted}'")
                                            else:
                                                if config.get("debug"):
                                                    typer.echo(f"   ⚠️  Failed to rename topic")
                        else:
                            if config.get("debug"):
                                typer.echo(f"🔍 DEBUG: Current topic '{topic_name}' no longer exists or is closed")
                    elif config.get("automatic_topic_detection"):
                        # No topics exist yet and no topic change detected (only check if automatic detection is enabled)
                        if config.get("debug"):
                            typer.echo(f"🔍 DEBUG: No current topic set, checking if we need to create first topic...")
                        
                        # Check if ANY topics exist in the database
                        from episodic.db import get_connection
                        with get_connection() as conn:
                            c = conn.cursor()
                            c.execute("SELECT COUNT(*) FROM topics")
                            topic_count = c.fetchone()[0]
                        
                        # If no topics exist at all, check if we should create the first one
                        if topic_count == 0:
                            if config.get('debug', False):
                                typer.echo(f"   DEBUG: No topics exist, checking if we should create first topic...")
                                typer.echo(f"   DEBUG: user_node_id = {user_node_id}")
                                user_node = get_node(user_node_id)
                                if user_node:
                                    typer.echo(f"   DEBUG: user_node content = '{user_node.get('content', '')[:50]}...'")
                            if should_create_first_topic(user_node_id):
                                # Look back to find the first user node and create topic from conversation start
                                from episodic.db import get_connection
                                with get_connection() as conn2:
                                    c2 = conn2.cursor()
                                    # Find the very first user node
                                    c2.execute("""
                                        SELECT id, short_id FROM nodes 
                                        WHERE role = 'user' 
                                        ORDER BY ROWID 
                                        LIMIT 1
                                    """)
                                    first_row = c2.fetchone()
                                    
                                    if first_row:
                                        first_user_node_id, first_user_short_id = first_row
                                        
                                        # Get all nodes from start up to current assistant node
                                        c2.execute("""
                                            SELECT id, short_id, role, content 
                                            FROM nodes 
                                            WHERE ROWID <= (SELECT ROWID FROM nodes WHERE id = ?)
                                            ORDER BY ROWID
                                        """, (assistant_node_id,))
                                        
                                        nodes = []
                                        for node_row in c2.fetchall():
                                            nodes.append({
                                                'id': node_row[0],
                                                'short_id': node_row[1],
                                                'role': node_row[2],
                                                'content': node_row[3]
                                            })
                                        
                                        # Build segment from the entire conversation
                                        segment = build_conversation_segment(nodes, max_length=2000)
                                        
                                        # Only extract topic if we have enough content
                                        user_nodes = [n for n in nodes if n.get('role') == 'user']
                                        if len(user_nodes) >= config.get('first_topic_threshold', 3):
                                            with benchmark_operation("Topic Name Extraction"):
                                                topic_name, extract_cost_info = extract_topic_ollama(segment)
                                            
                                            if not topic_name:
                                                topic_name = "conversation"
                                            
                                            # Create the initial topic - keep it open (no end_node_id)
                                            store_topic(topic_name, first_user_node_id, None, 'initial')
                                            # Set as current topic
                                            self.set_current_topic(topic_name, first_user_node_id)
                                            
                                            # Get assistant node short_id for display
                                            assistant_node = get_node(assistant_node_id)
                                            assistant_short_id = assistant_node.get('short_id', assistant_node_id) if assistant_node else assistant_node_id
                                            
                                            typer.echo("")
                                            typer.secho(f"📌 Created initial topic: {topic_name} (from {first_user_short_id} to {assistant_short_id})", fg=get_system_color())
                                        else:
                                            if config.get('debug', False):
                                                typer.echo(f"   DEBUG: Only {len(user_nodes)} user messages, skipping initial topic creation")
                                
                                # Add extraction costs if any
                                if extract_cost_info:
                                    self.session_costs["total_input_tokens"] += extract_cost_info.get("input_tokens", 0)
                                    self.session_costs["total_output_tokens"] += extract_cost_info.get("output_tokens", 0)
                                    self.session_costs["total_tokens"] += extract_cost_info.get("total_tokens", 0)
                                    self.session_costs["total_cost_usd"] += extract_cost_info.get("cost_usd", 0.0)
                            else:
                                # Not enough messages to create initial topic yet
                                if config.get("debug"):
                                    typer.echo("🔍 DEBUG: Skipping initial topic creation - not enough messages yet")
                        else:
                            # Topics exist but no change detected - should have extended existing topic above
                            if config.get("debug"):
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