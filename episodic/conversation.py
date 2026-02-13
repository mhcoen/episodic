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
    get_recent_topics, update_topic_name, update_topic_end_node
)
# Lazy import _execute_llm_query to avoid loading litellm at startup
# from episodic.llm import _execute_llm_query
from episodic.configuration import (
    get_llm_color, get_system_color, get_success_color, get_error_color,
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
from episodic.unified_streaming import unified_stream_response


class ConversationManager:
    """Manages conversation flow and coordinates specialized components."""
    
    def __init__(self):
        """Initialize the ConversationManager."""
        self.current_node_id = None
        self.current_topic = None  # Track current topic (name, start_node_id)
        self.drift_calculator = None
        self.last_loaded_start_id = None  # Track start of last loaded conversation
        self.last_loaded_end_id = None    # Track end of last loaded conversation
        self.session_costs = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }

        # Reactivation state (for implicit topic reactivation)
        self.reactivation_cooldown_turns = 0
        self.last_reactivation_topic_start_node_id = None

        # Correction state (for conversational disambiguation)
        # When DISAMBIGUATE proceeds with best guess, store runner-ups here
        self.pending_correction = None  # Optional[CorrectionState]

        # Initialize handlers
        self.topic_handler = TopicHandler(self)
        self.context_builder = ContextBuilder()
    
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

    def _get_turn_idx(self) -> int:
        """Get current turn index (max node rowid)."""
        from episodic.db_connection import get_connection
        with get_connection() as conn:
            cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
            row = cursor.fetchone()
            return row[0] if row and row[0] else 0

    def add_nodes_to_current_topic(self, user_node_id: str, assistant_node_id: str) -> None:
        """
        Add user and assistant nodes to the current topic's membership set.
        
        Called after each exchange to maintain topic_nodes table for 
        topic-local context assembly.
        """
        if not self.current_topic:
            return
        
        topic_start_node_id = self.current_topic[1]
        
        try:
            from episodic.db_topic_nodes import add_node_to_topic
            
            # Add user node
            add_node_to_topic(topic_start_node_id, user_node_id, 'user')
            
            # Add assistant node
            add_node_to_topic(topic_start_node_id, assistant_node_id, 'assistant')
            
            if config.get("debug"):
                debug_print(f"Added nodes to topic '{self.current_topic[0]}'", indent=True)
        except Exception as e:
            if config.get("debug"):
                debug_print(f"Failed to add nodes to topic: {e}", indent=True)

    def probe_topic_reactivation(
        self,
        user_input: str,
        recent_nodes: list,
        is_meta_query: bool = False,
        is_recall_intent: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Probe for implicit topic reactivation.

        Args:
            user_input: The user's message text
            recent_nodes: Recent conversation nodes
            is_meta_query: Whether this is a meta/command query
            is_recall_intent: Whether this is a memory recall query

        Returns:
            Tuple of (should_reactivate, topic_name, topic_start_node_id)
        """
        from datetime import datetime
        import numpy as np

        # Skip probe for meta queries, recall intents, or very short inputs
        if is_meta_query or is_recall_intent or len(user_input.split()) < 4:
            return False, None, None

        # Decrement cooldown
        if self.reactivation_cooldown_turns > 0:
            self.reactivation_cooldown_turns -= 1

        # Get active topic
        active_topic = self.get_current_topic()
        active_start_node_id = active_topic[1] if active_topic else None

        try:
            from episodic.recall.reactivation import probe_reactivation
            from episodic.rag_collections import get_multi_collection_rag, CollectionType

            # Get embedding for user input
            rag = get_multi_collection_rag()
            collection = rag.get_collection(CollectionType.CONVERSATION)
            embeddings = collection._embedding_function([user_input])
            user_embedding = np.array(embeddings[0])

            # Probe for reactivation
            decision = probe_reactivation(
                user_input=user_input,
                user_embedding=user_embedding,
                active_topic_start_node_id=active_start_node_id,
                cooldown_turns=self.reactivation_cooldown_turns,
                now=datetime.now(),
                recent_nodes=recent_nodes
            )

            # Store the decision on self for later persistence
            # (will be persisted once we have the user_node_id)
            self._last_reactivation_decision = decision

            # Gate reactivation to self
            if (decision.action == "REACTIVATE" and
                decision.topic_start_node_id == active_start_node_id):
                return False, None, None

            # Handle DISAMBIGUATE with best-guess-then-correction approach
            if decision.action == "DISAMBIGUATE" and decision.options:
                from episodic.recall.correction import CorrectionState

                # Store runner-ups for potential correction on next turn
                best_option = decision.options[0]
                runner_ups = decision.options[1:] if len(decision.options) > 1 else []

                if runner_ups:
                    self.pending_correction = CorrectionState(
                        query=user_input,
                        chosen_option=best_option,
                        runner_ups=runner_ups,
                        turn_created=self._get_turn_idx(),
                    )

                # Proceed with best option (conversational flow)
                decision.action = "REACTIVATE"
                decision.topic_name = best_option.topic_name
                decision.topic_start_node_id = best_option.topic_start_node_id
                decision.debug["disambiguation_choice"] = "auto_best"
                decision.debug["correction_state_stored"] = bool(runner_ups)
                return True, best_option.topic_name, best_option.topic_start_node_id

            if decision.action == "REACTIVATE":
                return True, decision.topic_name, decision.topic_start_node_id

            return False, None, None

        except Exception as e:
            debug_print(f"Reactivation probe error: {e}", category="memory")
            return False, None, None

    def apply_topic_reactivation(
        self,
        topic_name: str,
        topic_start_node_id: str,
        user_input: str
    ) -> Optional[str]:
        """
        Apply topic reactivation: switch topic and assemble context packet.

        Args:
            topic_name: Name of topic to reactivate
            topic_start_node_id: Start node ID of topic
            user_input: User's input (for anchor selection)

        Returns:
            Context packet string to inject, or None on error
        """
        import numpy as np

        try:
            from episodic.recall.reactivation import assemble_reactivation_packet
            from episodic.rag_collections import get_multi_collection_rag, CollectionType

            # Switch to reactivated topic
            self.set_current_topic(topic_name, topic_start_node_id)

            # Get embedding for anchor selection
            rag = get_multi_collection_rag()
            collection = rag.get_collection(CollectionType.CONVERSATION)
            embeddings = collection._embedding_function([user_input])
            user_embedding = np.array(embeddings[0])

            # Assemble context packet
            packet, debug_info = assemble_reactivation_packet(
                topic_start_node_id=topic_start_node_id,
                user_embedding=user_embedding,
                token_budget=150
            )

            # Set cooldown
            self.reactivation_cooldown_turns = 3
            self.last_reactivation_topic_start_node_id = topic_start_node_id

            if config.get("debug"):
                debug_print(f"Reactivated topic: {topic_name}")
                debug_print(f"Packet length: {len(packet)} chars")

            return packet if packet else None

        except Exception as e:
            debug_print(f"Reactivation apply error: {e}", category="memory")
            return None

    def finalize_current_topic(self) -> None:
        """
        Finalize the current topic by:
        1. Closing it if still open
        2. Giving it a proper name if it has a placeholder name
        This is called when the conversation ends or when explicitly requested.
        """
        # Only finalize topics if automatic topic detection is enabled
        if not config.get("automatic_topic_detection"):
            return
            
        # If current_node_id is not set, try to get it from the database
        if not self.current_node_id:
            self.current_node_id = get_head()
            
        # Get all topics to find open ones
        all_topics = get_recent_topics(limit=100)
        if not all_topics:
            return
            
        # Find any topics that are still open (no end_node_id)
        open_topics = [t for t in all_topics if not t.get('end_node_id')]
        
        if not open_topics:
            return  # No open topics to finalize
            
        for current_topic in open_topics:
            # Close the topic at current head
            if self.current_node_id:
                update_topic_end_node(
                    current_topic['name'], 
                    current_topic['start_node_id'], 
                    self.current_node_id
                )
                
                if config.get("debug"):
                    debug_print(f"Closed open topic '{current_topic['name']}' at session end")
            
            # Check if it needs a proper name
            if not current_topic['name'].startswith('ongoing-'):
                continue  # Already has a proper name
            
            # Extract topic name from the conversation
            if config.get("debug"):
                typer.echo("")
                debug_print(f"Finalizing topic '{current_topic['name']}'")
                
            # Get nodes in the topic
            topic_nodes = []
            # Use the end_node_id we just set, or current head
            end_node = current_topic.get('end_node_id') or self.current_node_id
            if end_node:
                ancestry = get_ancestry(end_node)
            else:
                ancestry = []
            
            if ancestry:
                # Collect nodes from topic start to end
                found_start = False
                for node in ancestry:
                    if node['id'] == current_topic['start_node_id']:
                        found_start = True
                    if found_start:
                        topic_nodes.append(node)
                    if node['id'] == end_node:
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
                        secho_color(f"   ✅ Finalized topic: '{current_topic['name']}' → '{topic_name}' ({rows_updated} rows)", fg=get_success_color(), bold=True)
                        
                    # Update current topic reference if this was the current topic
                    if self.current_topic and self.current_topic[0] == current_topic['name']:
                        self.set_current_topic(topic_name, self.current_topic[1])
    
    def initialize_conversation(self) -> None:
        """Initialize the conversation state from the database."""
        # Ensure startup-safe schema upgrades on existing installations.
        from episodic.db_migrations import ensure_runtime_schema
        ensure_runtime_schema()

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
        # Check if drift detection is disabled in config (check every time for runtime changes)
        if not config.get("show_drift"):
            return None

        # Get current embedding settings from config
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")

        # Check if settings changed - recreate if so
        if (self.drift_calculator is not None and
            self.drift_calculator is not False and
            (getattr(self.drift_calculator, '_embedding_provider', None) != embedding_provider or
             getattr(self.drift_calculator, '_embedding_model', None) != embedding_model)):
            self.drift_calculator = None  # Force recreation

        if self.drift_calculator is None or self.drift_calculator is False:
            try:
                self.drift_calculator = ConversationalDrift(
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model
                )
                # Store settings for change detection
                self.drift_calculator._embedding_provider = embedding_provider
                self.drift_calculator._embedding_model = embedding_model
                if config.get("debug"):
                    typer.echo(f"✅ Initialized drift calculator with {embedding_provider}/{embedding_model}")
            except Exception as e:
                # If drift calculator fails to initialize (e.g., missing dependencies),
                # disable drift detection for this session
                if config.get("debug"):
                    typer.echo(f"⚠️  Drift detection disabled: {e}")
                self.drift_calculator = False  # Mark as disabled
        return self.drift_calculator if self.drift_calculator is not False else None

    def compute_semantic_drift(self, current_user_node_id: str) -> Optional[float]:
        """
        Compute semantic drift between current and previous user message.

        Returns:
            Drift score (0.0-1.0) or None if not computable (e.g., < 2 user messages)
        """
        calc = self.get_drift_calculator()
        if not calc:
            return None

        try:
            # Get conversation history from root to current node
            conversation_chain = get_ancestry(current_user_node_id)

            # Filter to user messages only
            user_messages = [node for node in conversation_chain
                            if node.get("role") == "user" and node.get("content", "").strip()]

            # Need at least 2 user messages for comparison
            if len(user_messages) < 2:
                return None

            previous_user = user_messages[-2]
            current_user = user_messages[-1]

            return calc.calculate_drift(previous_user, current_user, text_field="content")
        except Exception as e:
            if config.get("debug"):
                typer.echo(f"⚠️  Drift computation error: {e}")
            return None

    def display_semantic_drift(
        self,
        current_user_node_id: str,
        cached_drift: Optional[float] = None
    ) -> None:
        """
        Display semantic drift between consecutive user messages.

        Args:
            current_user_node_id: The current user node ID
            cached_drift: Pre-computed drift score (avoids recomputation)
        """
        try:
            # Use cached drift if provided, otherwise compute
            if cached_drift is not None:
                drift_score = cached_drift
            else:
                drift_score = self.compute_semantic_drift(current_user_node_id)
                if drift_score is None:
                    return  # Not enough data for drift

            # Get previous user info for display (need ancestry for short_id)
            conversation_chain = get_ancestry(current_user_node_id)
            user_messages = [node for node in conversation_chain
                            if node.get("role") == "user" and node.get("content", "").strip()]
            if len(user_messages) < 2:
                return
            previous_user = user_messages[-2]
            current_user = user_messages[-1]
            
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
            
            # Display drift information (subtle diagnostic)
            from episodic.configuration import get_drift_color
            prev_short_id = previous_user.get("short_id", "??")
            secho_color(f"\n{drift_emoji}  Semantic drift: {drift_score:.3f} ({drift_desc}) from user message {prev_short_id}", fg=get_drift_color(), dim=True)
            
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

        Delegates to phase functions in conversation_pipeline.py.

        Args:
            user_input: The user's chat message
            model: The LLM model to use
            system_message: The system prompt
            context_depth: Number of messages to include in context

        Returns:
            Tuple of (assistant_node_id, display_response)
        """
        from episodic.conversation_pipeline import (
            TurnContext,
            phase_setup,
            phase_correction_reactivation,
            phase_topic_detection,
            phase_skip_llm,
            phase_memory_enhancement,
            phase_context_assembly,
            phase_message_augmentation,
            phase_llm_query,
            phase_postprocessing,
        )

        ctx = TurnContext(
            user_input=user_input,
            model=model,
            system_message=system_message,
            context_depth=context_depth,
        )

        with benchmark_operation("Message Processing"):
            # Phase 1: Insert user node, compute drift
            phase_setup(self, ctx)

            # Phase 2: Correction and reactivation probing
            phase_correction_reactivation(self, ctx)

            # Phase 3: Topic detection
            phase_topic_detection(self, ctx)

            # Phase 4: Skip-LLM early return (testing/debug mode)
            phase_skip_llm(self, ctx)
            if ctx.early_return:
                return ctx.early_return_value

            # Phase 5: Memory enhancement via RAG
            phase_memory_enhancement(self, ctx)

            # Phase 6: Context assembly
            phase_context_assembly(self, ctx)

            # Phase 7: Message augmentation (memory, reactivation, persona, style, voice)
            phase_message_augmentation(self, ctx)

            # Phase 8: LLM query (muse synthesis or regular)
            phase_llm_query(self, ctx)
            if ctx.early_return:
                return ctx.early_return_value

            # Phase 9: Post-processing (store response, indexing, topics)
            phase_postprocessing(self, ctx)

            return ctx.assistant_node_id, ctx.display_response


# Background indexing helper for non-blocking memory storage
def _fire_and_forget_index(
    user_node: Dict,
    assistant_node: Dict,
    topic_start_node_id: Optional[str] = None
):
    """Schedule conversation indexing without blocking the main thread.

    Skips indexing for recall/referential queries to prevent memory pollution.
    Only indexes actual informational exchanges, not meta-queries like
    "what did we discuss about X?" which would pollute memory with
    hallucinated recall responses.

    Args:
        user_node: The user message node dict
        assistant_node: The assistant response node dict
        topic_start_node_id: Topic identifier for anchor retrieval filtering
    """
    import threading

    def _index_in_background():
        try:
            from episodic.rag_memory_sqlite import memory_rag
            from episodic.rag_memory_smart import detect_recall_intent

            user_content = user_node.get('content', '')

            # Check if this is a recall query - don't index meta-queries
            should_retrieve, confidence, reason = detect_recall_intent(user_content)
            if should_retrieve and confidence > 0.5:
                debug_print(
                    f"Skipping indexing for recall query (conf={confidence:.2f}): {user_content[:50]}...",
                    category="memory"
                )
                return

            # Only index non-recall exchanges (with topic_start_node_id for anchor filtering)
            memory_rag.index_exchange(user_node, assistant_node, topic_start_node_id)
            debug_print("Indexed conversation in ChromaDB", category="memory")
        except Exception as e:
            debug_print(f"Background indexing failed: {e}", category="memory")

    # Fire-and-forget in background thread
    thread = threading.Thread(target=_index_in_background, daemon=True)
    thread.start()


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
