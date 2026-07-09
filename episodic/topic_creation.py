"""TopicHandler creation/naming/centroid methods.

Mixin split out of topic_management.py; TopicHandler inherits it, so these run
on the instance (self.conversation_manager, self._message_count, ... resolve via
inheritance).
"""

import time
from typing import Optional, Dict, Any, Tuple, List

import typer
from episodic.color_utils import secho_color
from episodic.config import config
from episodic.configuration import get_system_color, get_topic_change_color
from episodic.db import (
    get_recent_topics, update_topic_end_node, store_topic,
    update_topic_name, get_ancestry, store_topic_detection_scores, get_connection
)
from episodic.topics import (
    extract_topic_ollama, should_create_first_topic,
    build_conversation_segment, log_topic_decision
)
from episodic.debug_utils import debug_print
from episodic.debug_system import debug_enabled
from episodic.benchmark import benchmark_operation
from episodic.recall.centroid import update_topic_centroid


class _TopicCreationMixin:
    """First-topic creation, ongoing-topic naming, threshold, and centroid update."""

    def check_and_create_first_topic(
        self,
        user_node_id: str,
        assistant_node_id: str
    ) -> None:
        """Check if we need to create the first topic in a conversation."""
        # No topics exist yet and no topic change detected
        if config.get("debug"):
            typer.echo(f"🔍 DEBUG: No current topic set, checking if we need to create first topic...")
        
        # Check if ANY topics exist in the database
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM topics")
            topic_count = c.fetchone()[0]
        
        # If no topics exist at all, check if we should create the first one
        if topic_count == 0:
            if config.get('debug', False):
                typer.echo(f"   DEBUG: No topics exist, checking if we should create first topic...")
                typer.echo(f"   DEBUG: user_node_id = {user_node_id}")
            if should_create_first_topic(user_node_id):
                # Look back to find the first user node and create topic from conversation start
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
                        
                        if len(nodes) >= 4:  # At least 2 complete exchanges
                            # Extract topic from initial conversation
                            segment = build_conversation_segment(nodes, max_length=1500)
                            topic_name, extract_cost_info = extract_topic_ollama(segment)
                            
                            # Use fallback if extraction failed
                            if not topic_name:
                                topic_name = "initial-conversation"
                            
                            # Store the initial topic - leave it OPEN (end_node_id = None)
                            # It will be closed by handle_topic_boundaries() when a
                            # topic change is detected
                            store_topic(topic_name, first_user_node_id, None, 'initial')
                            # Set as current topic so handle_topic_boundaries knows to close it
                            self.conversation_manager.set_current_topic(topic_name, first_user_node_id)

                            # Initialize topic_nodes and topic_working_set for initial topic
                            try:
                                from episodic.db_topic_nodes import (
                                    add_nodes_to_topic_range,
                                    ensure_topic_working_set
                                )
                                # Add all nodes from start to current assistant
                                add_nodes_to_topic_range(first_user_node_id, first_user_node_id, assistant_node_id)
                                # Ensure working set entry exists
                                ensure_topic_working_set(first_user_node_id, topic_name)
                                if config.get("debug"):
                                    debug_print(f"Initialized topic_nodes for initial topic", indent=True)
                            except Exception as e:
                                if config.get("debug"):
                                    debug_print(f"Failed to initialize topic membership tables: {e}", indent=True)

                            # Initialize centroid for initial topic
                            update_topic_centroid(first_user_node_id, force=True)

                            typer.echo("")
                            secho_color(f"📌 Created initial topic: {topic_name}", fg=get_topic_change_color())
            else:
                if config.get("debug"):
                    typer.echo(f"🔍 DEBUG: Not enough messages for first topic yet")
    
    def update_ongoing_topic_name(self, assistant_node_id: str) -> None:
        """Check if current topic needs renaming from placeholder."""
        current_topic = self.conversation_manager.get_current_topic()
        if not current_topic:
            return
            
        topic_name, start_node_id = current_topic
        
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
                        secho_color(f"\n🔍 DEBUG:", fg='yellow', bold=True, nl=False)
                        secho_color(f" Auto-extracting name for topic '{topic_name}'")
                        typer.echo(f"   Messages in topic: {user_messages}")
                    
                    topic_extracted, _ = extract_topic_ollama(segment)
                    
                    if topic_extracted and topic_extracted != topic_name:
                        # Update the topic name
                        rows = update_topic_name(topic_name, start_node_id, topic_extracted)
                        if rows > 0:
                            # Update our current topic reference
                            self.conversation_manager.set_current_topic(topic_extracted, start_node_id)
                            if config.get("debug"):
                                typer.echo(f"   ✅ Auto-renamed topic: '{topic_name}' → '{topic_extracted}'")
                        else:
                            if config.get("debug"):
                                typer.echo(f"   ⚠️  Failed to rename topic")
    
    def _calculate_effective_threshold(
        self,
        topic_cost_info: Optional[Dict[str, Any]],
        topic_count: int
    ) -> float:
        """Calculate the effective threshold based on topic count."""
        base_threshold = float(config.get("drift_threshold", 0.9))

        # For sliding window detection, use the threshold from the detector
        if topic_cost_info and topic_cost_info.get("method") == "sliding_window":
            return topic_cost_info.get("threshold_used", base_threshold)

        # For the first 2 topics, use half the threshold
        if topic_count < 2:
            return base_threshold / 2

        return base_threshold

    def update_current_topic_centroid(self) -> bool:
        """
        Update the centroid for the current topic.

        Should be called after each exchange is added to maintain
        topic centroid information for implicit reactivation.

        Returns:
            True if centroid was updated, False otherwise
        """
        current_topic = self.conversation_manager.get_current_topic()
        if not current_topic:
            return False

        topic_name, start_node_id = current_topic

        try:
            # Update centroid (will only recompute at checkpoints)
            updated = update_topic_centroid(start_node_id)
            if updated and config.get("debug"):
                debug_print(f"Updated centroid for topic '{topic_name}'", indent=True)
            return updated
        except Exception as e:
            if config.get("debug"):
                debug_print(f"Failed to update centroid: {e}", indent=True)
            return False