"""
Topic management functionality for Episodic.

This module handles topic detection, creation, and management
during conversations.
"""

import time
from typing import Optional, Dict, Any, Tuple, List

import typer
from episodic.color_utils import secho_color
from episodic.config import config
from episodic.configuration import get_system_color
from episodic.db import (
    get_recent_topics, update_topic_end_node, store_topic,
    update_topic_name, get_ancestry, store_topic_detection_scores, get_connection
)
from episodic.topics import (
    extract_topic_ollama, should_create_first_topic,
    build_conversation_segment
)
from episodic.topic_boundary_analyzer import analyze_topic_boundary
from episodic.debug_utils import debug_print
from episodic.benchmark import benchmark_operation


class TopicHandler:
    """Handles topic detection and management for conversations."""
    
    def __init__(self, conversation_manager):
        """Initialize with reference to conversation manager."""
        self.conversation_manager = conversation_manager
        
    def detect_and_handle_topic_change(
        self,
        recent_nodes: List[Dict[str, Any]],
        user_input: str,
        user_node_id: str
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Detect topic change and handle topic management.
        
        Returns:
            Tuple of (topic_changed, new_topic_name, topic_cost_info, topic_change_info)
        """
        topic_changed = False
        new_topic_name = None
        topic_cost_info = None
        topic_change_info = None
        
        # Check if automatic topic detection is enabled
        if not config.get("automatic_topic_detection"):
            if config.get("debug"):
                typer.echo("")
                debug_print("Automatic topic detection is disabled")
            return topic_changed, new_topic_name, topic_cost_info, topic_change_info
        
        if config.get("debug"):
            secho_color(f"\n🔍 DEBUG:", fg='yellow', bold=True, nl=False)
            secho_color(f" Topic detection check")
            debug_print(f"Recent nodes count: {len(recent_nodes) if recent_nodes else 0}", indent=True)
            debug_print(f"Current topic: {self.conversation_manager.current_topic}", indent=True)
            debug_print(f"Min messages before topic change: {config.get('min_messages_before_topic_change')}", indent=True)
        
        if recent_nodes and len(recent_nodes) >= 2:  # Need at least some history
            try:
                with benchmark_operation("Topic Detection"):
                    # Debug: show which detector will be used
                    if config.get("debug"):
                        debug_print(f"Detection config: hybrid={config.get('use_hybrid_topic_detection')}, sliding={config.get('use_sliding_window_detection')}", indent=True)
                    
                    # Use hybrid detection if enabled
                    if config.get("use_hybrid_topic_detection"):
                        if config.get("debug"):
                            debug_print("Using HYBRID detection", indent=True)
                        from episodic.topics.hybrid import HybridTopicDetector
                        detector = HybridTopicDetector()
                        topic_changed, new_topic_name, topic_cost_info = detector.detect_topic_change(
                            recent_nodes,
                            user_input,
                            current_topic=self.conversation_manager.current_topic
                        )
                    elif config.get("use_sliding_window_detection"):
                        if config.get("debug"):
                            debug_print("Using SLIDING WINDOW detection", indent=True)
                        # Use sliding window detection (3-3 windows)
                        from episodic.topics.realtime_windows import RealtimeWindowDetector
                        window_size = config.get("sliding_window_size", 3)
                        detector = RealtimeWindowDetector(window_size=window_size)
                        topic_changed, new_topic_name, topic_cost_info = detector.detect_topic_change(
                            recent_nodes,
                            user_input,
                            current_topic=self.conversation_manager.current_topic
                        )
                    else:
                        # Use standard LLM-based detection
                        from episodic.topics.detector import topic_manager
                        topic_changed, new_topic_name, topic_cost_info = topic_manager.detect_topic_change_separately(
                            recent_nodes, 
                            user_input,
                            current_topic=self.conversation_manager.current_topic
                        )
                    if config.get("debug"):
                        debug_print(f"Topic change detected: {topic_changed}", indent=True)
                        if topic_changed:
                            debug_print(f"New topic: {new_topic_name}", indent=True)
                            
                    # Store topic change info to display later
                    if topic_changed:
                        topic_change_info = {
                            'changed': True,
                            'detection_info': topic_cost_info
                        }
                        
            except Exception as e:
                if config.get("debug"):
                    secho_color(f"   ❌ Topic detection error: {e}", fg='red', bold=True)
                # Continue without topic detection on error
                topic_changed = False
        else:
            if config.get("debug"):
                secho_color("   ⚠️  Not enough history for topic detection", fg='yellow', bold=True)
                
        return topic_changed, new_topic_name, topic_cost_info, topic_change_info
    
    def store_topic_detection_scores(
        self,
        recent_nodes: List[Dict[str, Any]],
        user_node_id: str,
        topic_cost_info: Optional[Dict[str, Any]],
        topic_changed: bool
    ) -> None:
        """Store topic detection scores for debugging."""
        if not config.get("automatic_topic_detection") or not recent_nodes or len(recent_nodes) < 2:
            return
            
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
                            debug_print(f"Storing window for {user_node['short_id']}", indent=True)
                            debug_print(f"Window A messages: {[m.get('short_id', '?') for m in window_a_messages]}", indent=True)
                            debug_print(f"Start: {window_a_messages[0].get('short_id', '?')}, End: {window_a_messages[-1].get('short_id', '?')}", indent=True)
                        
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
                                topic_cost_info.get("combined_score", 0.0),
                                topic_cost_info.get("is_boundary", False),
                                topic_cost_info.get("transition_phrase"),
                                topic_cost_info.get("threshold_used", 0.9)
                            ))
                            conn.commit()
                            
                        if config.get("debug"):
                            typer.echo(f"   ✅ Stored window-based detection scores")
                            
            except Exception as e:
                if config.get("debug"):
                    typer.echo(f"   ⚠️ Failed to store window scores: {e}")
                    
        # Also store general detection scores
        try:
            from episodic.db import get_node as get_node_info
            user_node = get_node_info(user_node_id)
            
            # Get context information
            current_topic = self.conversation_manager.get_current_topic()
            topic_name = current_topic[0] if current_topic else None
            
            # Count user messages in current topic
            user_messages_in_topic = 0
            if current_topic:
                from episodic.topics import TopicManager
                tm = TopicManager()
                user_messages_in_topic = tm.count_user_messages_in_topic(current_topic[1], None)
            
            # Calculate effective threshold
            effective_threshold = self._calculate_effective_threshold(
                topic_cost_info,
                len(get_recent_topics(limit=10))
            )
            
            # Extract scores based on detection method
            drift_score = 0.0
            keyword_score = 0.0
            combined_score = 0.0
            detection_method = topic_cost_info.get("method", "unknown") if topic_cost_info else "unknown"
            
            if topic_cost_info:
                if detection_method == "sliding_window":
                    drift_score = topic_cost_info.get("drift_score", 0.0)
                    keyword_score = topic_cost_info.get("keyword_score", 0.0)
                    combined_score = topic_cost_info.get("combined_score", 0.0)
                elif detection_method == "hybrid":
                    debug_info = topic_cost_info
                    if "signals" in debug_info:
                        drift_score = debug_info["signals"].get("semantic_drift", 0.0)
                        keyword_score = max(
                            debug_info["signals"].get("keywords", {}).get("explicit_transition", 0.0),
                            debug_info["signals"].get("keywords", {}).get("domain_shift", 0.0)
                        )
                    combined_score = debug_info.get("final_score", 0.0)
                elif detection_method == "llm":
                    combined_score = 1.0 if topic_changed else 0.0
                
            # Store the scores
            scores_data = {
                'user_node_short_id': user_node.get('short_id', 'unknown'),
                'detection_method': detection_method,
                'current_topic': topic_name,
                'messages_in_topic': user_messages_in_topic,
                'drift_score': drift_score,
                'keyword_score': keyword_score,
                'combined_score': combined_score,
                'effective_threshold': effective_threshold,
                'topic_changed': topic_changed,
                'detection_response': topic_cost_info.get('detection_response') if topic_cost_info else None
            }
            
            store_topic_detection_scores(**scores_data)
            
            if config.get("debug"):
                typer.echo(f"   ✅ Stored topic detection scores: {detection_method}")
                
        except Exception as e:
            if config.get("debug"):
                typer.echo(f"   ⚠️ Failed to store detection scores: {e}")
    
    def handle_topic_boundaries(
        self,
        topic_changed: bool,
        user_node_id: str,
        assistant_node_id: str,
        topic_change_info: Optional[Dict[str, Any]],
        new_topic_name: Optional[str] = None
    ) -> None:
        """Handle topic boundary detection and management."""
        # Import at function level to ensure availability throughout
        from episodic.db import get_node, get_ancestry
        
        if topic_changed:
            # A topic change was detected - close any open topics
            # First check if we have a current topic in memory
            if self.conversation_manager.current_topic:
                topic_name, start_node_id = self.conversation_manager.current_topic
            else:
                # No current topic in memory - check database for open topics
                from episodic.db import get_recent_topics
                all_topics = get_recent_topics(limit=100)
                open_topics = [t for t in all_topics if not t.get('end_node_id')]
                
                if open_topics:
                    # Use the most recent open topic
                    current_db_topic = open_topics[-1]
                    topic_name = current_db_topic['name']
                    start_node_id = current_db_topic['start_node_id']
                    
                    if config.get("debug"):
                        debug_print(f"Found open topic in database: '{topic_name}'", indent=True)
                else:
                    # No open topics found - nothing to close
                    topic_name = None
                    start_node_id = None
            
            if topic_name and start_node_id:
                # Analyze where the actual topic boundary should be
                if config.get("analyze_topic_boundaries", True):
                    actual_boundary = analyze_topic_boundary(start_node_id, assistant_node_id, user_node_id)
                else:
                    # Use simple heuristic - topic ends at last assistant response before change
                    # If no assistant response exists, find the previous node
                    if assistant_node_id:
                        actual_boundary = assistant_node_id
                    else:
                        # Find the node before user_node_id to avoid overlap
                        ancestry = get_ancestry(user_node_id)
                        if len(ancestry) >= 2:
                            # Get the node just before user_node_id
                            actual_boundary = ancestry[-2]['id']
                        else:
                            # Edge case: this is the very first exchange
                            actual_boundary = None
                
                # Close the previous topic at the determined boundary
                if actual_boundary:
                    update_topic_end_node(topic_name, start_node_id, actual_boundary)
                else:
                    # Cannot determine a clean boundary - log warning
                    if config.get("debug"):
                        debug_print(f"Warning: Could not determine topic boundary for {topic_name}", indent=True)
                
                # Extract a proper name for the topic now that it's complete
                if topic_name.startswith('ongoing-'):
                    # Get nodes in the completed topic
                    topic_nodes = []
                    ancestry = get_ancestry(actual_boundary)
                    
                    # Collect nodes from topic start to boundary
                    found_start = False
                    for node in ancestry:
                        if node['id'] == start_node_id:
                            found_start = True
                        if found_start:
                            topic_nodes.append(node)
                        if node['id'] == actual_boundary:
                            break
                    
                    if topic_nodes and len(topic_nodes) >= 4:  # At least 2 exchanges
                        # Build segment and extract name
                        segment = build_conversation_segment(topic_nodes, max_length=1500)
                        
                        if config.get("debug"):
                            secho_color(f"\n🔍 DEBUG:", fg='yellow', bold=True, nl=False)
                            secho_color(f" Extracting name for completed topic '{topic_name}'")
                        
                        topic_extracted, extract_cost_info = extract_topic_ollama(segment)
                        
                        if topic_extracted and topic_extracted != topic_name:
                            # Update the topic name
                            rows = update_topic_name(topic_name, start_node_id, topic_extracted)
                            if rows > 0:
                                if config.get("debug"):
                                    typer.echo(f"   ✅ Renamed completed topic: '{topic_name}' → '{topic_extracted}'")
                                topic_name = topic_extracted
                            else:
                                if config.get("debug"):
                                    typer.echo(f"   ⚠️  Failed to rename topic")
                
                # Queue the closed topic for compression
                if config.get("auto_compress_topics", True) and actual_boundary:
                    from episodic.compression import queue_topic_for_compression
                    queue_topic_for_compression(start_node_id, actual_boundary, topic_name)
                
                # Clear current topic since it's closed
                self.conversation_manager.current_topic = None
        
        # Always create new topic if topic_changed is True
        if topic_changed:
            # Create a new topic starting from this user message
            # Use the detected topic name if available, otherwise use placeholder
            if new_topic_name and not new_topic_name.startswith('ongoing-'):
                # We have a proper name from detection
                topic_name_to_use = new_topic_name
            else:
                # Fallback to placeholder name that will be updated later
                timestamp = int(time.time())
                topic_name_to_use = f"ongoing-{timestamp}"
                if config.get("debug"):
                    debug_print(f"Warning: No topic name from detection, using placeholder: {topic_name_to_use}", indent=True)
            
            # Create the topic - keep it open!
            store_topic(topic_name_to_use, user_node_id, None, 'detected')
            
            # Set as current topic
            self.conversation_manager.set_current_topic(topic_name_to_use, user_node_id)
            
            if config.get("topic_change_info", True):
                typer.echo("")
                if new_topic_name and not new_topic_name.startswith('ongoing-'):
                    secho_color(f"📌 Topic changed to: {topic_name_to_use}", fg=get_system_color())
                else:
                    secho_color(f"📌 Topic change detected", fg=get_system_color())
    
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
                            
                            # Store the initial topic - don't set end_node_id yet!
                            store_topic(topic_name, first_user_node_id, None, 'initial')
                            # Set as current topic
                            self.conversation_manager.set_current_topic(topic_name, first_user_node_id)
                            
                            # Determine the actual boundary for the initial topic
                            if config.get("analyze_topic_boundaries", True):
                                actual_boundary = analyze_topic_boundary(
                                    first_user_node_id, 
                                    assistant_node_id, 
                                    user_node_id
                                )
                            else:
                                # Use simple heuristic - if no assistant response, use previous node
                                if assistant_node_id:
                                    actual_boundary = assistant_node_id
                                else:
                                    # Get the node before user_node_id
                                    ancestry = get_ancestry(user_node_id)
                                    if len(ancestry) >= 2:
                                        actual_boundary = ancestry[-2]['id']
                                    else:
                                        actual_boundary = None
                            
                            # Now close the initial topic at the actual boundary
                            if actual_boundary:
                                update_topic_end_node(topic_name, first_user_node_id, actual_boundary)
                            else:
                                if config.get("debug"):
                                    debug_print(f"Warning: Could not determine topic boundary for initial topic", indent=True)
                            
                            # Queue for compression
                            if actual_boundary:
                                from episodic.compression import queue_topic_for_compression
                                queue_topic_for_compression(first_user_node_id, actual_boundary, topic_name)
                            
                            typer.echo("")
                            secho_color(f"📌 Created initial topic: {topic_name}", fg=get_system_color())
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