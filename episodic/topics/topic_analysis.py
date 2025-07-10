"""
Topic analysis and counting functionality.

This module handles analyzing topics, counting messages, and checking boundaries.
"""

import logging

import typer

from episodic.config import config
from episodic.db import get_ancestry, get_head, get_recent_topics

# Set up logging
logger = logging.getLogger(__name__)


def should_create_first_topic(user_node_id: str) -> bool:
    """
    Check if we should proactively create the first topic.
    
    Returns True if:
    - No topics exist yet
    - We have at least 3-4 message exchanges (6-8 nodes total)
    
    Args:
        user_node_id: ID of the current user node
        
    Returns:
        True if we should create the first topic, False otherwise
    """
    # Check if any topics exist
    existing_topics = get_recent_topics(limit=1)
    if existing_topics:
        return False  # Topics already exist
    
    # Count the conversation depth
    try:
        # Get the full conversation chain to count exchanges
        conversation_chain = get_ancestry(user_node_id)
        
        # Count user messages (each user message represents an exchange)
        user_message_count = sum(1 for node in conversation_chain if node.get('role') == 'user')
        
        # We want at least 3 user messages (3 exchanges) before creating first topic
        # This can be configured via the 'first_topic_threshold' config option
        threshold = config.get('first_topic_threshold', 3)
        
        if config.get('debug'):
            logger.info(f"First topic check: {user_message_count} user messages, threshold: {threshold}")
            typer.echo(f"   DEBUG: should_create_first_topic: {user_message_count} user messages, threshold: {threshold}, returning: {user_message_count >= threshold}")
        
        return user_message_count >= threshold
        
    except Exception as e:
        logger.warning(f"Error checking conversation depth for first topic: {e}")
        return False


def is_node_in_topic_range(
    node_id: str, 
    topic_start_id: str, 
    topic_end_id: str
) -> bool:
    """
    Check if a node is within a topic's range.
    
    Args:
        node_id: The node to check
        topic_start_id: Start node of the topic
        topic_end_id: End node of the topic
        
    Returns:
        True if node is within the topic range
    """
    # Get ancestry of the end node to find all nodes in the topic
    topic_ancestry = get_ancestry(topic_end_id)
    
    # Check if our node is in this ancestry and after/at the start node
    found_start = False
    for node in topic_ancestry:
        if node['id'] == topic_start_id:
            found_start = True
        if found_start and node['id'] == node_id:
            return True
        if node['id'] == topic_end_id:
            # Reached the end, check if node_id is the end node
            return node_id == topic_end_id
    
    return False


def count_nodes_in_topic(topic_start_id: str, topic_end_id: str) -> int:
    """
    Count the number of nodes in a topic range.
    
    Args:
        topic_start_id: Start node of the topic
        topic_end_id: End node of the topic (can be None for ongoing topics)
        
    Returns:
        Number of nodes in the topic (including start and end)
    """
    # Handle ongoing topics (no end node)
    if not topic_end_id:
        # For ongoing topics, count from start_node to current head
        current_head = get_head()
        if not current_head:
            return 0
        topic_end_id = current_head
        
    # Get ancestry of the end node
    topic_ancestry = get_ancestry(topic_end_id)
    
    # Debug: Check if start_id exists in ancestry
    ancestry_ids = [node['id'] for node in topic_ancestry]
    if config.get("debug") and topic_start_id not in ancestry_ids:
        typer.echo(f"\n🔍 DEBUG: Topic boundary issue detected!")
        typer.echo(f"   Start node {topic_start_id} not found in ancestry of end node {topic_end_id}")
        typer.echo(f"   Ancestry chain has {len(topic_ancestry)} nodes")
        if len(topic_ancestry) > 0:
            typer.echo(f"   First in chain: {topic_ancestry[-1]['short_id']} ({topic_ancestry[-1]['id']})")
            typer.echo(f"   Last in chain: {topic_ancestry[0]['short_id']} ({topic_ancestry[0]['id']})")
    
    # Count nodes from start to end
    count = 0
    found_start = False
    for node in topic_ancestry:
        if node['id'] == topic_start_id:
            found_start = True
        if found_start:
            count += 1
        if node['id'] == topic_end_id:
            break
    
    # If we didn't find the start node, it might be due to branching or broken chains
    # In this case, count nodes from the end node backwards
    if not found_start and count == 0:
        # Count all nodes in the ancestry up to a reasonable limit
        # This handles broken chains gracefully
        count = min(len(topic_ancestry), 20)  # Cap at 20 to avoid counting entire history
        if config.get("debug"):
            typer.echo(f"   ⚠️  Using fallback count due to broken chain: {count} nodes")
    
    return max(count, 2)  # Ensure minimum of 2


def count_user_messages_in_topic(topic_start_id: str, topic_end_id: str) -> int:
    """
    Count the number of USER messages in a topic range.
    
    Args:
        topic_start_id: Start node of the topic
        topic_end_id: End node of the topic (None for ongoing topics)
        
    Returns:
        Number of user messages in the topic
    """
    # Handle ongoing topics (no end node)
    if not topic_end_id:
        # For ongoing topics, count from start to current head
        current_head = get_head()
        if not current_head:
            return 0
        topic_end_id = current_head
        
    # Get ancestry of the end node
    topic_ancestry = get_ancestry(topic_end_id)
    
    # Count only user messages from start to end
    count = 0
    found_start = False
    for node in topic_ancestry:
        if node['id'] == topic_start_id:
            found_start = True
        if found_start and node.get('role') == 'user':
            count += 1
        if node['id'] == topic_end_id:
            break
    
    # If we didn't find the start node, count all user messages in ancestry
    if not found_start and count == 0:
        # Count user messages in the ancestry up to a reasonable limit
        user_count = sum(1 for node in topic_ancestry[:20] if node.get('role') == 'user')
        count = user_count
        if config.get("debug"):
            typer.echo(f"   ⚠️  Using fallback count due to broken chain: {count} user messages")
    
    return count


def display_topic_evolution(current_node_id: str) -> None:
    """
    Display topic evolution showing current and previous topics.
    
    Args:
        current_node_id: ID of the current conversation node
    """
    try:
        # Get recent topics to show evolution
        recent_topics = get_recent_topics(limit=5)
        
        if len(recent_topics) < 2:
            # Not enough topic history to show evolution
            return
        
        # Find the current topic that contains the current node
        current_topic = None
        previous_topic = None
        
        # Look through topics to find which one we're currently in
        for i, topic in enumerate(recent_topics):
            # Check if current node is within this topic's range
            # A node is in a topic if it's a descendant of the start node
            # and an ancestor of or equal to the end node
            if is_node_in_topic_range(current_node_id, topic['start_node_id'], topic['end_node_id']):
                current_topic = topic
                # Get the previous topic if available
                if i + 1 < len(recent_topics):
                    previous_topic = recent_topics[i + 1]
                break
        
        # If we didn't find a current topic, use the most recent one
        if not current_topic and recent_topics:
            current_topic = recent_topics[0]
            if len(recent_topics) > 1:
                previous_topic = recent_topics[1]
        
        # Format the evolution display
        if current_topic and previous_topic:
            prev_name = previous_topic['name']
            curr_name = current_topic['name']
            
            # Only show if topics are actually different
            if prev_name != curr_name:
                # Topic evolution display disabled - provides little value since topics are named retroactively
                pass
                # typer.echo(f"\nTopic evolution: {prev_name} (completed) → {curr_name} (active)")
            
    except Exception as e:
        if config.get("debug"):
            typer.echo(f"⚠️  Topic evolution display error: {e}")