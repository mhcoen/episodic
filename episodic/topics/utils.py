"""
Utility functions for topic management.

These functions are extracted from the original topics.py file and
provide common functionality used across the topic detection module.
"""

from typing import List, Dict, Any
from episodic.config import config
import typer


def build_conversation_segment(nodes: List[Dict[str, Any]], max_length: int = 500) -> str:
    """
    Build a conversation segment for topic extraction.
    
    Args:
        nodes: List of conversation nodes
        max_length: Maximum character length for the segment
        
    Returns:
        Formatted conversation segment
    """
    segment_parts = []
    current_length = 0
    
    if config.get("debug"):
        typer.echo(f"   Building segment from {len(nodes)} nodes (max_length={max_length})")
    
    # For topic extraction, we want to prioritize the beginning of the conversation
    # This gives better topic names that reflect what the conversation started about
    for node in nodes:  # Process in chronological order
        content = node.get("content", "").strip()
        role = node.get("role", "unknown")
        
        if content:
            part = f"{role}: {content}"
            # If adding this part would exceed max_length
            if current_length + len(part) > max_length:
                # If we haven't added anything yet, truncate this part
                if not segment_parts:
                    part = part[:max_length-3] + "..."
                    segment_parts.append(part)
                    if config.get("debug"):
                        typer.echo(f"   Truncated first part to fit max_length")
                else:
                    if config.get("debug"):
                        typer.echo(f"   Stopping - would exceed max_length")
                break
            else:
                segment_parts.append(part)
                current_length += len(part)
        else:
            if config.get("debug"):
                typer.echo(f"   Skipping node with empty content (role={role})")
    
    result = "\n".join(segment_parts)
    if config.get("debug"):
        typer.echo(f"   Final segment length: {len(result)} chars")
    
    return result


def is_node_in_topic_range(node_id: str, topic_start_id: str, topic_end_id: str) -> bool:
    """
    Check if a node is within a topic's range.
    
    Args:
        node_id: The node to check
        topic_start_id: Start node of the topic
        topic_end_id: End node of the topic
        
    Returns:
        True if node is within the topic range
    """
    from episodic.db import get_ancestry
    
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
    from episodic.db import get_ancestry, get_head
    
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


def _display_topic_evolution(current_node_id: str) -> None:
    """
    Display topic evolution showing current and previous topics.
    
    This is a legacy function kept for backward compatibility.
    Use TopicManager.display_topic_evolution() instead.
    
    Args:
        current_node_id: ID of the current conversation node
    """
    from .detector import topic_manager
    topic_manager.display_topic_evolution(current_node_id)