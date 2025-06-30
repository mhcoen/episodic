"""
Topic boundary detection and analysis functionality.

This module contains functions for analyzing and finding precise topic
boundaries within conversations.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import typer

from episodic.llm import query_llm
from episodic.config import config
from episodic.prompt_manager import PromptManager


def analyze_topic_boundary(
    previous_nodes: List[Dict[str, Any]], 
    detection_node_id: str,
    model: Optional[str] = None,
    use_llm: bool = True
) -> Optional[str]:
    """
    Analyze recent conversation to find the precise topic boundary.
    
    When a topic change is detected at a certain node, this function looks back
    at the previous conversation to determine where the topic actually changed.
    Often the actual transition happens a few messages before the detection point.
    
    Args:
        previous_nodes: List of nodes leading up to the detection point (chronological order)
        detection_node_id: The node ID where topic change was detected
        model: Optional model to use for analysis (defaults to topic detection model)
        use_llm: Whether to use LLM analysis (True) or heuristic fallback (False)
        
    Returns:
        Node ID where the topic boundary should be placed, or None if analysis fails
    """
    if not previous_nodes:
        return None
        
    # If disabled or no nodes to analyze
    if not config.get("analyze_topic_boundaries", True) or len(previous_nodes) < 2:
        return None
    
    # Check configuration for LLM vs heuristic
    if use_llm and config.get("use_llm_boundary_analysis", True):
        # Use LLM-based analysis
        boundary_node_id = _analyze_with_llm(previous_nodes, detection_node_id, model)
        
        # If LLM analysis fails, fall back to heuristic
        if not boundary_node_id:
            if config.get("debug"):
                typer.echo("   LLM boundary analysis failed, using heuristic fallback")
            boundary_node_id = find_transition_point_heuristic(previous_nodes)
            
        return boundary_node_id
    else:
        # Use heuristic-based analysis
        return find_transition_point_heuristic(previous_nodes)


def _analyze_with_llm(
    previous_nodes: List[Dict[str, Any]], 
    detection_node_id: str,
    model: Optional[str] = None
) -> Optional[str]:
    """
    Use LLM to analyze where the topic boundary should be placed.
    
    Args:
        previous_nodes: List of nodes to analyze
        detection_node_id: Where the change was detected
        model: Optional model override
        
    Returns:
        Node ID for the boundary, or None if analysis fails
    """
    # Build conversation context
    messages = []
    node_map = {}  # Map position to node ID
    
    for i, node in enumerate(previous_nodes):
        role = node.get('role', 'unknown')
        content = node.get('content', '').strip()
        short_id = node.get('short_id', f"node_{i}")
        
        if content:
            messages.append(f"[{i+1}] {role}: {content}")
            node_map[i+1] = node['id']
    
    if len(messages) < 2:
        return None
    
    # Load prompt template
    prompt_manager = PromptManager()
    prompt_template = prompt_manager.get("topic_boundary_analysis")
    
    if not prompt_template:
        # Fallback prompt
        prompt_template = """You are analyzing a conversation to find where a topic change occurred.

The following messages are shown in chronological order. A topic change was detected somewhere in this sequence.

Your task: Identify the message number where the NEW TOPIC BEGINS.

Messages:
{messages}

Look for:
- Abrupt subject changes
- Transition phrases like "By the way", "Speaking of", "On another note"
- Questions that introduce completely new subjects
- Clear shifts in domain (e.g., technical to personal, science to entertainment)

Reply with ONLY the message number where the new topic begins."""

    prompt = prompt_template.format(messages="\n".join(messages))
    
    # Use topic detection model by default
    if not model:
        model = config.get("topic_detection_model", "ollama/llama3")
    
    try:
        if config.get("debug"):
            typer.echo(f"\n🔍 DEBUG: Analyzing topic boundary with LLM")
            typer.echo(f"   Model: {model}")
            typer.echo(f"   Analyzing {len(messages)} messages")
        
        # Query LLM
        topic_params = config.get_model_params('topic', model=model)
        response, _ = query_llm(prompt, model=model, **topic_params)
        
        if response:
            # Extract number from response
            numbers = re.findall(r'\d+', response.strip())
            if numbers:
                position = int(numbers[0])
                
                if config.get("debug"):
                    typer.echo(f"   LLM identified position: {position}")
                
                # Validate position
                if 1 <= position <= len(messages):
                    boundary_node_id = node_map.get(position)
                    if boundary_node_id:
                        if config.get("debug"):
                            typer.echo(f"   Boundary node: {boundary_node_id}")
                        return boundary_node_id
    
    except Exception as e:
        if config.get("debug"):
            typer.echo(f"   LLM boundary analysis error: {e}")
    
    return None


def find_transition_point_heuristic(previous_nodes: List[Dict[str, Any]]) -> Optional[str]:
    """
    Use heuristics to find where a topic transition likely occurred.
    
    This is a fallback method that looks for:
    - Transition keywords/phrases
    - Questions that might introduce new topics
    - Long gaps in conversation
    
    Args:
        previous_nodes: List of recent nodes in chronological order
        
    Returns:
        Node ID where the transition likely occurred, or None
    """
    if not previous_nodes or len(previous_nodes) < 2:
        return None
    
    # Transition indicators
    transition_phrases = [
        "by the way", "speaking of", "on another note", "changing topics",
        "different question", "unrelated", "separate topic", "also",
        "oh", "btw", "anyway", "moving on", "let me ask",
        "can you help with", "i need help with", "how about"
    ]
    
    # Look for transition indicators (search in reverse to find the most recent)
    for i in range(len(previous_nodes) - 1, 0, -1):
        node = previous_nodes[i]
        content = node.get('content', '').lower()
        
        # Skip very short messages
        if len(content) < 10:
            continue
        
        # Check for transition phrases
        for phrase in transition_phrases:
            if phrase in content:
                if config.get("debug"):
                    typer.echo(f"   Found transition phrase '{phrase}' in node {node.get('short_id')}")
                return node['id']
        
        # Check if this is a question that might introduce a new topic
        if node.get('role') == 'user' and '?' in content:
            # Look for questions that seem to introduce new subjects
            question_starters = ['what', 'how', 'can you', 'could you', 'tell me about', 'explain']
            if any(content.strip().startswith(starter) for starter in question_starters):
                if config.get("debug"):
                    typer.echo(f"   Found potential topic-introducing question in node {node.get('short_id')}")
                # Check if this question is sufficiently different from previous context
                if i > 0:
                    prev_content = previous_nodes[i-1].get('content', '').lower()
                    # Simple check: if the question has little word overlap with previous message
                    prev_words = set(prev_content.split())
                    curr_words = set(content.split())
                    overlap = len(prev_words & curr_words) / max(len(curr_words), 1)
                    if overlap < 0.3:  # Less than 30% word overlap
                        return node['id']
    
    # If no clear transition found, default to a position 2-3 messages back
    # This handles cases where the transition is gradual
    default_position = max(len(previous_nodes) - 3, 0)
    if default_position < len(previous_nodes):
        return previous_nodes[default_position]['id']
    
    return None