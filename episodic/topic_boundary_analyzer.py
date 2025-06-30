"""
Topic boundary analyzer - finds the actual transition point when a topic change is detected.

This module helps identify where a topic actually changed by analyzing recent messages
and finding the true transition point, which is often 2-4 messages before detection.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

import typer

from episodic.config import config
from episodic.llm import query_llm
from episodic.benchmark import benchmark_resource

logger = logging.getLogger(__name__)


def analyze_topic_boundary(
    recent_nodes: List[Dict[str, Any]], 
    detection_point: str,
    topic_model: str = None
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Analyze recent conversation to find where the topic actually changed.
    
    When a topic change is detected at message N, this function looks backwards
    to find where the transition actually occurred (often at N-2 or N-3).
    
    Args:
        recent_nodes: List of recent conversation nodes (oldest to newest)
        detection_point: Node ID where topic change was detected
        topic_model: Model to use for analysis (defaults to topic_detection_model)
        
    Returns:
        Tuple of (actual_boundary_node_id, transition_type, cost_info)
        - actual_boundary_node_id: Where to end the previous topic (None if no clear boundary)
        - transition_type: "user_initiated" or "assistant_initiated" 
        - cost_info: LLM cost information
    """
    if not recent_nodes or len(recent_nodes) < 4:
        # Not enough history to analyze
        return None, None, None
        
    # Find the detection point in the list
    detection_idx = None
    for i, node in enumerate(recent_nodes):
        if node['id'] == detection_point:
            detection_idx = i
            break
            
    if detection_idx is None or detection_idx < 3:
        # Detection point not found or too early in list
        return None, None, None
    
    # Get the last 6-8 messages before detection for analysis
    analysis_window = recent_nodes[max(0, detection_idx - 8):detection_idx + 1]
    
    # Build conversation context
    context_parts = []
    for i, node in enumerate(analysis_window):
        role = node.get('role', 'unknown')
        content = node.get('content', '').strip()
        node_id = node.get('id', '')
        short_id = node.get('short_id', '')
        
        if content:
            # Include node ID for reference
            context_parts.append(f"[{i}] {role} (id:{short_id}): {content}")
    
    context = "\n".join(context_parts)
    
    # Create prompt for boundary analysis
    prompt = f"""Analyze this conversation to find where the topic actually changed.

A topic change was detected at the last message, but the actual transition often occurs earlier.
Look for the FIRST message that introduces or shifts to the new topic.

Conversation (with message indices):
{context}

Identify:
1. The index of the FIRST message that introduces the new topic
2. Whether it was user-initiated or assistant-initiated

Examples of topic transitions:
- User asks about something completely different → user-initiated at that message
- Assistant finishes answering and user moves on → user-initiated at next user message
- Natural conclusion leads to new direction → look for first divergence

Respond with JSON:
{{
  "boundary_index": <number>,
  "transition_type": "user_initiated" or "assistant_initiated",
  "reasoning": "brief explanation"
}}"""

    if config.get("debug", False):
        typer.echo(f"\n🔍 DEBUG: Analyzing topic boundary")
        typer.echo(f"   Analysis window: {len(analysis_window)} messages")
        typer.echo(f"   Detection point index: {detection_idx}")
    
    # Use configured topic model
    if not topic_model:
        topic_model = config.get("topic_detection_model", "ollama/llama3")
    
    try:
        topic_params = config.get_model_params('topic', model=topic_model)
        
        with benchmark_resource("LLM Call", f"boundary analysis - {topic_model}"):
            response, cost_info = query_llm(
                prompt, 
                model=topic_model,
                **topic_params
            )
        
        if response:
            import json
            try:
                result = json.loads(response.strip())
                boundary_idx = result.get("boundary_index")
                transition_type = result.get("transition_type")
                reasoning = result.get("reasoning", "")
                
                if config.get("debug", False):
                    typer.echo(f"   Boundary analysis result: index={boundary_idx}, type={transition_type}")
                    typer.echo(f"   Reasoning: {reasoning}")
                
                if boundary_idx is not None and 0 <= boundary_idx < len(analysis_window):
                    # Get the actual node ID at this boundary
                    boundary_node = analysis_window[boundary_idx]
                    
                    # For topic boundaries, we typically want to end the previous topic
                    # at the message BEFORE the transition
                    if boundary_idx > 0:
                        # End previous topic at the node before the transition
                        end_node = analysis_window[boundary_idx - 1]
                        boundary_node_id = end_node['id']
                        
                        if config.get("debug", False):
                            typer.echo(f"   Setting boundary at node {end_node['short_id']} (before transition)")
                    else:
                        # Transition at very start of window
                        boundary_node_id = boundary_node['id']
                    
                    return boundary_node_id, transition_type, cost_info
                    
            except json.JSONDecodeError as e:
                if config.get("debug", False):
                    typer.echo(f"   Failed to parse boundary analysis: {e}")
                    typer.echo(f"   Raw response: {response}")
    
    except Exception as e:
        logger.warning(f"Topic boundary analysis error: {e}")
        if config.get("debug", False):
            typer.echo(f"⚠️  Boundary analysis error: {e}")
    
    return None, None, cost_info


def find_transition_point_heuristic(
    recent_nodes: List[Dict[str, Any]],
    detection_point: str
) -> Optional[str]:
    """
    Use heuristics to find likely topic transition point.
    
    This is a fallback method that doesn't require LLM calls.
    
    Args:
        recent_nodes: List of recent conversation nodes
        detection_point: Where topic change was detected
        
    Returns:
        Node ID where previous topic should end, or None
    """
    # Find detection point
    detection_idx = None
    for i, node in enumerate(recent_nodes):
        if node['id'] == detection_point:
            detection_idx = i
            break
    
    if detection_idx is None or detection_idx < 2:
        return None
    
    # Simple heuristic: Look for the last user message before detection
    # that seems to be asking something new
    for i in range(detection_idx - 1, max(0, detection_idx - 6), -1):
        node = recent_nodes[i]
        if node.get('role') == 'user':
            content = node.get('content', '').lower()
            
            # Check for transition indicators
            transition_phrases = [
                "let me ask", "different question", "another topic",
                "by the way", "changing subjects", "moving on",
                "can you help", "can you tell",
                "let's talk about", "tell me about",
                "switch to", "different subject", "new question"
            ]
            
            # More specific patterns that indicate topic change
            strong_indicators = [
                "let me ask about a different",
                "changing the subject",
                "on a different topic",
                "completely different",
                "unrelated question"
            ]
            
            # Check strong indicators first
            if any(phrase in content for phrase in strong_indicators):
                # Found likely transition - end topic at previous message
                if i > 0:
                    return recent_nodes[i - 1]['id']
                return node['id']
            
            # Check weaker indicators only if they're far enough from detection
            if detection_idx - i >= 2 and any(phrase in content for phrase in transition_phrases):
                # Found likely transition - end topic at previous message
                if i > 0:
                    return recent_nodes[i - 1]['id']
                return node['id']
    
    # Default: End topic 2 messages before detection
    # (one exchange before the detecting message)
    if detection_idx >= 2:
        return recent_nodes[detection_idx - 2]['id']
    
    return None