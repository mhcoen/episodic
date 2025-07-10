"""
Topic name extraction functionality.

This module handles extracting meaningful topic names from conversations.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple

import typer

from episodic.config import config
from episodic.llm import query_llm
from episodic.benchmark import benchmark_resource

# Set up logging
logger = logging.getLogger(__name__)


def extract_topic_ollama(conversation_segment: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Extract topic name from conversation segment using Ollama.
    
    Args:
        conversation_segment: Text containing recent conversation exchanges
        
    Returns:
        Tuple of (topic_name: Optional[str], cost_info: Optional[Dict])
    """
    try:
        prompt = f"""Identify the main topic of this conversation. Reply with ONLY the topic name (1-3 words, lowercase, use hyphens for spaces).

Examples:
- Conversation about movies and directors → movies
- Discussion of quantum physics concepts → quantum-physics
- Debugging code and performance → programming
- Talking about semantic drift → semantic-drift

Conversation:
{conversation_segment}

Topic name:"""

        # Get the topic detection model from config (default to ollama/llama3)
        topic_model = config.get("topic_detection_model", "ollama/llama3")
        
        # Use configured model for topic extraction
        if config.get("debug"):
            typer.echo(f"\n🔍 DEBUG: Topic extraction prompt:")
            typer.echo(f"   Model: {topic_model}")
            typer.echo(f"   Prompt preview: {prompt[:300]}...")
        
        topic_params = config.get_model_params('topic', model=topic_model)
        with benchmark_resource("LLM Call", f"topic extraction - {topic_model}"):
            response, cost_info = query_llm(prompt, model=topic_model, **topic_params)
        
        if response:
            topic = _clean_topic_name(response)
            if topic:
                return topic, cost_info
            
            if config.get("debug"):
                typer.echo(f"⚠️  Topic extraction failed or invalid: '{response}'")
            return None, cost_info
            
    except Exception as e:
        if config.get("debug"):
            typer.echo(f"⚠️  Topic extraction error: {e}")
        return None, None


def _clean_topic_name(response: str) -> Optional[str]:
    """Clean and validate extracted topic name."""
    # Debug: Show raw response
    if config.get("debug"):
        typer.echo(f"   DEBUG: Raw topic extraction response: '{response}'")
    
    # Clean and normalize the response
    topic = response.strip().lower()
    # Remove quotes if present
    topic = topic.strip('"\'')
    # Replace spaces with hyphens
    topic = topic.replace(' ', '-')
    # Remove any extra characters, keep only letters, numbers, hyphens
    topic = re.sub(r'[^a-z0-9-]', '', topic)
    # Remove leading and trailing dashes
    topic = topic.strip('-')
    
    # Validate the topic name
    if not topic or topic == "no-topic":
        return None
        
    # Check if the topic is too long (more than 5 words worth)
    if len(topic) > 50:
        if config.get("debug"):
            typer.echo(f"⚠️  Topic name too long ({len(topic)} chars): '{topic[:50]}...'")
        # Try to extract just the first few words
        parts = topic.split('-')[:3]
        topic = '-'.join(parts)
    
    # Additional validation - check if it looks like the model included extra text
    if any(phrase in topic for phrase in ['extract', 'topic', 'conversation', 'words', 'lowercase', 'hyphens']):
        if config.get("debug"):
            typer.echo(f"⚠️  Topic contains prompt keywords: '{topic}'")
        # Try to find the actual topic after common phrases
        for delimiter in [':', 'is', 'are', '-']:
            if delimiter in topic:
                parts = topic.split(delimiter)
                # Take the last part that doesn't contain prompt keywords
                for part in reversed(parts):
                    cleaned_part = part.strip('-')
                    if cleaned_part and not any(kw in cleaned_part for kw in ['extract', 'topic', 'conversation', 'words']):
                        topic = cleaned_part
                        break
    
    # Final length check
    if len(topic) > 30:
        topic = topic[:30].rsplit('-', 1)[0]  # Cut at last hyphen before 30 chars
    
    if topic and len(topic) >= 2:  # Minimum 2 characters
        if config.get("debug"):
            typer.echo(f"   DEBUG: Final topic name: '{topic}'")
        return topic
    
    return None


def build_conversation_segment(
    nodes: List[Dict[str, Any]], 
    max_length: int = 500
) -> str:
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