"""
Topic detection and management functionality for Episodic.

This module handles all topic-related operations including:
- Detecting topic changes in conversations
- Extracting topic names from conversation segments
- Managing topic boundaries and evolution
- First topic creation logic
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple

import typer

from episodic.db import (
    get_recent_topics, get_node, get_ancestry, 
    store_topic, update_topic_end_node
)
from episodic.llm import query_llm
from episodic.config import config
from episodic.prompt_manager import PromptManager
from episodic.compression import queue_topic_for_compression

# Set up logging
logger = logging.getLogger(__name__)


class TopicManager:
    """Manages topic detection and tracking in conversations."""
    
    def __init__(self):
        """Initialize the TopicManager."""
        self.prompt_manager = PromptManager()
    
    def detect_topic_change_separately(
        self, 
        recent_messages: List[Dict[str, Any]], 
        new_message: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if the topic has changed by analyzing recent messages and the new message.
        
        This function runs separately from the main conversation flow, using a focused
        LLM call to determine if the topic has shifted.
        
        Args:
            recent_messages: List of recent conversation nodes (3-5 messages)
            new_message: The new user message to analyze
            
        Returns:
            Tuple of (topic_changed: bool, new_topic_name: Optional[str])
        """
        try:
            # Check if we should skip topic detection due to recency
            recent_topics = get_recent_topics(limit=1)
            if recent_topics:
                current_topic = recent_topics[0]
                # Use node count if available, otherwise count messages
                if 'node_count' in current_topic:
                    messages_in_topic = current_topic['node_count']
                else:
                    # Count nodes between start and end
                    messages_in_topic = self.count_nodes_in_topic(
                        current_topic['start_node_id'], 
                        current_topic['end_node_id']
                    )
                
                # Skip topic detection if current topic has fewer than 6 messages (3 exchanges)
                min_messages_before_change = config.get('min_messages_before_topic_change', 6)
                if messages_in_topic < min_messages_before_change:
                    if config.get("debug", False):
                        typer.echo(f"\n🔍 DEBUG: Skipping topic detection - current topic has only {messages_in_topic} messages (min: {min_messages_before_change})")
                    return False, None
            # Build context from recent messages
            context_parts = []
            for msg in recent_messages[-6:]:  # Last 3 exchanges (6 messages)
                role = msg.get('role', 'unknown')
                content = msg.get('content', '').strip()
                if content:
                    # Truncate long messages for context
                    truncated = content[:200] + '...' if len(content) > 200 else content
                    context_parts.append(f"{role}: {truncated}")
            
            context = "\n".join(context_parts)
            
            # Load topic detection prompt
            topic_detection_prompt = self.prompt_manager.get_prompt("topic_detection")
            
            if topic_detection_prompt:
                # Use the loaded prompt template
                prompt_template = topic_detection_prompt['content']
                prompt = prompt_template.format(
                    recent_conversation=context,
                    new_message=f"user: {new_message}"
                )
            else:
                # Fallback to default prompt if file not found
                prompt = f"""Analyze if there is a MAJOR topic change in this conversation.

IMPORTANT: Only detect changes when switching between COMPLETELY DIFFERENT subjects.
Minor variations within the same general subject are NOT topic changes.

Previous conversation:
{context}

New user message:
user: {new_message}

Has the topic changed SIGNIFICANTLY? Answer with ONLY:
- "YES: [new-topic-name]" if there is a MAJOR topic change (use 1-3 words, lowercase with hyphens)
- "NO" if continuing the same general subject area

Examples of NO CHANGE (continuing same topic):
- "What color is the sky?" → "What color are roses?" (both about colors)
- "What is 2+2?" → "What is 10*10?" (both about math)
- "How do I debug Python?" → "What about performance optimization?" (both about programming)
- "Tell me about dogs" → "What about cats?" (both about animals)
- "Explain quantum physics" → "What about relativity?" (both about physics)

Examples of YES CHANGE (major topic shift):
- "How do I debug Python?" → "What's the weather today?" → "YES: weather"
- "Tell me about quantum physics" → "What's a good restaurant nearby?" → "YES: restaurants"
- "What is 2+2?" → "Tell me about ancient Rome" → "YES: ancient-rome"

Answer:"""

            if config.get("debug", False):
                typer.echo(f"\n🔍 DEBUG: Topic change detection")
                typer.echo(f"   Model: ollama/llama3")
                typer.echo(f"   Recent messages: {len(recent_messages)}")
                typer.echo(f"   New message preview: {new_message[:100]}...")
            
            # Use ollama for fast detection
            response, _ = query_llm(
                prompt, 
                model="ollama/llama3",
                max_tokens=20  # Very short response expected
            )
            
            if response:
                response = response.strip()
                
                if config.get("debug", False):
                    typer.echo(f"   LLM response: {response}")
                
                if response.upper().startswith("YES:"):
                    # Extract the new topic name
                    topic_part = response[4:].strip()
                    # Clean the topic name
                    topic = topic_part.lower().strip()
                    topic = topic.strip('"\'')
                    topic = topic.replace(' ', '-')
                    topic = re.sub(r'[^a-z0-9-]', '', topic)
                    
                    if topic:
                        if config.get("debug", False):
                            typer.echo(f"   ✅ Topic changed to: {topic}")
                        return True, topic
                    else:
                        if config.get("debug", False):
                            typer.echo(f"   ⚠️ Topic changed but couldn't extract name")
                        return True, None
                else:
                    if config.get("debug", False):
                        typer.echo(f"   ➡️ Continuing same topic")
                    return False, None
        
        except Exception as e:
            logger.warning(f"Topic change detection error: {e}")
            if config.get("debug", False):
                typer.echo(f"⚠️  Topic detection error: {e}")
            return False, None
    
    def extract_topic_ollama(self, conversation_segment: str) -> Optional[str]:
        """
        Extract topic name from conversation segment using Ollama.
        
        Args:
            conversation_segment: Text containing recent conversation exchanges
            
        Returns:
            Topic name as lowercase string with hyphens, or None if extraction fails
        """
        try:
            prompt = f"""Extract the main topic from this conversation in 1-3 words. Use lowercase with hyphens.

Examples:
- Conversation about movies and directors → "movies"
- Discussion of quantum physics concepts → "quantum-physics" 
- Debugging code and performance → "programming"
- Talking about semantic drift → "semantic-drift"

Conversation: {conversation_segment}

Topic:"""

            # Use ollama for silent topic extraction
            if config.get("debug", False):
                typer.echo(f"\n🔍 DEBUG: Topic extraction prompt:")
                typer.echo(f"   Model: ollama/llama3")
                typer.echo(f"   Prompt preview: {prompt[:300]}...")
            
            response, _ = query_llm(prompt, model="ollama/llama3")
            
            if response:
                # Clean and normalize the response
                topic = response.strip().lower()
                # Remove quotes if present
                topic = topic.strip('"\'')
                # Replace spaces with hyphens
                topic = topic.replace(' ', '-')
                # Remove any extra characters, keep only letters, numbers, hyphens
                topic = re.sub(r'[^a-z0-9-]', '', topic)
                
                return topic if topic else None
                
        except Exception as e:
            if config.get("debug", False):
                typer.echo(f"⚠️  Topic extraction error: {e}")
            return None
    
    def should_create_first_topic(self, user_node_id: str) -> bool:
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
            return user_message_count >= threshold
            
        except Exception as e:
            logger.warning(f"Error checking conversation depth for first topic: {e}")
            return False
    
    def build_conversation_segment(
        self, 
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
        
        for node in reversed(nodes):  # Start from most recent
            content = node.get("content", "").strip()
            role = node.get("role", "unknown")
            
            if content:
                part = f"{role}: {content}"
                if current_length + len(part) > max_length:
                    break
                segment_parts.insert(0, part)  # Insert at beginning to maintain order
                current_length += len(part)
        
        return "\n".join(segment_parts)
    
    def is_node_in_topic_range(
        self, 
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
    
    def count_nodes_in_topic(self, topic_start_id: str, topic_end_id: str) -> int:
        """
        Count the number of nodes in a topic range.
        
        Args:
            topic_start_id: Start node of the topic
            topic_end_id: End node of the topic
            
        Returns:
            Number of nodes in the topic (including start and end)
        """
        # Get ancestry of the end node
        topic_ancestry = get_ancestry(topic_end_id)
        
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
        
        return count
    
    def display_topic_evolution(self, current_node_id: str) -> None:
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
                if self.is_node_in_topic_range(current_node_id, topic['start_node_id'], topic['end_node_id']):
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
                    typer.echo(f"\nTopic evolution: {prev_name} (completed) → {curr_name} (active)")
                
        except Exception as e:
            if config.get("debug", False):
                typer.echo(f"⚠️  Topic evolution display error: {e}")


# Create a global instance for convenience
topic_manager = TopicManager()


# Expose the main functions at module level for backward compatibility
def detect_topic_change_separately(recent_messages: List[Dict[str, Any]], new_message: str) -> Tuple[bool, Optional[str]]:
    """See TopicManager.detect_topic_change_separately for documentation."""
    return topic_manager.detect_topic_change_separately(recent_messages, new_message)


def extract_topic_ollama(conversation_segment: str) -> Optional[str]:
    """See TopicManager.extract_topic_ollama for documentation."""
    return topic_manager.extract_topic_ollama(conversation_segment)


def should_create_first_topic(user_node_id: str) -> bool:
    """See TopicManager.should_create_first_topic for documentation."""
    return topic_manager.should_create_first_topic(user_node_id)


def build_conversation_segment(nodes: List[Dict[str, Any]], max_length: int = 500) -> str:
    """See TopicManager.build_conversation_segment for documentation."""
    return topic_manager.build_conversation_segment(nodes, max_length)


def is_node_in_topic_range(node_id: str, topic_start_id: str, topic_end_id: str) -> bool:
    """See TopicManager.is_node_in_topic_range for documentation."""
    return topic_manager.is_node_in_topic_range(node_id, topic_start_id, topic_end_id)


def count_nodes_in_topic(topic_start_id: str, topic_end_id: str) -> int:
    """See TopicManager.count_nodes_in_topic for documentation."""
    return topic_manager.count_nodes_in_topic(topic_start_id, topic_end_id)


def _display_topic_evolution(current_node_id: str) -> None:
    """See TopicManager.display_topic_evolution for documentation."""
    return topic_manager.display_topic_evolution(current_node_id)