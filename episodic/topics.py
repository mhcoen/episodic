"""
Topic detection and management functionality for Episodic.

This module handles all topic-related operations including:
- Detecting topic changes in conversations
- Extracting topic names from conversation segments
- Managing topic boundaries and evolution
- First topic creation logic
"""

import re
import json
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
from episodic.benchmark import benchmark_resource

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
        new_message: str,
        current_topic: Optional[Tuple[str, str]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detect if the topic has changed by analyzing recent messages and the new message.
        
        This function runs separately from the main conversation flow, using a focused
        LLM call to determine if the topic has shifted.
        
        Args:
            recent_messages: List of recent conversation nodes (3-5 messages)
            new_message: The new user message to analyze
            current_topic: Optional tuple of (topic_name, start_node_id) for the current topic
            
        Returns:
            Tuple of (topic_changed: bool, new_topic_name: Optional[str], cost_info: Optional[Dict])
        """
        try:
            # Check if we should skip topic detection due to recency
            if current_topic:
                topic_name, start_node_id = current_topic
                # Count USER messages only in the current active topic
                # For active topics, count up to the most recent node
                user_messages_in_topic = self.count_user_messages_in_topic(
                    start_node_id, 
                    None  # None means count to the end of the conversation
                )
                
                # Skip topic detection if current topic has fewer than threshold messages
                # But allow detection if we have many topics already (to avoid one giant topic)
                min_messages_before_change = config.get('min_messages_before_topic_change', 8)
                total_topics = len(get_recent_topics(limit=100))
                
                # Apply consistent threshold - no special cases for early topics
                # This prevents topics from being created with too few messages
                effective_min = min_messages_before_change
                    
                if user_messages_in_topic < effective_min:
                    if config.get("debug", False):
                        typer.echo(f"\n🔍 DEBUG: Skipping topic detection - current topic has only {user_messages_in_topic} user messages (min: {effective_min}, total topics: {total_topics})")
                    return False, None, None
            else:
                # No current topic set - check if we have enough messages overall
                # Count total user messages in the conversation
                user_messages = [msg for msg in recent_messages if msg.get('role') == 'user']
                total_user_messages = len(user_messages)
                min_messages = config.get('min_messages_before_topic_change', 8)
                
                if total_user_messages < min_messages:
                    if config.get("debug", False):
                        typer.echo(f"\n🔍 DEBUG: No current topic, only {total_user_messages} total user messages (min: {min_messages})")
                    return False, None, None
            # Build context from recent messages
            use_v2_prompt = config.get("topic_detection_v2", False)
            
            if use_v2_prompt:
                # V2 prompt expects only user messages in a specific format
                context_parts = []
                user_messages = [msg for msg in recent_messages if msg.get('role') == 'user']
                # Take the last 3-5 user messages
                for msg in user_messages[-5:]:
                    content = msg.get('content', '').strip()
                    if content:
                        context_parts.append(f"- User: {content}")
                context = "\n".join(context_parts)
            else:
                # Original format includes both user and assistant messages
                context_parts = []
                # Take the last 4 messages and reverse to get chronological order
                messages_for_context = list(reversed(recent_messages[-4:]))
                for msg in messages_for_context:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '').strip()
                    if content:
                        context_parts.append(f"{role}: {content}")
                context = "\n".join(context_parts)
            
            # Load topic detection prompt
            prompt_name = "topic_detection_v2" if use_v2_prompt else "topic_detection"
            
            topic_detection_prompt_content = self.prompt_manager.get(prompt_name)
            
            if topic_detection_prompt_content:
                # Use the loaded prompt template
                prompt_template = topic_detection_prompt_content
                # Format new message based on prompt version
                if use_v2_prompt:
                    formatted_new_message = new_message  # V2 expects just the message
                else:
                    formatted_new_message = f"user: {new_message}"  # V1 expects role prefix
                
                prompt = prompt_template.format(
                    recent_conversation=context,
                    new_message=formatted_new_message
                )
                
                if config.get("debug", False):
                    typer.echo(f"   Context preview: {context[:200]}...")
                    typer.echo(f"   Prompt length: {len(prompt)} chars")
                    typer.echo(f"   Full prompt:\n{prompt}\n   ---End prompt---")
            else:
                # Fallback to default prompt if file not found
                prompt = f"""Analyze if there is a MAJOR topic change in this conversation.

IMPORTANT RULES:
1. Only detect changes when switching to a COMPLETELY DIFFERENT domain of knowledge
2. Continuing to ask about the same subject, even with variations, is NOT a topic change
3. Default to "No" unless you are absolutely certain the topic has changed dramatically

Previous conversation:
{context}

New user message:
user: {new_message}

Has the topic changed to a COMPLETELY DIFFERENT subject?

Examples that should be "No":
- Asking more questions about the same subject
- Requesting clarification or more details
- Asking for examples of the same thing
- Variations on the same theme
- Related or connected topics

Only answer "Yes" for dramatic shifts like:
- Technical discussion → Personal life
- Science → Entertainment
- Programming → Food/Cooking
- Math → Travel plans

Respond with a JSON object containing only an "answer" field with value "Yes" or "No"."""

            # Get the topic detection model from config (default to ollama/llama3)
            topic_model = config.get("topic_detection_model", "ollama/llama3")
            
            if config.get("debug", False):
                typer.echo(f"\n🔍 DEBUG: Topic change detection")
                typer.echo(f"   Model: {topic_model}")
                typer.echo(f"   Recent messages: {len(recent_messages)}")
                typer.echo(f"   New message preview: {new_message[:100]}...")
            
            # Use configured model for detection with topic parameters
            topic_params = config.get_model_params('topic', model=topic_model)
            
            if config.get("debug", False):
                typer.echo(f"   Topic params being used: {topic_params}")
            
            with benchmark_resource("LLM Call", f"topic detection - {topic_model}"):
                response, cost_info = query_llm(
                    prompt, 
                    model=topic_model,
                    **topic_params
                )
            
            if response:
                response = response.strip()
                
                if config.get("debug", False):
                    typer.echo(f"   LLM response: {response}")
                    typer.echo(f"   Response type: {type(response)}")
                    typer.echo(f"   Response length: {len(response)}")
                
                try:
                    # Parse JSON response
                    result = json.loads(response)
                    
                    # Support both simple and intent-based formats
                    if "intent" in result and "shift" in result:
                        # New intent-based format
                        intent = result.get("intent", "UNKNOWN")
                        shift = result.get("shift", "NO")
                        
                        if config.get("debug", False):
                            typer.echo(f"   Parsed intent-based JSON: intent={intent}, shift={shift}")
                        
                        # Validate intent
                        valid_intents = ["JUST_COMMENT", "DEVELOP_TOPIC", "INTRODUCE_TOPIC", "CHANGE_TOPIC"]
                        if intent not in valid_intents:
                            typer.echo(f"   ⚠️  Invalid intent: {intent}")
                        
                        if shift == "YES":
                            if config.get("debug", False):
                                typer.echo(f"   ✅ Topic change detected (intent: {intent})")
                            return True, None, cost_info
                        else:
                            if config.get("debug", False):
                                typer.echo(f"   ➡️ Continuing same topic (intent: {intent})")
                            return False, None, cost_info
                    
                    else:
                        # Simple format (backward compatibility)
                        answer = result.get("answer", "No")
                        
                        if config.get("debug", False):
                            typer.echo(f"   Parsed simple JSON: answer={answer}")
                        
                        if answer == "Yes":
                            if config.get("debug", False):
                                typer.echo(f"   ✅ Topic change detected")
                            return True, None, cost_info
                        else:
                            if config.get("debug", False):
                                typer.echo(f"   ➡️ Continuing same topic")
                            return False, None, cost_info
                        
                except json.JSONDecodeError as e:
                    # Fallback to old parsing if JSON parsing fails
                    typer.echo(f"⚠️  Topic detection JSON parsing failed: {e}")
                    if config.get("debug", False):
                        typer.echo(f"   Raw response: {response}")
                        typer.echo(f"   Falling back to text parsing")
                    
                    # Check for clear YES response (fallback)
                    if response.upper().startswith("YES"):
                        if config.get("debug", False):
                            typer.echo(f"   ✅ Topic change detected (fallback)")
                        return True, None, cost_info
                    
                    # Default to no change
                    if config.get("debug", False):
                        typer.echo(f"   ➡️ Continuing same topic (fallback)")
                    return False, None, cost_info
        
        except Exception as e:
            logger.warning(f"Topic change detection error: {e}")
            if config.get("debug", False):
                typer.echo(f"⚠️  Topic detection error: {e}")
            return False, None, None
    
    def extract_topic_ollama(self, conversation_segment: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
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
            if config.get("debug", False):
                typer.echo(f"\n🔍 DEBUG: Topic extraction prompt:")
                typer.echo(f"   Model: {topic_model}")
                typer.echo(f"   Prompt preview: {prompt[:300]}...")
            
            topic_params = config.get_model_params('topic', model=topic_model)
            with benchmark_resource("LLM Call", f"topic extraction - {topic_model}"):
                response, cost_info = query_llm(prompt, model=topic_model, **topic_params)
            
            if response:
                # Debug: Show raw response
                if config.get("debug", False):
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
                if topic and topic != "no-topic":
                    # Check if the topic is too long (more than 5 words worth)
                    if len(topic) > 50:
                        if config.get("debug", False):
                            typer.echo(f"⚠️  Topic name too long ({len(topic)} chars): '{topic[:50]}...'")
                        # Try to extract just the first few words
                        parts = topic.split('-')[:3]
                        topic = '-'.join(parts)
                    
                    # Additional validation - check if it looks like the model included extra text
                    if any(phrase in topic for phrase in ['extract', 'topic', 'conversation', 'words', 'lowercase', 'hyphens']):
                        if config.get("debug", False):
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
                        if config.get("debug", False):
                            typer.echo(f"   DEBUG: Final topic name: '{topic}'")
                        return topic, cost_info
                
                if config.get("debug", False):
                    typer.echo(f"⚠️  Topic extraction failed or invalid: '{response}'")
                return None, cost_info
                
        except Exception as e:
            if config.get("debug", False):
                typer.echo(f"⚠️  Topic extraction error: {e}")
            return None, None
    
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
            
            if config.get('debug', False):
                logger.info(f"First topic check: {user_message_count} user messages, threshold: {threshold}")
                typer.echo(f"   DEBUG: should_create_first_topic: {user_message_count} user messages, threshold: {threshold}, returning: {user_message_count >= threshold}")
            
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
        
        if config.get("debug", False):
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
                        if config.get("debug", False):
                            typer.echo(f"   Truncated first part to fit max_length")
                    else:
                        if config.get("debug", False):
                            typer.echo(f"   Stopping - would exceed max_length")
                    break
                else:
                    segment_parts.append(part)
                    current_length += len(part)
            else:
                if config.get("debug", False):
                    typer.echo(f"   Skipping node with empty content (role={role})")
        
        result = "\n".join(segment_parts)
        if config.get("debug", False):
            typer.echo(f"   Final segment length: {len(result)} chars")
        
        return result
    
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
        # Handle ongoing topics (no end node)
        if not topic_end_id:
            return 0
            
        # Get ancestry of the end node
        topic_ancestry = get_ancestry(topic_end_id)
        
        # Debug: Check if start_id exists in ancestry
        ancestry_ids = [node['id'] for node in topic_ancestry]
        if config.get("debug", False) and topic_start_id not in ancestry_ids:
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
            if config.get("debug", False):
                typer.echo(f"   ⚠️  Using fallback count due to broken chain: {count} nodes")
        
        return max(count, 2)  # Ensure minimum of 2
    
    def count_user_messages_in_topic(self, topic_start_id: str, topic_end_id: str) -> int:
        """
        Count the number of USER messages in a topic range.
        
        Args:
            topic_start_id: Start node of the topic
            topic_end_id: End node of the topic
            
        Returns:
            Number of user messages in the topic
        """
        # Handle ongoing topics (no end node)
        if not topic_end_id:
            return 0
            
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
            if config.get("debug", False):
                typer.echo(f"   ⚠️  Using fallback count due to broken chain: {count} user messages")
        
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
                    # Topic evolution display disabled - provides little value since topics are named retroactively
                    pass
                    # typer.echo(f"\nTopic evolution: {prev_name} (completed) → {curr_name} (active)")
                
        except Exception as e:
            if config.get("debug", False):
                typer.echo(f"⚠️  Topic evolution display error: {e}")


# Create a global instance for convenience
topic_manager = TopicManager()


# Expose the main functions at module level for backward compatibility
def detect_topic_change_separately(recent_messages: List[Dict[str, Any]], new_message: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """See TopicManager.detect_topic_change_separately for documentation."""
    return topic_manager.detect_topic_change_separately(recent_messages, new_message)


def extract_topic_ollama(conversation_segment: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
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