"""
Main topic detection and management functionality.

This module contains the TopicManager class and related functions for
detecting topic changes and extracting topic names from conversations.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

from episodic.prompt_manager import PromptManager

# Import from new modular files
from episodic.topics.topic_detection import detect_topic_change_separately as _detect_topic_change
from episodic.topics.topic_extraction import extract_topic_ollama as _extract_topic, build_conversation_segment
from episodic.topics.topic_analysis import (
    should_create_first_topic as _should_create_first_topic,
    is_node_in_topic_range,
    count_nodes_in_topic,
    count_user_messages_in_topic,
    display_topic_evolution
)

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
        # Check if we should skip topic detection due to recency
        if current_topic:
            topic_name, start_node_id = current_topic
            # Count USER messages only in the current active topic
            # For active topics, count up to the most recent node
            user_messages_in_topic = self.count_user_messages_in_topic(
                start_node_id, 
                None  # None means count to the end of the conversation
            )
            
            # REMOVED - Let topic detection run based on drift alone
            # Topics change when drift exceeds threshold, not based on message counts
            pass
            
        # Let detection proceed - topics change based on drift
        return _detect_topic_change(recent_messages, new_message, current_topic, self.prompt_manager)
    
    def extract_topic_ollama(self, conversation_segment: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Extract topic name from conversation segment using Ollama.
        
        Args:
            conversation_segment: Text containing recent conversation exchanges
            
        Returns:
            Tuple of (topic_name: Optional[str], cost_info: Optional[Dict])
        """
        return _extract_topic(conversation_segment)
    
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
        return _should_create_first_topic(user_node_id)
    
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
        return build_conversation_segment(nodes, max_length)
    
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
        return is_node_in_topic_range(node_id, topic_start_id, topic_end_id)
    
    def count_nodes_in_topic(self, topic_start_id: str, topic_end_id: str) -> int:
        """
        Count the number of nodes in a topic range.
        
        Args:
            topic_start_id: Start node of the topic
            topic_end_id: End node of the topic (can be None for ongoing topics)
            
        Returns:
            Number of nodes in the topic (including start and end)
        """
        return count_nodes_in_topic(topic_start_id, topic_end_id)
    
    def count_user_messages_in_topic(self, topic_start_id: str, topic_end_id: str) -> int:
        """
        Count the number of USER messages in a topic range.
        
        Args:
            topic_start_id: Start node of the topic
            topic_end_id: End node of the topic (None for ongoing topics)
            
        Returns:
            Number of user messages in the topic
        """
        return count_user_messages_in_topic(topic_start_id, topic_end_id)
    
    def display_topic_evolution(self, current_node_id: str) -> None:
        """
        Display topic evolution showing current and previous topics.
        
        Args:
            current_node_id: ID of the current conversation node
        """
        display_topic_evolution(current_node_id)


# Create a global instance for convenience
topic_manager = TopicManager()


# Module-level functions for backward compatibility
def detect_topic_change_separately(recent_messages: List[Dict[str, Any]], new_message: str, current_topic: Optional[Tuple[str, str]] = None) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Module-level wrapper for TopicManager.detect_topic_change_separately()"""
    return topic_manager.detect_topic_change_separately(recent_messages, new_message, current_topic)


def extract_topic_ollama(conversation_segment: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Module-level wrapper for TopicManager.extract_topic_ollama()"""
    return topic_manager.extract_topic_ollama(conversation_segment)


def should_create_first_topic(user_node_id: str) -> bool:
    """Module-level wrapper for TopicManager.should_create_first_topic()"""
    return topic_manager.should_create_first_topic(user_node_id)