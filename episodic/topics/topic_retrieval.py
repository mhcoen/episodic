"""
Topic-aware context retrieval.

Retrieves relevant context from previous topics when the user
returns to a topic they discussed earlier.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from episodic.config import config
from episodic.debug_utils import debug_print

logger = logging.getLogger(__name__)


def get_topic_messages(start_node_id: str, end_node_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get messages belonging to a topic.

    Args:
        start_node_id: The first node in the topic
        end_node_id: The last node in the topic (optional, uses start if None)

    Returns:
        List of messages in the topic
    """
    from episodic.db import get_ancestry

    # Get ancestry to the end node (or start if no end)
    target_node = end_node_id or start_node_id
    full_chain = get_ancestry(target_node)

    if not full_chain:
        return []

    # Find where the topic starts
    messages = []
    in_topic = False

    for node in full_chain:
        if node.get('id') == start_node_id or node.get('short_id') == start_node_id:
            in_topic = True

        if in_topic:
            messages.append({
                'role': node.get('role'),
                'content': node.get('content'),
                'node_id': node.get('id'),
            })

            # Stop if we've reached the end
            if end_node_id and (node.get('id') == end_node_id or node.get('short_id') == end_node_id):
                break

    return messages


def retrieve_topic_context(
    query: str,
    current_messages: List[Dict[str, Any]],
    max_messages: int = 10,
    max_tokens: int = 2000
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve relevant context from previous topics.

    Uses the configured topic strategy to detect if the current query
    relates to a previous topic and retrieves relevant messages.

    Args:
        query: The current user query
        current_messages: The current conversation context
        max_messages: Maximum number of messages to retrieve
        max_tokens: Maximum tokens in retrieved context

    Returns:
        Tuple of (retrieved_messages, retrieval_info)
    """
    # Check if topic retrieval is enabled
    if not config.get('topic_context_retrieval', False):
        return [], {'enabled': False}

    try:
        from episodic.topics import get_current_strategy
        from episodic.topics.strategy import Thread
        from episodic.db import get_recent_topics

        strategy = get_current_strategy()

        # Get recent topics from database
        topics = get_recent_topics(limit=20)

        if not topics:
            return [], {'reason': 'no_topics'}

        # Convert topics to Thread objects
        threads = []
        for topic in topics:
            # Get messages for this topic
            topic_messages = get_topic_messages(
                topic['start_node_id'],
                topic.get('end_node_id')
            )

            if topic_messages:
                threads.append(Thread(
                    id=f"{topic['name']}_{topic['start_node_id'][:8]}",
                    name=topic['name'],
                    start_node_id=topic['start_node_id'],
                    end_node_id=topic.get('end_node_id'),
                    message_count=len(topic_messages),
                    created_at=datetime.now(),
                    messages=topic_messages
                ))

        if not threads:
            return [], {'reason': 'no_thread_messages'}

        # Build a "current thread" from recent messages
        current_thread = None
        if current_messages:
            current_thread = Thread(
                id="current",
                name="current_conversation",
                start_node_id=current_messages[0].get('node_id', 'unknown'),
                end_node_id=None,
                message_count=len(current_messages),
                created_at=datetime.now(),
                messages=current_messages
            )

        # Use strategy to detect thread links
        links = strategy.detect_thread_link(query, threads, current_thread)

        if not links:
            return [], {'reason': 'no_links_detected'}

        # Retrieve messages from linked threads
        retrieved_messages = []
        retrieval_info = {
            'strategy': strategy.name,
            'links': []
        }

        # Sort links by weight (strongest first)
        sorted_links = sorted(links, key=lambda l: l.weight, reverse=True)

        token_count = 0
        for link in sorted_links:
            # Find the thread
            for thread in threads:
                if thread.id == link.to_thread_id:
                    # Add messages from this thread
                    for msg in thread.messages:
                        content = msg.get('content', '')
                        msg_tokens = len(content) // 4  # Rough estimate

                        if token_count + msg_tokens > max_tokens:
                            break
                        if len(retrieved_messages) >= max_messages:
                            break

                        retrieved_messages.append(msg)
                        token_count += msg_tokens

                    retrieval_info['links'].append({
                        'thread_id': thread.id,
                        'thread_name': thread.name,
                        'weight': link.weight,
                        'messages_retrieved': len(thread.messages)
                    })
                    break

            if len(retrieved_messages) >= max_messages:
                break
            if token_count >= max_tokens:
                break

        retrieval_info['total_messages'] = len(retrieved_messages)
        retrieval_info['total_tokens'] = token_count

        if config.get('debug'):
            debug_print(f"Topic retrieval: {len(retrieved_messages)} messages from {len(retrieval_info['links'])} threads")

        return retrieved_messages, retrieval_info

    except Exception as e:
        logger.error(f"Topic retrieval error: {e}")
        if config.get('debug'):
            debug_print(f"Topic retrieval error: {e}")
        return [], {'error': str(e)}


def format_topic_context(
    messages: List[Dict[str, Any]],
    thread_name: Optional[str] = None
) -> str:
    """
    Format retrieved topic messages for injection into context.

    Args:
        messages: Messages to format
        thread_name: Name of the topic thread (optional)

    Returns:
        Formatted context string
    """
    if not messages:
        return ""

    parts = []

    if thread_name:
        parts.append(f"[Previous discussion about {thread_name}]")
    else:
        parts.append("[Relevant context from earlier in conversation]")

    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')

        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."

        parts.append(f"{role.capitalize()}: {content}")

    return "\n".join(parts)
