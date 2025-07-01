"""Test fixtures and utilities for Episodic tests."""

from .conversations import (
    create_test_node,
    create_test_conversation,
    assert_topics_detected,
    get_test_messages_only,
    THREE_TOPICS_CONVERSATION,
    GRADUAL_DRIFT_CONVERSATION,
    SINGLE_TOPIC_CONVERSATION
)

from .test_utils import (
    temp_database,
    mock_llm_response,
    isolated_config,
    create_test_topics,
    insert_test_nodes,
    assert_command_output,
    mock_topic_detection,
    capture_cli_output
)

__all__ = [
    # Conversation fixtures
    'create_test_node',
    'create_test_conversation',
    'assert_topics_detected',
    'get_test_messages_only',
    'THREE_TOPICS_CONVERSATION',
    'GRADUAL_DRIFT_CONVERSATION',
    'SINGLE_TOPIC_CONVERSATION',
    
    # Test utilities
    'temp_database',
    'mock_llm_response',
    'isolated_config',
    'create_test_topics',
    'insert_test_nodes',
    'assert_command_output',
    'mock_topic_detection',
    'capture_cli_output'
]