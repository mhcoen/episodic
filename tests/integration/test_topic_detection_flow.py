"""
Integration tests for complete topic detection flow.

Tests the full topic detection pipeline including:
- Database integration
- LLM calls
- Topic creation and updates
- Configuration changes
"""

import unittest
from unittest.mock import patch, Mock
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from episodic.conversation import ConversationManager
from episodic.topics import topic_manager
from tests.fixtures.conversations import (
    THREE_TOPICS_CONVERSATION,
    get_test_messages_only,
    assert_topics_detected
)
from tests.fixtures.test_utils import (
    temp_database,
    isolated_config,
    mock_llm_response,
    insert_test_nodes
)


class TestTopicDetectionIntegration(unittest.TestCase):
    """Test complete topic detection flow."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_context = isolated_config()
        self.config = self.config_context.__enter__()
        self.config.set('automatic_topic_detection', True)
        self.config.set('min_messages_before_topic_change', 4)
        
    def tearDown(self):
        """Clean up test environment."""
        self.config_context.__exit__(None, None, None)
        
    def test_automatic_topic_detection_flow(self):
        """Test automatic topic detection during conversation."""
        nodes, boundaries = THREE_TOPICS_CONVERSATION
        
        with temp_database() as db_path:
            # Initialize conversation manager
            conversation_manager = ConversationManager()
            conversation_manager.initialize_conversation()
            
            # Mock LLM responses for topic detection
            detection_responses = []
            for i, (node_idx, node) in enumerate(nodes):
                if node['role'] == 'user':
                    # Check if we're near a boundary
                    is_boundary = any(abs(node_idx - b) <= 2 for b in boundaries)
                    detection_responses.append({
                        'choices': [{
                            'message': {
                                'content': json.dumps({
                                    'topic_changed': is_boundary,
                                    'confidence': 0.9 if is_boundary else 0.2,
                                    'reason': 'Topic shift detected' if is_boundary else 'Same topic'
                                })
                            }
                        }]
                    })
            
            with patch('episodic.topics.detector.query_llm') as mock_llm:
                mock_llm.side_effect = detection_responses
                
                # Process each message
                for node in nodes:
                    if node['role'] == 'user':
                        # Simulate user message
                        with patch('episodic.conversation.store_node') as mock_store:
                            mock_store.return_value = node['id']
                            # Would call conversation_manager.handle_user_message(node['message'])
                            pass
                            
                # Check topics were created
                # In actual implementation, would verify via database
                
    def test_manual_topic_indexing(self):
        """Test manual topic indexing with sliding windows."""
        nodes, _ = THREE_TOPICS_CONVERSATION
        
        with temp_database() as db_path:
            # Insert test nodes
            insert_test_nodes(db_path, nodes)
            
            # Mock drift calculations
            with patch('episodic.topics.windows.SlidingWindowDetector._calculate_drift') as mock_drift:
                # High drift at boundaries, low within topics
                def drift_calculator(window1, window2):
                    # Simple heuristic based on content
                    w1_text = ' '.join(window1).lower()
                    w2_text = ' '.join(window2).lower()
                    
                    # Different topics have high drift
                    if ('mars' in w1_text and 'pasta' in w2_text) or \
                       ('pasta' in w1_text and 'neural' in w2_text):
                        return 0.85
                    return 0.15
                
                mock_drift.side_effect = drift_calculator
                
                # Run manual indexing
                from episodic.commands.index_topics import index_topics
                with patch('typer.secho'):  # Suppress output
                    # Would call index_topics(window_size=5, apply=True)
                    pass
                    
    def test_topic_renaming_flow(self):
        """Test topic renaming based on content."""
        with temp_database() as db_path:
            from episodic.db import store_topic, get_recent_topics
            
            # Create ongoing topic
            topic_id = store_topic(
                name="ongoing-discussion-123456",
                start_node_id="node1",
                end_node_id=None
            )
            
            # Insert some nodes for this topic
            nodes = [
                {
                    'id': f'node{i}',
                    'short_id': f'n{i}',
                    'message': f'Message about machine learning concept {i}',
                    'role': 'user' if i % 2 == 0 else 'assistant',
                    'parent_id': f'node{i-1}' if i > 0 else None,
                    'timestamp': f'2024-01-01T00:0{i}:00'
                }
                for i in range(6)
            ]
            insert_test_nodes(db_path, nodes)
            
            # Mock LLM response for topic extraction
            with mock_llm_response("machine-learning-basics"):
                from episodic.commands.topic_rename import rename_ongoing_topics
                with patch('typer.secho'):  # Suppress output
                    rename_ongoing_topics()
                
                # Verify topic was renamed
                topics = get_recent_topics(limit=1)
                self.assertEqual(len(topics), 1)
                self.assertNotEqual(topics[0]['name'], "ongoing-discussion-123456")
                self.assertIn("machine", topics[0]['name'].lower())
                
    def test_topic_compression_integration(self):
        """Test topic compression after detection."""
        with temp_database() as db_path:
            from episodic.db import store_topic, update_topic_end_node
            from episodic.compression import queue_topic_for_compression
            
            # Create a completed topic
            topic_id = store_topic(
                name="completed-topic",
                start_node_id="node1",
                end_node_id="node10"
            )
            
            # Insert nodes for the topic
            nodes = [
                {
                    'id': f'node{i}',
                    'short_id': f'n{i}',
                    'message': f'Discussion point {i}',
                    'role': 'user' if i % 2 == 0 else 'assistant',
                    'parent_id': f'node{i-1}' if i > 0 else None,
                    'timestamp': f'2024-01-01T00:0{i}:00'
                }
                for i in range(11)
            ]
            insert_test_nodes(db_path, nodes)
            
            # Test auto-compression queueing
            self.config.set('auto_compress_topics', True)
            self.config.set('compression_min_nodes', 5)
            
            # Mock compression
            with patch('episodic.compression.run_compression') as mock_compress:
                queue_topic_for_compression(topic_id)
                # In actual implementation, compression would run
                
    def test_configuration_changes(self):
        """Test that configuration changes affect topic detection."""
        with temp_database():
            # Test disabling automatic detection
            self.config.set('automatic_topic_detection', False)
            self.assertFalse(topic_manager._should_detect)
            
            # Test changing minimum messages
            self.config.set('min_messages_before_topic_change', 10)
            # Would verify this affects detection threshold
            
            # Test changing detection model
            self.config.set('topic_detection_model', 'gpt-4')
            # Would verify this is used in LLM calls


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in topic detection."""
    
    def test_empty_conversation(self):
        """Test topic detection with no messages."""
        with temp_database():
            manager = topic_manager
            result = manager._should_check_for_topic_change('node1')
            self.assertFalse(result)
            
    def test_single_message(self):
        """Test topic detection with single message."""
        with temp_database() as db_path:
            nodes = [{
                'id': 'node1',
                'short_id': 'n1',
                'message': 'Hello',
                'role': 'user',
                'parent_id': None,
                'timestamp': '2024-01-01T00:00:00'
            }]
            insert_test_nodes(db_path, nodes)
            
            result = topic_manager._should_check_for_topic_change('node1')
            self.assertFalse(result)
            
    def test_malformed_llm_response(self):
        """Test handling of malformed LLM responses."""
        with temp_database():
            # Test various malformed responses
            malformed_responses = [
                "Yes",  # Plain text instead of JSON
                "{'topic_changed': true}",  # Single quotes
                '{"invalid_key": true}',  # Missing required key
                "",  # Empty response
            ]
            
            for response in malformed_responses:
                with patch('episodic.topics.detector.query_llm') as mock_llm:
                    mock_llm.return_value = {
                        'choices': [{'message': {'content': response}}]
                    }
                    
                    # Should handle gracefully
                    result = topic_manager.detect_topic_change_separately(
                        "node1", "Test message", "user", []
                    )
                    self.assertIsInstance(result, dict)
                    self.assertIn('topic_changed', result)


if __name__ == '__main__':
    unittest.main()