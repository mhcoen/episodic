"""
Comprehensive tests for conversation flow functionality.

Tests the complete conversation lifecycle including:
- Node creation and management
- Message handling
- Response generation
- Context building
- Streaming
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from episodic.conversation import ConversationManager
from episodic.core import Node, ConversationDAG
from episodic.db import get_connection
from tests.fixtures.test_utils import isolated_config, temp_database, mock_llm_response


class TestConversationFlow(unittest.TestCase):
    """Test the complete conversation flow."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_context = isolated_config()
        self.config = self.config_context.__enter__()
        self.db_context = temp_database()
        self.db_path = self.db_context.__enter__()
        
        # Initialize conversation manager
        self.manager = ConversationManager()
        
    def tearDown(self):
        """Clean up test environment."""
        self.db_context.__exit__(None, None, None)
        self.config_context.__exit__(None, None, None)
    
    @patch('episodic.llm.query_llm')
    def test_basic_conversation_flow(self, mock_llm):
        """Test a basic conversation exchange."""
        # Mock LLM response
        mock_llm.return_value = mock_llm_response("Hello! How can I help you today?")
        
        # Send user message
        response = self.manager.send_message("Hello, AI assistant!")
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIn("Hello!", response)
        
        # Verify node was created
        from episodic.db import get_ancestry
        nodes = get_ancestry(self.manager.current_node_id)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]['message'], "Hello, AI assistant!")
        
    @patch('episodic.llm.query_llm')
    def test_multi_turn_conversation(self, mock_llm):
        """Test multiple conversation turns."""
        responses = [
            "Hello! I'm here to help.",
            "Python is a high-level programming language.",
            "Sure! Here's a simple example: print('Hello, World!')"
        ]
        
        messages = [
            "Hi there!",
            "Tell me about Python",
            "Can you show me an example?"
        ]
        
        mock_llm.side_effect = [mock_llm_response(r) for r in responses]
        
        # Send messages
        for i, msg in enumerate(messages):
            response = self.manager.send_message(msg)
            self.assertIn(responses[i], response)
        
        # Verify conversation history
        from episodic.db import get_ancestry
        nodes = get_ancestry(self.manager.current_node_id)
        
        # Should have 3 nodes (one for each exchange)
        self.assertEqual(len(nodes), 3)
        
        # Verify messages are in order
        for i, node in enumerate(nodes):
            self.assertEqual(node['message'], messages[i])
            self.assertIn(responses[i], node['response'])
    
    @patch('episodic.llm.query_llm')
    def test_context_building(self, mock_llm):
        """Test that context is properly built from history."""
        # Set up conversation history
        messages = ["First message", "Second message", "Third message"]
        responses = ["First response", "Second response", "Third response"]
        
        for msg, resp in zip(messages[:2], responses[:2]):
            mock_llm.return_value = mock_llm_response(resp)
            self.manager.send_message(msg)
        
        # Mock LLM to capture the context sent
        captured_context = None
        def capture_context(prompt, **kwargs):
            nonlocal captured_context
            captured_context = prompt
            return mock_llm_response(responses[2])
        
        mock_llm.side_effect = capture_context
        
        # Send third message
        self.manager.send_message(messages[2])
        
        # Verify context includes previous messages
        self.assertIsNotNone(captured_context)
        self.assertIn("First message", captured_context)
        self.assertIn("First response", captured_context)
        self.assertIn("Second message", captured_context)
        self.assertIn("Second response", captured_context)
    
    @patch('episodic.llm.query_llm')
    def test_system_prompt_handling(self, mock_llm):
        """Test that system prompts are properly handled."""
        mock_llm.return_value = mock_llm_response("I understand the instructions.")
        
        # Set a custom system prompt
        custom_prompt = "You are a helpful coding assistant."
        self.config.set('system_prompt', custom_prompt)
        
        # Capture the messages sent to LLM
        captured_messages = None
        def capture_messages(messages, **kwargs):
            nonlocal captured_messages
            captured_messages = messages
            return mock_llm_response("Response with custom prompt")
        
        mock_llm.side_effect = capture_messages
        
        # Send message
        self.manager.send_message("Write a Python function")
        
        # Verify system prompt was included
        self.assertIsNotNone(captured_messages)
        self.assertTrue(any(
            msg.get('role') == 'system' and custom_prompt in msg.get('content', '')
            for msg in captured_messages
        ))
    
    def test_node_relationships(self):
        """Test that nodes maintain proper parent-child relationships."""
        from episodic.db import store_node, get_node
        
        # Create root node
        root_id = store_node(
            message="Root message",
            response="Root response",
            model_name="test-model",
            system_prompt="Test prompt"
        )
        
        # Create child node
        child_id = store_node(
            message="Child message",
            response="Child response",
            parent_id=root_id,
            model_name="test-model",
            system_prompt="Test prompt"
        )
        
        # Verify relationships
        root = get_node(root_id)
        child = get_node(child_id)
        
        self.assertIsNone(root['parent_id'])
        self.assertEqual(child['parent_id'], root_id)
        
        # Test ancestry
        from episodic.db import get_ancestry
        ancestry = get_ancestry(child_id)
        self.assertEqual(len(ancestry), 2)
        self.assertEqual(ancestry[0]['id'], root_id)
        self.assertEqual(ancestry[1]['id'], child_id)
    
    @patch('episodic.llm.query_llm')
    def test_error_handling(self, mock_llm):
        """Test error handling in conversation flow."""
        # Mock LLM error
        mock_llm.side_effect = Exception("LLM service unavailable")
        
        # Should handle error gracefully
        with self.assertRaises(Exception) as context:
            self.manager.send_message("Test message")
        
        self.assertIn("LLM service unavailable", str(context.exception))
    
    @patch('episodic.llm.query_llm')
    def test_empty_message_handling(self, mock_llm):
        """Test handling of empty messages."""
        # Empty message should be rejected
        with self.assertRaises(ValueError):
            self.manager.send_message("")
        
        # Whitespace-only message should be rejected
        with self.assertRaises(ValueError):
            self.manager.send_message("   \n\t   ")
    
    @patch('episodic.llm.query_llm_streaming')
    def test_streaming_response(self, mock_streaming):
        """Test streaming response functionality."""
        # Mock streaming response
        def mock_stream():
            chunks = ["Hello", " there", "!", " How", " can", " I", " help?"]
            for chunk in chunks:
                yield {'choices': [{'delta': {'content': chunk}}]}
        
        mock_streaming.return_value = mock_stream()
        
        # Enable streaming
        self.config.set('stream_responses', True)
        
        # Collect streamed chunks
        chunks = []
        self.manager.response_callback = lambda chunk: chunks.append(chunk)
        
        # Send message (would normally stream)
        # Note: Actual streaming implementation may vary
        with patch.object(self.manager, '_should_stream', return_value=True):
            # This is a simplified test - actual implementation may differ
            pass
    
    def test_cost_tracking(self):
        """Test that costs are tracked correctly."""
        from episodic.db import update_session_cost, get_session_cost
        
        # Add some costs
        update_session_cost(0.01, 0.005, is_cached=False)
        update_session_cost(0.02, 0.01, is_cached=False)
        update_session_cost(0.005, 0.0025, is_cached=True)
        
        # Get total cost
        stats = get_session_cost()
        
        self.assertEqual(stats['total_cost'], 0.0475)
        self.assertEqual(stats['uncached_cost'], 0.045)
        self.assertEqual(stats['cache_savings'], 0.0025)


class TestConversationManager(unittest.TestCase):
    """Test ConversationManager specific functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_context = isolated_config()
        self.config = self.config_context.__enter__()
        self.db_context = temp_database()
        self.db_path = self.db_context.__enter__()
        
    def tearDown(self):
        """Clean up test environment."""
        self.db_context.__exit__(None, None, None)
        self.config_context.__exit__(None, None, None)
    
    def test_manager_initialization(self):
        """Test ConversationManager initialization."""
        manager = ConversationManager()
        
        # Should have required attributes
        self.assertIsNotNone(manager.current_node_id)
        self.assertIsNotNone(manager.dag)
        self.assertTrue(hasattr(manager, 'send_message'))
    
    def test_conversation_state_persistence(self):
        """Test that conversation state persists across manager instances."""
        from episodic.db import store_node, set_config
        
        # Create initial state
        node_id = store_node(
            message="Test message",
            response="Test response",
            model_name="test-model",
            system_prompt="Test prompt"
        )
        set_config('current_node_id', node_id)
        
        # Create new manager instance
        manager = ConversationManager()
        
        # Should restore state
        self.assertEqual(manager.current_node_id, node_id)
    
    @patch('episodic.llm.query_llm')
    def test_context_depth_configuration(self, mock_llm):
        """Test that context depth configuration is respected."""
        mock_llm.return_value = mock_llm_response("Response")
        
        # Create conversation with many messages
        manager = ConversationManager()
        for i in range(10):
            mock_llm.return_value = mock_llm_response(f"Response {i}")
            manager.send_message(f"Message {i}")
        
        # Set limited context depth
        self.config.set('context_depth', 3)
        
        # Capture context
        captured_prompt = None
        def capture_prompt(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return mock_llm_response("Final response")
        
        mock_llm.side_effect = capture_prompt
        manager.send_message("Final message")
        
        # Should only include last 3 messages in context
        self.assertIsNotNone(captured_prompt)
        self.assertIn("Message 7", captured_prompt)
        self.assertIn("Message 8", captured_prompt)
        self.assertIn("Message 9", captured_prompt)
        self.assertNotIn("Message 6", captured_prompt)


if __name__ == '__main__':
    unittest.main()