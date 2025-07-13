"""
Comprehensive tests for topic detection functionality.

Tests the complete topic detection system including:
- Automatic topic detection
- Manual topic indexing
- Topic naming and renaming
- Topic boundaries
- Hybrid detection methods
- Configuration handling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from episodic.topics.detector import TopicManager
from episodic.conversation import ConversationManager
from tests.fixtures.test_utils import isolated_config, temp_database, mock_llm_response
from tests.fixtures.conversations import (
    THREE_TOPICS_CONVERSATION,
    GRADUAL_DRIFT_CONVERSATION,
    SINGLE_TOPIC_CONVERSATION,
    create_test_node
)


class TestTopicDetectionIntegration(unittest.TestCase):
    """Test topic detection integration with conversation flow."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_context = isolated_config()
        self.config = self.config_context.__enter__()
        self.db_context = temp_database()
        self.db_path = self.db_context.__enter__()
        
        # Configure for testing
        self.config.set('automatic_topic_detection', True)
        self.config.set('min_messages_before_topic_change', 4)
        self.config.set('topic_detection_model', 'test-model')
        
        self.manager = ConversationManager()
        self.topic_manager = TopicManager()
        
    def tearDown(self):
        """Clean up test environment."""
        self.db_context.__exit__(None, None, None)
        self.config_context.__exit__(None, None, None)
    
    @patch('episodic.llm.query_llm')
    def test_automatic_topic_detection(self, mock_llm):
        """Test automatic topic detection during conversation."""
        # Mock responses for conversation
        conversation_responses = [
            "Mars is a fascinating planet with many challenges for colonization.",
            "The main challenges include radiation, atmosphere, and resources.",
            "The journey would take 6-9 months depending on orbital alignment.",
            "Italian pasta is made with tipo 00 flour and fresh eggs.",
            "Cook pasta in plenty of salted boiling water.",
            "There are hundreds of pasta shapes, each for specific sauces."
        ]
        
        # Mock topic detection responses
        topic_detection_responses = [
            json.dumps({"topic_changed": False, "confidence": 0.9}),
            json.dumps({"topic_changed": False, "confidence": 0.9}),
            json.dumps({"topic_changed": True, "confidence": 0.95}),  # Topic change!
            json.dumps({"topic_changed": False, "confidence": 0.9}),
        ]
        
        # Setup mock to handle both conversation and detection
        response_index = 0
        detection_index = 0
        
        def mock_llm_handler(*args, **kwargs):
            nonlocal response_index, detection_index
            
            # Check if this is a topic detection call
            if isinstance(args[0], list) and any('topic detection' in str(msg) for msg in args[0]):
                response = topic_detection_responses[detection_index % len(topic_detection_responses)]
                detection_index += 1
                return {'choices': [{'message': {'content': response}}]}
            else:
                # Regular conversation
                response = conversation_responses[response_index % len(conversation_responses)]
                response_index += 1
                return mock_llm_response(response)
        
        mock_llm.side_effect = mock_llm_handler
        
        # Send messages that should trigger topic change
        messages = [
            "Tell me about Mars colonization",
            "What are the main challenges?",
            "How long is the journey?",
            "I want to learn about Italian pasta",  # Topic change
            "How do I cook it properly?",
            "Tell me about pasta shapes"
        ]
        
        for msg in messages:
            self.manager.send_message(msg)
        
        # Verify topics were created
        from episodic.db import get_recent_topics
        topics = get_recent_topics()
        
        # Should have at least 2 topics
        self.assertGreaterEqual(len(topics), 2)
    
    def test_topic_boundary_detection(self):
        """Test accurate detection of topic boundaries."""
        from episodic.db import store_node, store_topic, update_topic_end_node
        
        # Create nodes
        nodes = []
        parent_id = None
        
        # Mars topic
        for i in range(6):
            node_id = store_node(
                message=f"Mars message {i}",
                response=f"Mars response {i}",
                parent_id=parent_id,
                model_name="test-model",
                system_prompt="Test prompt"
            )
            nodes.append(node_id)
            parent_id = node_id
        
        # Create topic
        topic_id = store_topic("Mars Exploration", nodes[0], nodes[5])
        
        # Pasta topic
        for i in range(6):
            node_id = store_node(
                message=f"Pasta message {i}",
                response=f"Pasta response {i}",
                parent_id=parent_id,
                model_name="test-model",
                system_prompt="Test prompt"
            )
            nodes.append(node_id)
            parent_id = node_id
        
        # Test boundary is correctly identified
        from episodic.db import get_node_topic
        
        # Nodes 0-5 should be in Mars topic
        for i in range(6):
            topic = get_node_topic(nodes[i])
            self.assertEqual(topic['id'], topic_id)
        
        # Nodes 6+ should not be in Mars topic
        for i in range(6, 12):
            topic = get_node_topic(nodes[i])
            self.assertIsNone(topic)
    
    @patch('episodic.llm.query_llm')
    def test_topic_naming_extraction(self, mock_llm):
        """Test automatic topic name extraction."""
        from episodic.db import store_node, store_topic, get_topic
        
        # Create conversation about space
        nodes = []
        parent_id = None
        messages = [
            ("Tell me about black holes", "Black holes are regions of spacetime..."),
            ("How do they form?", "Black holes form when massive stars collapse..."),
            ("What's an event horizon?", "The event horizon is the boundary...")
        ]
        
        for msg, resp in messages:
            node_id = store_node(msg, resp, parent_id, "test-model", "prompt")
            nodes.append(node_id)
            parent_id = node_id
        
        # Create topic with placeholder name
        topic_id = store_topic(f"ongoing-{datetime.now().isoformat()}", nodes[0])
        
        # Mock topic name extraction
        mock_llm.return_value = {'choices': [{'message': {'content': 'Black Holes'}}]}
        
        # Extract topic name
        from episodic.topic_management import extract_and_update_topic_name
        extract_and_update_topic_name(topic_id)
        
        # Verify topic was renamed
        topic = get_topic(topic_id)
        self.assertEqual(topic['name'], 'Black Holes')
    
    def test_manual_topic_indexing(self):
        """Test manual topic indexing functionality."""
        from episodic.db import store_node, manual_topic_index, get_topic_detection_scores
        
        # Create conversation
        nodes = []
        parent_id = None
        for i in range(10):
            node_id = store_node(
                f"Message {i}",
                f"Response {i}",
                parent_id,
                "test-model",
                "prompt"
            )
            nodes.append(node_id)
            parent_id = node_id
        
        # Manually mark topic change at node 5
        success = manual_topic_index(5, "test-user", "Testing manual indexing")
        self.assertTrue(success)
        
        # Verify score was recorded
        scores = get_topic_detection_scores()
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]['position'], 5)
        self.assertEqual(scores[0]['indexer'], 'test-user')
    
    def test_topic_compression_eligibility(self):
        """Test topic compression eligibility checks."""
        from episodic.db import store_node, store_topic, is_topic_eligible_for_compression
        
        # Create a topic with enough messages
        nodes = []
        parent_id = None
        for i in range(8):
            node_id = store_node(
                f"Message {i}",
                f"Response {i}",
                parent_id,
                "test-model",
                "prompt"
            )
            nodes.append(node_id)
            parent_id = node_id
        
        # Create closed topic
        topic_id = store_topic("Test Topic", nodes[0], nodes[7])
        
        # Should be eligible for compression
        eligible = is_topic_eligible_for_compression(topic_id)
        self.assertTrue(eligible)
        
        # Create ongoing topic (no end_node_id)
        ongoing_topic_id = store_topic("Ongoing Topic", nodes[7])
        
        # Ongoing topics should not be eligible
        eligible = is_topic_eligible_for_compression(ongoing_topic_id)
        self.assertFalse(eligible)
    
    def test_topic_detection_configuration(self):
        """Test various topic detection configurations."""
        # Test disabling automatic detection
        self.config.set('automatic_topic_detection', False)
        self.assertFalse(self.config.get('automatic_topic_detection'))
        
        # Test changing detection model
        self.config.set('topic_detection_model', 'gpt-4')
        self.assertEqual(self.config.get('topic_detection_model'), 'gpt-4')
        
        # Test minimum messages threshold
        self.config.set('min_messages_before_topic_change', 10)
        self.assertEqual(self.config.get('min_messages_before_topic_change'), 10)
        
        # Test detection parameters
        self.config.set('topic_params', {
            'temperature': 0,
            'max_tokens': 100
        })
        params = self.config.get('topic_params')
        self.assertEqual(params['temperature'], 0)
        self.assertEqual(params['max_tokens'], 100)
    
    @patch('episodic.topics.detector.get_embedding')
    def test_hybrid_detection_method(self, mock_embedding):
        """Test hybrid detection using embeddings and keywords."""
        from episodic.topics.hybrid import HybridDetector
        
        # Mock embeddings
        def mock_get_embedding(text):
            # Return different embeddings for different topics
            if 'mars' in text.lower():
                return [1.0, 0.0, 0.0]  # Mars vector
            elif 'pasta' in text.lower():
                return [0.0, 1.0, 0.0]  # Pasta vector
            else:
                return [0.0, 0.0, 1.0]  # Other vector
        
        mock_embedding.side_effect = mock_get_embedding
        
        detector = HybridDetector()
        
        # Test messages from same topic (low drift)
        mars_messages = ["Tell me about Mars", "Mars has red soil", "Mars atmosphere"]
        pasta_messages = ["Italian pasta recipe", "Cook pasta al dente", "Pasta shapes"]
        
        # Calculate drift within topic (should be low)
        mars_drift = detector.calculate_drift(mars_messages[:2], mars_messages[1:])
        self.assertLess(mars_drift, 0.5)
        
        # Calculate drift between topics (should be high)
        cross_drift = detector.calculate_drift(mars_messages, pasta_messages)
        self.assertGreater(cross_drift, 0.5)
    
    def test_topic_statistics(self):
        """Test topic statistics calculation."""
        from episodic.db import (
            store_node, store_topic, get_topic_statistics,
            count_nodes_in_topic
        )
        
        # Create topics of different sizes
        all_nodes = []
        parent_id = None
        
        # Topic 1: 5 messages
        topic1_nodes = []
        for i in range(5):
            node_id = store_node(f"T1 Message {i}", f"T1 Response {i}", 
                               parent_id, "model", "prompt")
            topic1_nodes.append(node_id)
            all_nodes.append(node_id)
            parent_id = node_id
        
        topic1_id = store_topic("Topic 1", topic1_nodes[0], topic1_nodes[-1])
        
        # Topic 2: 8 messages
        topic2_nodes = []
        for i in range(8):
            node_id = store_node(f"T2 Message {i}", f"T2 Response {i}",
                               parent_id, "model", "prompt")
            topic2_nodes.append(node_id)
            all_nodes.append(node_id)
            parent_id = node_id
        
        topic2_id = store_topic("Topic 2", topic2_nodes[0], topic2_nodes[-1])
        
        # Verify statistics
        self.assertEqual(count_nodes_in_topic(topic1_id), 5)
        self.assertEqual(count_nodes_in_topic(topic2_id), 8)
        
        # Test overall statistics
        stats = get_topic_statistics()
        self.assertIsNotNone(stats)
        self.assertEqual(stats['total_topics'], 2)
        self.assertEqual(stats['total_messages'], 13)


class TestTopicBoundaryAnalysis(unittest.TestCase):
    """Test topic boundary analysis functionality."""
    
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
    
    @patch('episodic.llm.query_llm')
    def test_boundary_analysis_with_llm(self, mock_llm):
        """Test LLM-based boundary analysis."""
        from episodic.topic_boundary_analyzer import analyze_topic_boundary_with_llm
        
        # Create test messages
        messages = [
            "Tell me about Mars",
            "Mars is the fourth planet",
            "It has a red appearance",
            "Now let's talk about cooking",  # Transition
            "Italian pasta is delicious",
            "Use fresh ingredients"
        ]
        
        # Mock LLM to identify boundary
        mock_llm.return_value = {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'boundary_index': 3,
                        'confidence': 0.95,
                        'reasoning': 'Clear topic shift from Mars to cooking'
                    })
                }
            }]
        }
        
        # Analyze boundary
        nodes = []
        for i, msg in enumerate(messages):
            nodes.append({
                'id': f'node{i}',
                'message': msg,
                'role': 'user' if i % 2 == 0 else 'assistant'
            })
        
        result = analyze_topic_boundary_with_llm(nodes, detection_point=4)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['boundary_index'], 3)
        self.assertGreater(result['confidence'], 0.9)
    
    def test_heuristic_boundary_detection(self):
        """Test heuristic-based boundary detection."""
        from episodic.topic_boundary_analyzer import find_topic_boundary_heuristic
        
        # Create messages with clear transition
        messages = [
            create_test_node("What's machine learning?", role="user"),
            create_test_node("Machine learning is...", role="assistant"),
            create_test_node("Tell me more about neural networks", role="user"),
            create_test_node("Neural networks are...", role="assistant"),
            create_test_node("Actually, let's switch topics. What's Italian cuisine?", role="user"),
            create_test_node("Italian cuisine is renowned...", role="assistant"),
        ]
        
        # Find boundary using heuristics
        boundary_idx = find_topic_boundary_heuristic(messages, detection_point=5)
        
        # Should identify the explicit transition
        self.assertEqual(boundary_idx, 4)  # The "switch topics" message


if __name__ == '__main__':
    unittest.main()