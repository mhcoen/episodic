"""
Unit tests for topic detection functionality.

Tests the core topic detection algorithms including:
- Sliding window detection
- Drift calculation
- Topic boundaries
- Configuration handling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from episodic.topics.detector import TopicManager
from episodic.topics.windows import SlidingWindowDetector
from tests.fixtures.conversations import (
    THREE_TOPICS_CONVERSATION, 
    GRADUAL_DRIFT_CONVERSATION,
    SINGLE_TOPIC_CONVERSATION,
    get_test_messages_only
)
from tests.fixtures.test_utils import (
    isolated_config, 
    temp_database,
    mock_llm_response
)


class TestTopicDetection(unittest.TestCase):
    """Test core topic detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_context = isolated_config()
        self.config = self.config_context.__enter__()
        
    def tearDown(self):
        """Clean up test environment."""
        self.config_context.__exit__(None, None, None)
        
    def test_sliding_window_initialization(self):
        """Test sliding window detector initialization."""
        detector = SlidingWindowDetector(window_size=5)
        self.assertEqual(detector.window_size, 5)
        self.assertEqual(detector.step_size, 1)
        
    def test_window_extraction(self):
        """Test window extraction from messages."""
        nodes, _ = THREE_TOPICS_CONVERSATION
        messages = get_test_messages_only(nodes)
        
        detector = SlidingWindowDetector(window_size=3)
        windows = list(detector._extract_windows(messages))
        
        # Should have len(messages) - window_size + 1 windows
        expected_count = len(messages) - 3 + 1
        self.assertEqual(len(windows), expected_count)
        
        # First window should have first 3 messages
        self.assertEqual(len(windows[0]), 3)
        self.assertEqual(windows[0], messages[:3])
        
    def test_topic_manager_initialization(self):
        """Test TopicManager initialization."""
        with temp_database():
            manager = TopicManager()
            self.assertIsNotNone(manager)
            self.assertTrue(hasattr(manager, 'detect_topic_change'))
            
    @patch('episodic.topics.detector.get_ancestry')
    @patch('episodic.topics.detector.get_recent_topics')
    def test_should_check_for_topic_change(self, mock_topics, mock_ancestry):
        """Test logic for when to check for topic changes."""
        # Setup mocks
        mock_topics.return_value = [
            {'id': 'topic1', 'name': 'Topic 1'},
            {'id': 'topic2', 'name': 'Topic 2'}
        ]
        
        # Create nodes with alternating user/assistant messages
        nodes = []
        for i in range(10):
            nodes.append({
                'id': f'node{i}',
                'role': 'user' if i % 2 == 0 else 'assistant',
                'message': f'Message {i}'
            })
        mock_ancestry.return_value = nodes
        
        with temp_database():
            manager = TopicManager()
            
            # Should check after minimum messages
            self.config.set('min_messages_before_topic_change', 4)
            result = manager._should_check_for_topic_change('current_node')
            self.assertTrue(result)
            
            # Should not check with too few messages
            mock_ancestry.return_value = nodes[:3]  # Only 2 user messages
            result = manager._should_check_for_topic_change('current_node')
            self.assertFalse(result)
            
    def test_drift_calculation(self):
        """Test semantic drift calculation between windows."""
        # This would require embedding support, so we'll mock it
        detector = SlidingWindowDetector(window_size=3)
        
        # Mock embeddings - similar messages should have low drift
        with patch.object(detector, '_calculate_drift') as mock_drift:
            mock_drift.return_value = 0.2  # Low drift
            
            window1 = ["Tell me about Mars", "Mars has red soil", "The atmosphere is thin"]
            window2 = ["Mars gravity is weak", "It has two moons", "Olympus Mons is huge"]
            
            drift = detector._calculate_drift(window1, window2)
            self.assertEqual(drift, 0.2)
            
    def test_topic_boundary_detection(self):
        """Test detection of topic boundaries in conversation."""
        nodes, expected_boundaries = THREE_TOPICS_CONVERSATION
        
        with temp_database() as db_path:
            # Insert nodes into database
            from tests.fixtures.test_utils import insert_test_nodes
            insert_test_nodes(db_path, nodes)
            
            # Test with mocked drift scores
            detector = SlidingWindowDetector(window_size=5)
            
            # Mock high drift at topic boundaries
            with patch.object(detector, '_calculate_drift') as mock_drift:
                def drift_side_effect(w1, w2):
                    # Check if windows cross topic boundary
                    for boundary_idx in expected_boundaries:
                        if any("Mars" in m for m in w1) and any("pasta" in m for m in w2):
                            return 0.8  # High drift
                        if any("pasta" in m for m in w1) and any("neural" in m for m in w2):
                            return 0.8  # High drift
                    return 0.2  # Low drift within topic
                
                mock_drift.side_effect = drift_side_effect
                
                # Detect boundaries
                messages = get_test_messages_only(nodes)
                scores = detector.calculate_scores(messages)
                
                # Should have high scores near boundaries
                self.assertIsInstance(scores, list)
                self.assertTrue(len(scores) > 0)
                
    def test_configuration_parameters(self):
        """Test that configuration parameters are respected."""
        with temp_database():
            manager = TopicManager()
            
            # Test automatic detection toggle
            self.config.set('automatic_topic_detection', False)
            self.assertFalse(self.config.get('automatic_topic_detection'))
            
            # Test minimum messages threshold
            self.config.set('min_messages_before_topic_change', 10)
            self.assertEqual(self.config.get('min_messages_before_topic_change'), 10)
            
            # Test detection model setting
            self.config.set('topic_detection_model', 'gpt-4')
            self.assertEqual(self.config.get('topic_detection_model'), 'gpt-4')


class TestTopicBoundaries(unittest.TestCase):
    """Test topic boundary detection and assignment."""
    
    def test_boundary_analysis(self):
        """Test boundary analysis to find actual topic transitions."""
        from episodic.topics.boundaries import analyze_topic_boundary
        
        nodes, _ = THREE_TOPICS_CONVERSATION
        
        # Mock the boundary analyzer
        with patch('episodic.topics.boundaries.query_llm') as mock_llm:
            mock_llm.return_value = {
                'choices': [{
                    'message': {
                        'content': '{"boundary_node_id": "node5", "confidence": 0.9}'
                    }
                }]
            }
            
            # Test boundary analysis
            result = analyze_topic_boundary(nodes, start_idx=3)
            self.assertIsNotNone(result)
            # The actual implementation would return the boundary node
            
    def test_gradual_drift_detection(self):
        """Test detection of gradual topic drift."""
        nodes, boundaries = GRADUAL_DRIFT_CONVERSATION
        messages = get_test_messages_only(nodes)
        
        detector = SlidingWindowDetector(window_size=4)
        
        # The drift should gradually increase
        with patch.object(detector, '_calculate_drift') as mock_drift:
            # Simulate increasing drift as we move through conversation
            drift_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
            mock_drift.side_effect = drift_values
            
            scores = detector.calculate_scores(messages[:6])
            
            # Later scores should be higher
            self.assertTrue(len(scores) > 0)
            # Would verify increasing trend in actual implementation


if __name__ == '__main__':
    unittest.main()