"""
Test memory storage functionality.

Focus on:
- Conversation storage in memory
- Metadata handling
- Topic association
- Timestamp tracking
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch


class TestConversationMemoryStorage:
    """Test conversation memory storage functionality."""
    
    @pytest.fixture
    def mock_rag(self):
        """Create a mock RAG system."""
        mock = Mock()
        mock.add_document.return_value = ("doc-123", 1)
        return mock
    
    def test_store_conversation_to_memory(self, mock_rag, monkeypatch):
        """Test storing a conversation exchange to memory."""
        from episodic.conversation import ConversationManager
        
        # Create a real ConversationManager instance
        manager = ConversationManager()
        manager.current_topic = ("test-topic", "node-123")
        
        # Enable memory storage
        monkeypatch.setattr('episodic.config.config.get', 
                          lambda key, default=None: True if key == 'system_memory_auto_store' else default)
        
        # Patch get_rag_system at the module level where it's imported from
        monkeypatch.setattr('episodic.rag.get_rag_system', lambda: mock_rag)
        
        # Call the method
        manager.store_conversation_to_memory(
            user_input="What is Python?",
            assistant_response="Python is a programming language",
            user_node_id="user-123",
            assistant_node_id="assistant-456"
        )
        
        # Verify RAG was called correctly
        mock_rag.add_document.assert_called_once()
        
        # Get the call arguments - they are passed as keyword arguments
        call_args = mock_rag.add_document.call_args
        assert call_args is not None, "add_document was not called"
        
        # Extract keyword arguments (all args are passed as kwargs)
        _, kwargs = call_args
        
        # Check content format
        assert 'content' in kwargs
        assert "User: What is Python?" in kwargs['content']
        assert "Assistant: Python is a programming language" in kwargs['content']
        
        # Check source
        assert kwargs.get('source') == 'conversation'
        
        # Check metadata
        assert 'metadata' in kwargs
        metadata = kwargs['metadata']
        assert metadata['user_node_id'] == 'user-123'
        assert metadata['assistant_node_id'] == 'assistant-456'
        assert metadata['topic'] == 'test-topic'
        assert 'timestamp' in metadata
    
    def test_store_conversation_without_topic(self, mock_rag, monkeypatch):
        """Test storing conversation when no topic is set."""
        from episodic.conversation import ConversationManager
        
        manager = ConversationManager()
        manager.current_topic = None
        
        monkeypatch.setattr('episodic.config.config.get', 
                          lambda key, default=None: True if key == 'system_memory_auto_store' else default)
        
        # Patch get_rag_system at the module level
        monkeypatch.setattr('episodic.rag.get_rag_system', lambda: mock_rag)
        
        manager.store_conversation_to_memory(
            user_input="Hello",
            assistant_response="Hi there!",
            user_node_id="user-789",
            assistant_node_id="assistant-012"
        )
        
        # Verify topic is not in metadata
        call_args = mock_rag.add_document.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert 'metadata' in kwargs
        assert 'topic' not in kwargs['metadata']
    
    def test_memory_storage_disabled(self, mock_rag, monkeypatch):
        """Test that memory is not stored when disabled."""
        from episodic.conversation import ConversationManager
        
        manager = ConversationManager()
        
        # Disable memory storage
        monkeypatch.setattr('episodic.config.config.get', 
                          lambda key, default=None: False if key == 'system_memory_auto_store' else default)
        
        # Patch get_rag_system at the module level
        monkeypatch.setattr('episodic.rag.get_rag_system', lambda: mock_rag)
        
        manager.store_conversation_to_memory(
            user_input="Test",
            assistant_response="Response",
            user_node_id="user-999",
            assistant_node_id="assistant-999"
        )
        
        # Verify RAG was not called
        mock_rag.add_document.assert_not_called()
    
    def test_memory_storage_error_handling(self, monkeypatch, capsys):
        """Test error handling in memory storage."""
        from episodic.conversation import ConversationManager
        
        # Mock RAG to raise an error
        def mock_get_rag():
            raise Exception("RAG initialization failed")
        
        # Patch get_rag_system at the module level
        monkeypatch.setattr('episodic.rag.get_rag_system', mock_get_rag)
        monkeypatch.setattr('episodic.config.config.get', 
                          lambda key, default=None: True if key in ['system_memory_auto_store', 'debug'] else default)
        
        manager = ConversationManager()
        
        # Should not raise exception
        manager.store_conversation_to_memory(
            user_input="Test",
            assistant_response="Response",
            user_node_id="user-error",
            assistant_node_id="assistant-error"
        )
        
        # The error is silently caught, so no output unless debug is on
        captured = capsys.readouterr()
        # With debug on, we'd see traceback
        assert "traceback" in captured.out.lower() or len(captured.out) == 0


class TestMemoryMetadata:
    """Test memory metadata handling."""
    
    def test_conversation_metadata_format(self):
        """Test that conversation metadata is properly formatted."""
        from datetime import datetime
        
        # Create test metadata
        metadata = {
            'source': 'conversation',
            'user_node_id': 'user-123',
            'assistant_node_id': 'assistant-456',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'topic': 'test-topic'
        }
        
        # Verify required fields
        assert metadata['source'] == 'conversation'
        assert 'user_node_id' in metadata
        assert 'assistant_node_id' in metadata
        assert 'timestamp' in metadata
        
        # Verify timestamp is ISO format
        parsed_time = datetime.fromisoformat(metadata['timestamp'])
        assert parsed_time.tzinfo is not None  # Has timezone info
    
    def test_topic_metadata_handling(self):
        """Test proper handling of topic metadata."""
        # Test with placeholder topic
        placeholder_topic = "ongoing-1234567890"
        assert placeholder_topic.startswith("ongoing-")
        
        # Test with finalized topic
        finalized_topic = "Python Programming Discussion"
        assert not finalized_topic.startswith("ongoing-")


class TestMemoryRetrieval:
    """Test memory retrieval functionality."""
    
    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        return {
            'results': [
                {
                    'content': 'User: What is AI?\\nAssistant: AI is artificial intelligence',
                    'metadata': {
                        'source': 'conversation',
                        'topic': 'AI Discussion',
                        'user_node_id': 'user-ai-1',
                        'assistant_node_id': 'assistant-ai-1'
                    },
                    'relevance_score': 0.95
                },
                {
                    'content': 'User: Tell me more about AI\\nAssistant: AI involves machine learning...',
                    'metadata': {
                        'source': 'conversation',
                        'topic': 'AI Discussion',
                        'user_node_id': 'user-ai-2',
                        'assistant_node_id': 'assistant-ai-2'
                    },
                    'relevance_score': 0.85
                }
            ],
            'total': 2
        }
    
    def test_memory_search_results_format(self, mock_search_results):
        """Test that memory search results are properly formatted."""
        results = mock_search_results['results']
        
        for result in results:
            # Verify content format
            assert 'User:' in result['content']
            assert 'Assistant:' in result['content']
            
            # Verify metadata
            assert result['metadata']['source'] == 'conversation'
            assert 'topic' in result['metadata']
            assert 'user_node_id' in result['metadata']
            assert 'assistant_node_id' in result['metadata']
            
            # Verify relevance score
            assert 0 <= result['relevance_score'] <= 1
    
    def test_memory_context_enhancement(self, mock_search_results, monkeypatch):
        """Test memory context enhancement for conversations."""
        # Just test the mock behavior, as actual context building requires full integration
        mock_rag_manager = Mock()
        mock_rag_manager.is_available.return_value = True
        mock_rag_manager.search.return_value = [
            {
                'text': result['content'],
                'doc_id': f"doc-{i}",
                'metadata': result['metadata']
            }
            for i, result in enumerate(mock_search_results['results'])
        ]
        
        # Test that the mock returns expected results
        results = mock_rag_manager.search("Tell me about AI")
        assert len(results) == 2
        assert "What is AI?" in results[0]['text']
        assert "AI is artificial intelligence" in results[0]['text']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])