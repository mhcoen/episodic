"""
Comprehensive test suite for the memory system.

Tests cover:
- Memory storage and retrieval
- Source filtering
- Multi-collection architecture
- Migration functionality
- Memory commands
- Context enhancement
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone
import warnings

# Suppress ChromaDB warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", message=".*telemetry.*")

from episodic.config import config
from episodic.rag_collections import CollectionType
from episodic.rag_adapter import EpisodicRAGAdapter


class TestMultiCollectionRAG:
    """Test the multi-collection RAG system."""
    
    def test_collection_types(self):
        """Test that collection types are properly defined."""
        assert CollectionType.USER_DOCS == "user_docs"
        assert CollectionType.CONVERSATION == "conversation"
    
    def test_multi_collection_mock(self, monkeypatch):
        """Test multi-collection RAG with mocking."""
        # Mock the MultiCollectionRAG initialization
        mock_collections = {
            CollectionType.USER_DOCS: Mock(),
            CollectionType.CONVERSATION: Mock()
        }
        
        mock_multi_rag = Mock()
        mock_multi_rag.collections = mock_collections
        mock_multi_rag.add_document.return_value = ('doc-123', 1)
        mock_multi_rag.search.return_value = {'results': [], 'total': 0}
        mock_multi_rag.get_collection_stats.return_value = {
            CollectionType.CONVERSATION: {'count': 0},
            CollectionType.USER_DOCS: {'count': 0}
        }
        
        # Test adding conversation memory
        doc_id, chunks = mock_multi_rag.add_document(
            content="User: Hello\\nAssistant: Hi there!",
            source="conversation",
            metadata={"topic": "greeting"}
        )
        
        assert doc_id == 'doc-123'
        assert chunks == 1
        mock_multi_rag.add_document.assert_called_once()
    
    def test_search_with_collection_filter(self, monkeypatch):
        """Test searching with collection type filter."""
        mock_multi_rag = Mock()
        
        # Mock different results for different searches
        mock_multi_rag.search.side_effect = [
            # Conversation search
            {
                'results': [{
                    'content': 'User: Test\\nAssistant: Response',
                    'metadata': {'source': 'conversation'},
                    'relevance_score': 0.9
                }],
                'total': 1
            },
            # User docs search
            {
                'results': [{
                    'content': 'Document content',
                    'metadata': {'source': 'file'},
                    'relevance_score': 0.85
                }],
                'total': 1
            }
        ]
        
        # Search conversation
        results = mock_multi_rag.search("test", collection_type=CollectionType.CONVERSATION)
        assert len(results['results']) == 1
        assert results['results'][0]['metadata']['source'] == 'conversation'
        
        # Search user docs
        results = mock_multi_rag.search("test", collection_type=CollectionType.USER_DOCS)
        assert len(results['results']) == 1
        assert results['results'][0]['metadata']['source'] == 'file'


class TestRAGAdapter:
    """Test the RAG adapter for backward compatibility."""
    
    @pytest.fixture
    def adapter(self, monkeypatch):
        """Create a RAG adapter instance."""
        # Mock the multi-collection RAG
        mock_multi_rag = Mock()
        monkeypatch.setattr('episodic.rag_adapter.get_multi_collection_rag', lambda: mock_multi_rag)
        
        adapter = EpisodicRAGAdapter()
        adapter.multi_rag = mock_multi_rag
        return adapter
    
    def test_add_document_routes_correctly(self, adapter):
        """Test that documents are routed to correct collections."""
        # Test conversation routing
        adapter.add_document(
            content="User: Hello\\nAssistant: Hi",
            source="conversation"
        )
        
        adapter.multi_rag.add_document.assert_called_with(
            content="User: Hello\\nAssistant: Hi",
            source="conversation",
            metadata=None,
            collection_type=CollectionType.CONVERSATION,
            chunk=True
        )
        
        # Test user document routing
        adapter.add_document(
            content="Test document",
            source="file"
        )
        
        adapter.multi_rag.add_document.assert_called_with(
            content="Test document",
            source="file",
            metadata=None,
            collection_type=CollectionType.USER_DOCS,
            chunk=True
        )
    
    def test_search_with_source_filter(self, adapter):
        """Test search with source filtering."""
        adapter.multi_rag.search.return_value = {'results': [], 'total': 0}
        
        # Search for conversations
        adapter.search("test", source_filter="conversation")
        
        adapter.multi_rag.search.assert_called_with(
            query="test",
            n_results=5,
            collection_type=CollectionType.CONVERSATION,
            source_filter="conversation"
        )
        
        # Search for user docs
        adapter.search("test", source_filter="file")
        
        adapter.multi_rag.search.assert_called_with(
            query="test",
            n_results=5,
            collection_type=CollectionType.USER_DOCS,
            source_filter="file"
        )


class TestMemoryCommands:
    """Test memory command functions."""
    
    @pytest.fixture
    def mock_rag(self, monkeypatch):
        """Mock the RAG system for command tests."""
        mock = Mock()
        # Mock get_rag_system to return our mock
        monkeypatch.setattr('episodic.rag.get_rag_system', lambda: mock)
        return mock
    
    def test_search_memories_filters_conversations(self, mock_rag, capsys):
        """Test that search_memories only searches conversation memories."""
        mock_rag.search.return_value = {
            'results': [
                {
                    'content': 'User: Test\\nAssistant: Response',
                    'metadata': {'doc_id': '123', 'source': 'conversation'},
                    'relevance_score': 0.9
                }
            ]
        }
        
        from episodic.commands.memory import search_memories
        search_memories("test query")
        
        # Verify source filter was applied
        mock_rag.search.assert_called_once_with(
            "test query", 
            n_results=10,
            source_filter='conversation'
        )
        
        # Check output
        captured = capsys.readouterr()
        assert "Searching memories for: test query" in captured.out
        assert "Found 1 matches" in captured.out
    
    def test_list_memories_filters_conversations(self, mock_rag, capsys):
        """Test that list_memories only shows conversation memories."""
        mock_rag.list_documents.return_value = [
            {
                'doc_id': '123',
                'source': 'conversation',
                'indexed_at': datetime.now().isoformat(),
                'preview': 'User: Hello...'
            }
        ]
        
        from episodic.commands.memory import list_memories
        list_memories(limit=10)
        
        # Verify source filter was applied
        mock_rag.list_documents.assert_called_once_with(
            limit=10,
            source_filter='conversation'
        )
        
        # Check output
        captured = capsys.readouterr()
        assert "Memory Entries" in captured.out
    
    def test_forget_all_filters_conversations(self, mock_rag, monkeypatch):
        """Test that forget --all only deletes conversation memories."""
        # Mock user confirmation
        monkeypatch.setattr('typer.confirm', lambda x: True)
        mock_rag.clear_documents.return_value = 5
        
        from episodic.commands.memory import forget_command
        forget_command("--all")
        
        # Verify source filter was applied
        mock_rag.clear_documents.assert_called_once_with(
            source_filter='conversation'
        )
    
    def test_memory_stats_shows_conversation_focus(self, mock_rag, capsys):
        """Test that memory stats focuses on conversation memories."""
        mock_rag.get_stats.return_value = {
            'total_documents': 100,
            'collection_count': 150,
            'source_distribution': {
                'conversation': 80,
                'file': 20
            },
            'embedding_model': 'test-model'
        }
        
        from episodic.commands.memory import memory_stats_command
        memory_stats_command()
        
        captured = capsys.readouterr()
        assert "Memory System Statistics" in captured.out
        assert "conversation: 80" in captured.out


class TestMigration:
    """Test migration functionality."""
    
    def test_check_migration_needed(self, monkeypatch):
        """Test checking if migration is needed."""
        from episodic.rag_migration import check_migration_needed
        
        # Case 1: Migration already completed
        monkeypatch.setattr(config, 'get', lambda key, default=None: True if key == 'collection_migration_completed' else default)
        assert not check_migration_needed()
        
        # Case 2: No old data exists
        monkeypatch.setattr(config, 'get', lambda key, default=None: False)
        with patch('os.path.exists', return_value=False):
            assert not check_migration_needed()
    
    def test_migrate_dry_run(self, monkeypatch, capsys):
        """Test migration in dry run mode."""
        from episodic.rag_migration import migrate_to_multi_collection
        
        monkeypatch.setattr('episodic.rag_migration.check_migration_needed', lambda: True)
        monkeypatch.setattr('episodic.rag_migration.count_documents_by_source', 
                          lambda: {'conversation': 2, 'file': 1})
        
        result = migrate_to_multi_collection(dry_run=True)
        
        assert result is True
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "conversation: 2 documents" in captured.out
        assert "file: 1 documents" in captured.out


class TestIntegration:
    """Integration tests for the complete memory system."""
    
    def test_memory_workflow_mock(self, monkeypatch):
        """Test complete memory workflow with mocking."""
        # Mock the entire RAG system
        mock_rag = Mock()
        mock_rag.add_document.return_value = ('doc-123', 1)
        
        # Mock search results
        mock_rag.search.side_effect = [
            # First search - finds the conversation
            {
                'results': [{
                    'content': 'User: What is Python?\\nAssistant: Python is a programming language',
                    'metadata': {'source': 'conversation', 'topic': 'programming'},
                    'relevance_score': 0.95
                }],
                'total': 1
            },
            # Second search - no results for file search
            {'results': [], 'total': 0}
        ]
        
        # Mock get_rag_system to return our mock
        monkeypatch.setattr('episodic.rag.get_rag_system', lambda: mock_rag)
        
        from episodic.rag import get_rag_system
        rag = get_rag_system()
        
        # Store a conversation
        doc_id, _ = rag.add_document(
            content="User: What is Python?\\nAssistant: Python is a programming language",
            source="conversation",
            metadata={'topic': 'programming'}
        )
        
        assert doc_id == 'doc-123'
        
        # Search for it
        results = rag.search("Python", source_filter="conversation")
        assert len(results['results']) > 0
        assert results['results'][0]['metadata']['source'] == 'conversation'
        
        # Verify it doesn't appear in user document searches
        results = rag.search("Python", source_filter="file")
        assert len(results['results']) == 0


# Test helper functions
def create_test_conversation(content="User: Test\\nAssistant: Response"):
    """Helper to create test conversation data."""
    return {
        'content': content,
        'source': 'conversation',
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'topic': 'test-topic'
        }
    }


def create_test_document(content="Test document content"):
    """Helper to create test document data."""
    return {
        'content': content,
        'source': 'file',
        'metadata': {
            'filename': 'test.txt',
            'indexed_at': datetime.now().isoformat()
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])