"""
Unit tests for memory management commands.

Tests the /memory, /forget, and /memory-stats commands.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime

from episodic.commands.memory import (
    memory_command, forget_command, memory_stats_command,
    list_memories, search_memories, show_memory
)


class TestMemoryCommand:
    """Test the /memory command and its subcommands."""
    
    @patch('episodic.commands.memory.list_memories')
    def test_memory_no_args_shows_recent(self, mock_list):
        """Test that /memory with no args shows recent memories."""
        memory_command()
        mock_list.assert_called_once_with(limit=10)
    
    @patch('episodic.commands.memory.search_memories')
    def test_memory_search(self, mock_search):
        """Test /memory search functionality."""
        memory_command("search", "test", "query")
        mock_search.assert_called_once_with("test query")
    
    @patch('episodic.commands.memory.typer.secho')
    def test_memory_search_no_query(self, mock_secho):
        """Test /memory search without query shows error."""
        memory_command("search")
        mock_secho.assert_called_with("Usage: /memory search <query>", fg='red')
    
    @patch('episodic.commands.memory.show_memory')
    def test_memory_show(self, mock_show):
        """Test /memory show functionality."""
        memory_command("show", "abc123")
        mock_show.assert_called_once_with("abc123")
    
    @patch('episodic.commands.memory.typer.secho')
    def test_memory_show_no_id(self, mock_secho):
        """Test /memory show without ID shows error."""
        memory_command("show")
        mock_secho.assert_called_with("Usage: /memory show <id>", fg='red')
    
    @patch('episodic.commands.memory.list_memories')
    def test_memory_list_default(self, mock_list):
        """Test /memory list with default limit."""
        memory_command("list")
        mock_list.assert_called_once_with(limit=20)
    
    @patch('episodic.commands.memory.list_memories')
    def test_memory_list_custom_limit(self, mock_list):
        """Test /memory list with custom limit."""
        memory_command("list", "50")
        mock_list.assert_called_once_with(limit=50)
    
    @patch('episodic.commands.memory.typer.secho')
    def test_memory_unknown_action(self, mock_secho):
        """Test /memory with unknown action shows error."""
        memory_command("unknown")
        assert any("Unknown memory action: unknown" in str(call) for call in mock_secho.call_args_list)


class TestListMemories:
    """Test the list_memories function."""
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.commands.memory.typer.secho')
    def test_list_memories_rag_disabled(self, mock_secho, mock_config):
        """Test listing memories when RAG is disabled."""
        mock_config.get.return_value = False
        
        list_memories()
        
        # Should show warning about RAG being disabled
        assert any("Memory system is disabled" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_list_memories_no_rag_system(self, mock_secho, mock_get_rag, mock_config):
        """Test listing memories when RAG system fails to initialize."""
        mock_config.get.return_value = True
        mock_get_rag.return_value = None
        
        list_memories()
        
        assert any("Failed to initialize memory system" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_list_memories_empty(self, mock_secho, mock_get_rag, mock_config):
        """Test listing memories when none exist."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.list_documents.return_value = []
        mock_get_rag.return_value = mock_rag
        
        list_memories()
        
        assert any("No memories stored yet" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_list_memories_with_preview(self, mock_secho, mock_get_rag, mock_config):
        """Test listing memories with preview text."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.list_documents.return_value = [
            {
                'doc_id': 'test123456',
                'source': 'conversation',
                'indexed_at': '2024-01-01T12:00:00',
                'preview': 'This is a test preview that is longer than 100 characters and should be truncated when displayed in the memory list view',
                'metadata': {'topic': 'Test Topic'},
                'retrieval_count': 5
            }
        ]
        mock_get_rag.return_value = mock_rag
        
        list_memories(limit=10)
        
        # Check that preview is shown (truncated)
        assert any("This is a test preview" in str(call) for call in mock_secho.call_args_list)
        assert any("..." in str(call) for call in mock_secho.call_args_list)
        assert any("Test Topic" in str(call) for call in mock_secho.call_args_list)
        assert any("Retrieved: 5 times" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_list_memories_without_preview(self, mock_secho, mock_get_rag, mock_config):
        """Test listing memories without preview text."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.list_documents.return_value = [
            {
                'doc_id': 'test123456',
                'source': 'file',
                'indexed_at': '2024-01-01T12:00:00',
                'preview': None,  # No preview
                'metadata': {'filename': 'test.md'}
            }
        ]
        mock_get_rag.return_value = mock_rag
        
        list_memories(limit=10)
        
        # Should not crash, should show file info
        assert any("test.md" in str(call) for call in mock_secho.call_args_list)


class TestSearchMemories:
    """Test the search_memories function."""
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.commands.memory.typer.secho')
    def test_search_memories_rag_disabled(self, mock_secho, mock_config):
        """Test searching memories when RAG is disabled."""
        mock_config.get.return_value = False
        
        search_memories("test query")
        
        assert any("Memory system is disabled" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_search_memories_with_results(self, mock_secho, mock_get_rag, mock_config):
        """Test searching memories with results."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.search.return_value = {
            'results': [
                {
                    'content': 'This is test content that matches the search query',
                    'metadata': {
                        'doc_id': 'test123',
                        'source': 'conversation',
                        'topic': 'Test Topic'
                    },
                    'relevance_score': 0.95
                }
            ]
        }
        mock_get_rag.return_value = mock_rag
        
        search_memories("test query")
        
        mock_rag.search.assert_called_once_with("test query", n_results=10)
        assert any("Found 1 matches" in str(call) for call in mock_secho.call_args_list)
        assert any("relevance: 0.95" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_search_memories_no_results(self, mock_secho, mock_get_rag, mock_config):
        """Test searching memories with no results."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.search.return_value = {'results': []}
        mock_get_rag.return_value = mock_rag
        
        search_memories("nonexistent query")
        
        assert any("No matching memories found" in str(call) for call in mock_secho.call_args_list)


class TestShowMemory:
    """Test the show_memory function."""
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_show_memory_full_id(self, mock_secho, mock_get_rag, mock_config):
        """Test showing memory with full ID."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.get_document.return_value = {
            'doc_id': 'test123456789',
            'source': 'conversation',
            'indexed_at': '2024-01-01T12:00:00',
            'chunk_count': 3,
            'retrieval_count': 10,
            'preview': 'Test preview content',
            'metadata': {'topic': 'Test Topic'},
            'last_retrieved': '2024-01-02T12:00:00'
        }
        mock_get_rag.return_value = mock_rag
        
        show_memory("test123456789")
        
        mock_rag.get_document.assert_called_once_with("test123456789")
        assert any("Test preview content" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_show_memory_partial_id(self, mock_secho, mock_get_rag, mock_config):
        """Test showing memory with partial ID (8 chars)."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.list_documents.return_value = [
            {'doc_id': 'test123456789'}
        ]
        mock_rag.get_document.return_value = {
            'doc_id': 'test123456789',
            'source': 'file',
            'indexed_at': '2024-01-01T12:00:00',
            'chunk_count': 1,
            'retrieval_count': 0,
            'preview': None,
            'metadata': {}
        }
        mock_get_rag.return_value = mock_rag
        
        show_memory("test1234")
        
        mock_rag.get_document.assert_called_once_with("test123456789")
        assert any("Full content stored in chunks" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_show_memory_not_found(self, mock_secho, mock_get_rag, mock_config):
        """Test showing memory that doesn't exist."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.get_document.return_value = None
        mock_get_rag.return_value = mock_rag
        
        show_memory("nonexistent")
        
        assert any("Memory not found: nonexistent" in str(call) for call in mock_secho.call_args_list)


class TestForgetCommand:
    """Test the /forget command."""
    
    @patch('episodic.commands.memory.typer.secho')
    def test_forget_no_args(self, mock_secho):
        """Test /forget with no arguments shows usage."""
        forget_command()
        assert any("Usage: /forget" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.commands.memory.typer.secho')
    def test_forget_rag_disabled(self, mock_secho, mock_config):
        """Test /forget when RAG is disabled."""
        mock_config.get.return_value = False
        
        forget_command("test123")
        
        assert any("Memory system is disabled" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.confirm')
    @patch('episodic.commands.memory.typer.secho')
    def test_forget_all_confirmed(self, mock_secho, mock_confirm, mock_get_rag, mock_config):
        """Test /forget --all with confirmation."""
        mock_config.get.return_value = True
        mock_confirm.return_value = True
        mock_rag = MagicMock()
        mock_rag.clear_documents.return_value = 10
        mock_get_rag.return_value = mock_rag
        
        forget_command("--all")
        
        mock_confirm.assert_called_once()
        mock_rag.clear_documents.assert_called_once()
        assert any("Removed 10 memories" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.confirm')
    @patch('episodic.commands.memory.typer.secho')
    def test_forget_all_cancelled(self, mock_secho, mock_confirm, mock_get_rag, mock_config):
        """Test /forget --all cancelled."""
        mock_config.get.return_value = True
        mock_confirm.return_value = False
        mock_rag = MagicMock()
        mock_get_rag.return_value = mock_rag
        
        forget_command("--all")
        
        mock_confirm.assert_called_once()
        mock_rag.clear_documents.assert_not_called()
        assert any("Cancelled" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.confirm')
    @patch('episodic.commands.memory.typer.secho')
    def test_forget_contains(self, mock_secho, mock_confirm, mock_get_rag, mock_config):
        """Test /forget --contains."""
        mock_config.get.return_value = True
        mock_confirm.return_value = True
        mock_rag = MagicMock()
        mock_rag.search.return_value = {
            'results': [
                {'metadata': {'doc_id': 'doc1'}},
                {'metadata': {'doc_id': 'doc2'}},
                {'metadata': {'doc_id': 'doc1'}}  # Duplicate
            ]
        }
        mock_rag.remove_document.return_value = True
        mock_get_rag.return_value = mock_rag
        
        forget_command("--contains", "test", "text")
        
        mock_rag.search.assert_called_once_with("test text", n_results=50)
        assert mock_rag.remove_document.call_count == 2  # Should remove 2 unique docs
        assert any("Removed 2 memories" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_forget_source(self, mock_secho, mock_get_rag, mock_config):
        """Test /forget --source."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.clear_documents.return_value = 5
        mock_get_rag.return_value = mock_rag
        
        forget_command("--source", "file")
        
        mock_rag.clear_documents.assert_called_once_with(source_filter="file")
        assert any("Removed 5 memories from source: file" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_forget_specific_id(self, mock_secho, mock_get_rag, mock_config):
        """Test /forget with specific ID."""
        mock_config.get.return_value = True
        mock_rag = MagicMock()
        mock_rag.remove_document.return_value = True
        mock_get_rag.return_value = mock_rag
        
        forget_command("test123456789")
        
        mock_rag.remove_document.assert_called_once_with("test123456789")
        assert any("Removed memory: test1234" in str(call) for call in mock_secho.call_args_list)


class TestMemoryStatsCommand:
    """Test the /memory-stats command."""
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.commands.memory.typer.secho')
    def test_memory_stats_rag_disabled(self, mock_secho, mock_config):
        """Test memory stats when RAG is disabled."""
        mock_config.get.return_value = False
        
        memory_stats_command()
        
        assert any("Memory system is disabled" in str(call) for call in mock_secho.call_args_list)
    
    @patch('episodic.commands.memory.config')
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.commands.memory.typer.secho')
    def test_memory_stats_display(self, mock_secho, mock_get_rag, mock_config):
        """Test memory stats display."""
        mock_config.get.side_effect = lambda key, default=None: {
            'rag_enabled': True,
            'rag_auto_enhance': True,
            'rag_chunk_size': 1000,
            'rag_search_results': 5
        }.get(key, default)
        
        mock_rag = MagicMock()
        mock_rag.get_stats.return_value = {
            'total_documents': 100,
            'collection_count': 500,
            'avg_chunks_per_doc': 5.0,
            'total_retrievals': 1000,
            'source_distribution': {
                'conversation': 80,
                'file': 15,
                'web': 5
            },
            'db_size': 10485760,  # 10 MB
            'embedding_model': 'all-MiniLM-L6-v2',
            'recent_additions': [
                {'indexed_at': '2024-01-01T12:00:00', 'source': 'conversation'},
                {'indexed_at': '2024-01-01T11:00:00', 'source': 'file'}
            ]
        }
        mock_get_rag.return_value = mock_rag
        
        memory_stats_command()
        
        # Check that various stats are displayed
        assert any("Total documents: 100" in str(call) for call in mock_secho.call_args_list)
        assert any("Total chunks: 500" in str(call) for call in mock_secho.call_args_list)
        assert any("conversation: 80" in str(call) for call in mock_secho.call_args_list)
        assert any("Database size: 10.0 MB" in str(call) for call in mock_secho.call_args_list)
        assert any("RAG enabled: True" in str(call) for call in mock_secho.call_args_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])