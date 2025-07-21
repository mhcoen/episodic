"""
Integration tests for memory management functionality.

Tests the complete memory system including database operations and RAG integration.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path

from episodic.commands.memory import (
    memory_command, forget_command, memory_stats_command
)
from episodic.config import config
from episodic.rag import get_rag_system, _rag_system
from episodic.db import get_connection


@pytest.fixture
def temp_episodic_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    original_home = os.environ.get('EPISODIC_HOME')
    os.environ['EPISODIC_HOME'] = temp_dir
    
    # Reset global RAG instance
    import episodic.rag
    episodic.rag._rag_system = None
    
    # Reset database connection pool
    from episodic.db_connection import close_pool
    close_pool()
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    if original_home:
        os.environ['EPISODIC_HOME'] = original_home
    else:
        os.environ.pop('EPISODIC_HOME', None)
    
    # Reset global RAG instance
    episodic.rag._rag_system = None
    
    # Reset database connection pool
    close_pool()


@pytest.fixture
def enable_rag():
    """Enable RAG for tests."""
    original_value = config.get('rag_enabled', False)
    config.set('rag_enabled', True)
    yield
    config.set('rag_enabled', original_value)


class TestMemoryIntegration:
    """Integration tests for memory commands."""
    
    def test_memory_lifecycle(self, temp_episodic_dir, enable_rag, capsys):
        """Test complete memory lifecycle: add, list, search, show, forget."""
        # Initialize database tables
        from episodic.db_migrations import initialize_db
        initialize_db(create_root_node=False)
        
        # Create RAG tables
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
        
        # Check if preview column exists and add if needed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'preview' not in columns:
                cursor.execute('ALTER TABLE rag_documents ADD COLUMN preview TEXT')
                conn.commit()
        
        # Get RAG system and add test documents
        rag = get_rag_system()
        assert rag is not None
        
        # Add test documents
        doc1_id, _ = rag.add_document(
            content="This is the first test document about Python programming and testing.",
            source="test",
            metadata={'category': 'programming'}
        )
        
        doc2_id, _ = rag.add_document(
            content="This is the second test document about machine learning and AI.",
            source="test",
            metadata={'category': 'ai'}
        )
        
        # Test listing memories
        capsys.readouterr()  # Clear buffer
        memory_command()
        output = capsys.readouterr().out
        assert "Memory Entries" in output
        assert doc1_id[:8] in output
        assert doc2_id[:8] in output
        assert "This is the first test document" in output
        assert "This is the second test document" in output
        
        # Test searching memories
        capsys.readouterr()
        memory_command("search", "Python")
        output = capsys.readouterr().out
        assert "Searching memories for: Python" in output
        assert "programming" in output
        
        # Test showing specific memory
        capsys.readouterr()
        memory_command("show", doc1_id[:8])
        output = capsys.readouterr().out
        assert f"Memory Entry: {doc1_id[:8]}" in output
        assert "Category: programming" in output
        
        # Test memory stats
        capsys.readouterr()
        memory_stats_command()
        output = capsys.readouterr().out
        assert "Memory System Statistics" in output
        # Should have our 2 test documents
        assert "test: 2" in output
        assert "test: 2" in output
        
        # Test forgetting a specific memory
        capsys.readouterr()
        forget_command(doc1_id[:8])
        output = capsys.readouterr().out
        assert f"Removed memory: {doc1_id[:8]}" in output
        
        # Verify it's gone
        capsys.readouterr()
        memory_command()
        output = capsys.readouterr().out
        assert doc1_id[:8] not in output
        assert doc2_id[:8] in output  # Second doc should still be there
    
    def test_forget_contains(self, temp_episodic_dir, enable_rag, capsys, monkeypatch):
        """Test forgetting memories containing specific text."""
        # Initialize database
        from episodic.db_migrations import initialize_db
        initialize_db(create_root_node=False)
        
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
        
        # Check if preview column exists and add if needed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'preview' not in columns:
                cursor.execute('ALTER TABLE rag_documents ADD COLUMN preview TEXT')
                conn.commit()
        
        # Add test documents
        rag = get_rag_system()
        
        rag.add_document(
            content="Document about Python programming",
            source="test"
        )
        rag.add_document(
            content="Document about JavaScript programming",
            source="test"
        )
        rag.add_document(
            content="Document about machine learning",
            source="test"
        )
        
        # Mock confirmation to yes
        monkeypatch.setattr('episodic.commands.memory.typer.confirm', lambda x: True)
        
        # Forget documents containing "programming"
        capsys.readouterr()
        forget_command("--contains", "programming")
        output = capsys.readouterr().out
        assert "Searching for memories containing: programming" in output
        assert "Removed 2 memories" in output
        
        # Verify only ML document remains
        capsys.readouterr()
        memory_command()
        output = capsys.readouterr().out
        assert "machine learning" in output
        assert "Python" not in output
        assert "JavaScript" not in output
    
    def test_forget_source(self, temp_episodic_dir, enable_rag, capsys):
        """Test forgetting memories from specific source."""
        # Initialize database
        from episodic.db_migrations import initialize_db
        initialize_db(create_root_node=False)
        
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
        
        # Check if preview column exists and add if needed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'preview' not in columns:
                cursor.execute('ALTER TABLE rag_documents ADD COLUMN preview TEXT')
                conn.commit()
        
        # Add documents from different sources
        rag = get_rag_system()
        
        rag.add_document(
            content="File content",
            source="file",
            metadata={'filename': 'test.txt'}
        )
        rag.add_document(
            content="Web content",
            source="web",
            metadata={'url': 'http://example.com'}
        )
        rag.add_document(
            content="Conversation content",
            source="conversation"
        )
        
        # Forget file sources
        capsys.readouterr()
        forget_command("--source", "file")
        output = capsys.readouterr().out
        assert "Removed 1 memories from source: file" in output
        
        # Verify file source is gone but others remain
        capsys.readouterr()
        memory_command()
        output = capsys.readouterr().out
        assert "Web content" in output
        assert "Conversation content" in output
        assert "File content" not in output
    
    def test_memory_pagination(self, temp_episodic_dir, enable_rag, capsys):
        """Test memory listing with pagination."""
        # Initialize database
        from episodic.db_migrations import initialize_db
        initialize_db(create_root_node=False)
        
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
        
        # Check if preview column exists and add if needed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'preview' not in columns:
                cursor.execute('ALTER TABLE rag_documents ADD COLUMN preview TEXT')
                conn.commit()
        
        # Add multiple documents
        rag = get_rag_system()
        for i in range(15):
            rag.add_document(
                content=f"Test document number {i} with some content",
                source="test",
                metadata={'index': i}
            )
        
        # Test default limit (10)
        capsys.readouterr()
        memory_command()
        output = capsys.readouterr().out
        assert "Showing 10 of 10 memories" in output
        
        # Test custom limit
        capsys.readouterr()
        memory_command("list", "5")
        output = capsys.readouterr().out
        assert "Showing 5 of 5 memories" in output
        
        # Test listing all
        capsys.readouterr()
        memory_command("list", "20")
        output = capsys.readouterr().out
        assert "Showing 15 of 15 memories" in output
    
    def test_empty_memory_system(self, temp_episodic_dir, enable_rag, capsys):
        """Test commands with empty memory system."""
        # Initialize database
        from episodic.db_migrations import initialize_db
        initialize_db(create_root_node=False)
        
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
        
        # Check if preview column exists and add if needed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'preview' not in columns:
                cursor.execute('ALTER TABLE rag_documents ADD COLUMN preview TEXT')
                conn.commit()
        
        # Test listing empty memories
        capsys.readouterr()
        memory_command()
        output = capsys.readouterr().out
        assert "No memories stored yet" in output
        assert "Memories are created automatically from conversations" in output
        
        # Test searching empty memories
        capsys.readouterr()
        memory_command("search", "test")
        output = capsys.readouterr().out
        assert "No matching memories found" in output
        
        # Test stats with empty system
        capsys.readouterr()
        memory_stats_command()
        output = capsys.readouterr().out
        assert "Total documents: 0" in output
    
    def test_preview_truncation(self, temp_episodic_dir, enable_rag, capsys):
        """Test that long content is properly truncated in preview."""
        # Initialize database
        from episodic.db_migrations import initialize_db
        initialize_db(create_root_node=False)
        
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
        
        # Check if preview column exists and add if needed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'preview' not in columns:
                cursor.execute('ALTER TABLE rag_documents ADD COLUMN preview TEXT')
                conn.commit()
        
        # Add document with long content
        rag = get_rag_system()
        long_content = "This is a very long document. " * 50  # Make it really long
        doc_id, _ = rag.add_document(
            content=long_content,
            source="test"
        )
        
        # Check that preview is truncated
        capsys.readouterr()
        memory_command()
        output = capsys.readouterr().out
        
        # The preview should be truncated and end with ...
        assert "..." in output
        # But the full long content should not be in the list view
        assert long_content not in output
        
        # The preview should be reasonable length (around 100 chars shown)
        lines = output.split('\n')
        preview_line = next((line for line in lines if "This is a very long document" in line), None)
        assert preview_line is not None
        assert len(preview_line) < 150  # Reasonable line length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])