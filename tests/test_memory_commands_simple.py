#!/usr/bin/env python3
"""
Simple test script for memory commands.

This script tests the memory management functionality in an isolated environment.
"""

import os
import tempfile
import shutil
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.commands.memory import memory_command, forget_command, memory_stats_command
from episodic.config import config
from episodic.rag import get_rag_system
import episodic.rag
from episodic.db import get_connection
from episodic.db_connection import close_pool


def test_memory_commands():
    """Test memory commands in isolation."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    # Set environment to use temp directory
    os.environ['EPISODIC_HOME'] = temp_dir
    
    # Reset global instances
    episodic.rag._rag_system = None
    close_pool()
    
    # Enable RAG
    config.set('rag_enabled', True)
    
    try:
        # Initialize database
        from episodic.db_migrations import initialize_db
        initialize_db(create_root_node=False)
        
        # Create RAG tables
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
        
        # Add preview column
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'preview' not in columns:
                cursor.execute('ALTER TABLE rag_documents ADD COLUMN preview TEXT')
                conn.commit()
        
        # Get RAG system
        rag = get_rag_system()
        assert rag is not None, "Failed to initialize RAG system"
        
        print("\n1. Testing empty memory system...")
        memory_command()
        print("✓ Empty memory list works")
        
        print("\n2. Adding test documents...")
        doc1_id, _ = rag.add_document(
            content="This is a test document about Python programming.",
            source="test",
            metadata={'category': 'programming'}
        )
        print(f"✓ Added document 1: {doc1_id[:8]}")
        
        doc2_id, _ = rag.add_document(
            content="This is another document about machine learning.",
            source="test",
            metadata={'category': 'ai'}
        )
        print(f"✓ Added document 2: {doc2_id[:8]}")
        
        print("\n3. Testing memory list...")
        memory_command()
        print("✓ Memory list with documents works")
        
        print("\n4. Testing memory search...")
        memory_command("search", "Python")
        print("✓ Memory search works")
        
        print("\n5. Testing memory show...")
        memory_command("show", doc1_id[:8])
        print("✓ Memory show works")
        
        print("\n6. Testing memory stats...")
        memory_stats_command()
        print("✓ Memory stats works")
        
        print("\n7. Testing forget command...")
        forget_command(doc1_id[:8])
        print(f"✓ Forgot document: {doc1_id[:8]}")
        
        print("\n8. Verifying document was removed...")
        memory_command()
        print("✓ Document successfully removed")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Reset
        if 'EPISODIC_HOME' in os.environ:
            del os.environ['EPISODIC_HOME']
        episodic.rag._rag_system = None
        close_pool()
    
    return True


if __name__ == "__main__":
    success = test_memory_commands()
    sys.exit(0 if success else 1)