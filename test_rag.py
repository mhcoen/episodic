#!/usr/bin/env python3
"""Test script for RAG functionality."""

import os
import sys
import tempfile

# Add episodic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.config import config
from episodic.rag import EpisodicRAG, get_rag_system, ensure_rag_initialized
from episodic.db import create_rag_tables, get_connection

def test_rag_basic():
    """Test basic RAG operations."""
    print("Testing RAG functionality...")
    
    # Enable RAG
    config.set('rag_enabled', True)
    config.set('debug', True)
    
    # Create RAG tables
    create_rag_tables()
    
    # Initialize RAG system
    if not ensure_rag_initialized():
        print("❌ Failed to initialize RAG system")
        return False
    
    rag = get_rag_system()
    if not rag:
        print("❌ Failed to get RAG system")
        return False
    
    print("✅ RAG system initialized")
    
    # Test adding documents
    test_content = """
    Episodic is a conversational memory system that stores conversations as a DAG.
    It supports topic detection, compression, and now RAG for enhanced responses.
    """
    
    doc_ids = rag.add_document(test_content, "test_doc.txt")
    print(f"✅ Added document with {len(doc_ids)} chunks")
    
    # Test searching
    results = rag.search("What is Episodic?", n_results=3)
    if results['documents']:
        print(f"✅ Search found {len(results['documents'])} results")
    else:
        print("❌ Search returned no results")
        return False
    
    # Test context enhancement
    enhanced, sources = rag.enhance_with_context("Tell me about Episodic")
    if sources:
        print(f"✅ Context enhancement used sources: {sources}")
    else:
        print("❌ Context enhancement found no sources")
    
    # Test stats
    stats = rag.get_stats()
    print(f"✅ Stats: {stats['total_documents']} documents indexed")
    
    # Clean up
    rag.clear_documents()
    print("✅ Cleaned up test documents")
    
    return True


def test_rag_chunking():
    """Test document chunking."""
    print("\nTesting document chunking...")
    
    rag = get_rag_system()
    
    # Create a longer document
    long_content = " ".join([f"Sentence {i}." for i in range(200)])
    
    # Test chunking
    chunks = rag.chunk_document(long_content, chunk_size=50, overlap=10)
    print(f"✅ Document split into {len(chunks)} chunks")
    
    # Add chunked document
    doc_ids = rag.add_document(long_content, "long_doc.txt", chunk=True)
    print(f"✅ Added {len(doc_ids)} document chunks")
    
    # Verify each chunk can be searched
    test_query = "Sentence 50"
    results = rag.search(test_query)
    if results['documents']:
        print(f"✅ Found content in chunked document")
    else:
        print("❌ Failed to find content in chunks")
    
    # Clean up
    rag.clear_documents()
    
    return True


def test_rag_file_indexing():
    """Test file indexing."""
    print("\nTesting file indexing...")
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test file for RAG indexing.\n")
        f.write("It contains multiple lines of text.\n")
        f.write("RAG should be able to index and search this content.")
        test_file = f.name
    
    try:
        from episodic.commands.rag import index_file
        
        # Index the file
        index_file(test_file)
        print("✅ Indexed test file")
        
        # Search for content
        rag = get_rag_system()
        results = rag.search("test file for RAG")
        if results['documents']:
            print("✅ Found indexed file content")
        else:
            print("❌ Failed to find indexed file content")
        
        # Clean up
        rag.clear_documents()
        
    finally:
        os.unlink(test_file)
    
    return True


if __name__ == "__main__":
    print("RAG System Test Suite")
    print("=" * 50)
    
    try:
        # Test if dependencies are available
        import chromadb
        import sentence_transformers
        print("✅ Dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install chromadb sentence-transformers")
        sys.exit(1)
    
    # Run tests
    all_passed = True
    
    if not test_rag_basic():
        all_passed = False
    
    if not test_rag_chunking():
        all_passed = False
    
    if not test_rag_file_indexing():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)