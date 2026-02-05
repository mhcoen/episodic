#!/usr/bin/env python3
"""
Integration tests for RAG (Retrieval Augmented Generation) functionality.
"""

import pytest

import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

from episodic.config import config
from episodic.db import initialize_db
from episodic.rag import EpisodicRAG, get_rag_system


class DummyEmbeddingFunction:
    """Lightweight embedding stub for offline tests.

    Note: ChromaDB expects __call__ to have parameter named 'input'.
    """

    EMBEDDING_DIM = 384  # Matches all-MiniLM-L6-v2

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input):
        """Generate deterministic embeddings from text hashes."""
        if isinstance(input, str):
            input = [input]
        results = []
        for text in input:
            # Create deterministic vector from text hash
            text_hash = hash(text) & 0xFFFFFFFF
            vec = []
            for i in range(self.EMBEDDING_DIM):
                val = ((text_hash * (i + 1)) % 1000) / 1000.0
                vec.append(val - 0.5)
            results.append(vec)
        return results

    def embed_query(self, input=None, text=None):
        """Embed a single query. Returns list of embeddings."""
        query = input if input is not None else text
        # Handle both single string and list input
        if isinstance(query, list):
            return self.__call__(query)
        return self.__call__([query])

    def embed_documents(self, input=None, texts=None):
        """Embed multiple documents."""
        docs = input if input is not None else texts
        if isinstance(docs, str):
            docs = [docs]
        return self.__call__(docs)

    def name(self):
        return "dummy-embedding"

    def is_legacy(self):
        return True


class TestRAGIntegration(unittest.TestCase):
    """Test RAG system integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        # Use temporary directory for test database
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_home = os.environ.get('EPISODIC_HOME')
        cls.original_user_home = os.environ.get('HOME')
        os.environ['EPISODIC_HOME'] = cls.temp_dir
        os.environ['HOME'] = cls.temp_dir
        cls.db_path = os.environ['EPISODIC_DB_PATH']

        cls._embedding_patcher = patch(
            'episodic.rag_utils.SilentSentenceTransformerEmbeddingFunction',
            DummyEmbeddingFunction
        )
        cls._embedding_patcher.start()
        
        # Initialize database with RAG tables
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        initialize_db(migrate=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.original_home:
            os.environ['EPISODIC_HOME'] = cls.original_home
        else:
            os.environ.pop('EPISODIC_HOME', None)
        if cls.original_user_home:
            os.environ['HOME'] = cls.original_user_home
        else:
            os.environ.pop('HOME', None)
        cls._embedding_patcher.stop()

        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        
        # Clean up temp directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up each test."""
        # Enable RAG
        config.set('rag_enabled', True)
        
        # Create fresh RAG instance
        self.rag = EpisodicRAG()
        self.rag.clear_documents()
    
    def test_document_indexing(self):
        """Test basic document indexing."""
        content = "Python is a high-level programming language known for its simplicity."
        source = "test_doc.txt"
        
        doc_id, chunk_count = self.rag.add_document(content, source)
        
        self.assertIsInstance(doc_id, str)
        self.assertEqual(chunk_count, 1)  # Single chunk for short document
        
        # Verify document was stored
        doc = self.rag.get_document(doc_id)
        self.assertIsNotNone(doc)
        self.assertEqual(doc['metadata']['source'], source)
        self.assertEqual(doc['chunk_count'], chunk_count)
    
    def test_duplicate_detection(self):
        """Test that duplicate documents are detected."""
        content = "This is a test document for duplicate detection."
        source1 = "doc1.txt"
        source2 = "doc2.txt"
        
        # Add document first time
        doc_id1, chunk_count1 = self.rag.add_document(content, source1)
        self.assertEqual(chunk_count1, 1)
        
        # Try to add same content again
        doc_id2, chunk_count2 = self.rag.add_document(content, source2)
        self.assertEqual(doc_id1, doc_id2)  # Should return same ID
        self.assertEqual(chunk_count1, chunk_count2)
    
    def test_document_chunking(self):
        """Test document chunking for large documents."""
        # Create a large document
        words = ["word"] * 1000
        content = " ".join(words)
        source = "large_doc.txt"
        
        # Configure chunking
        config.set('rag_chunk_size', 100)
        config.set('rag_chunk_overlap', 20)
        
        doc_id, chunk_count = self.rag.add_document(content, source, chunk=True)
        
        # Should have multiple chunks
        self.assertGreater(chunk_count, 1)
        
        # Verify document metadata includes chunk count
        doc = self.rag.get_document(doc_id)
        self.assertEqual(doc['chunk_count'], chunk_count)
    
    def test_search_functionality(self):
        """Test search functionality."""
        # Add test documents
        docs = [
            ("Python is great for data science and machine learning.", "ml_guide.txt"),
            ("JavaScript is essential for web development.", "web_dev.txt"),
            ("Machine learning requires understanding of statistics.", "stats_ml.txt")
        ]
        
        for content, source in docs:
            self.rag.add_document(content, source)
        
        # Search for machine learning content
        results = self.rag.search("machine learning", n_results=3)
        
        self.assertIn('results', results)
        self.assertIn('total', results)
        
        # Should find at least 2 relevant documents
        self.assertGreaterEqual(results['total'], 2)
        
        # Check result structure
        for result in results['results']:
            self.assertIn('content', result)
            self.assertIn('metadata', result)
            # Note: With dummy embeddings, relevance scores are negative (distance-based)
            # Real embedding functions would produce scores in [0, 1] range
            if result['relevance_score'] is not None:
                # Just verify it's a number (dummy embeddings produce negative values)
                self.assertIsInstance(result['relevance_score'], (int, float))
    
    def test_context_enhancement(self):
        """Test context enhancement for messages."""
        # Add some knowledge base content
        self.rag.add_document(
            "The capital of France is Paris. It's known for the Eiffel Tower.",
            "geography.txt"
        )
        self.rag.add_document(
            "Python was created by Guido van Rossum in 1991.",
            "python_history.txt"
        )
        
        # Test enhancement
        message = "Tell me about Python"
        enhanced_msg = self.rag.enhance_with_context(message, n_results=2)
        
        # Should include original message
        self.assertIn(message, enhanced_msg)
        
        # Should include context
        self.assertIn("Guido van Rossum", enhanced_msg)
        
        # Should include source context
        self.assertIn("python_history.txt", enhanced_msg)
    
    def test_document_management(self):
        """Test document listing and removal."""
        # Add test documents
        doc1_ids = self.rag.add_document("Test document 1", "test1.txt")
        doc2_ids = self.rag.add_document("Test document 2", "test2.txt")
        
        # List documents
        docs = self.rag.list_documents()
        self.assertGreaterEqual(len(docs), 2)
        
        # Remove a document
        success = self.rag.remove_document(doc1_ids[0])
        self.assertTrue(success)
        
        # Verify removal
        doc = self.rag.get_document(doc1_ids[0])
        self.assertIsNone(doc)
        
        # Clear all documents
        count = self.rag.clear_documents()
        self.assertGreater(count, 0)
        
        # Verify all cleared
        docs = self.rag.list_documents()
        self.assertEqual(len(docs), 0)
    
    def test_source_filtering(self):
        """Test filtering by source."""
        # Add documents from different sources
        self.rag.add_document("PDF content 1", "pdf")
        self.rag.add_document("PDF content 2", "pdf")
        self.rag.add_document("Text content", "text")
        
        # Filter by PDF files
        pdf_docs = self.rag.list_documents(source_filter="pdf")
        self.assertEqual(len(pdf_docs), 2)
        
        # Clear only PDF documents
        count = self.rag.clear_documents(source_filter="pdf")
        self.assertEqual(count, 2)
        
        # Verify only text file remains
        remaining = self.rag.list_documents()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['source'], "text")
    
    def test_search_source_filtering(self):
        """Test search with source filtering."""
        self.rag.add_document("RAG source doc 1", "docs")
        self.rag.add_document("RAG source doc 2", "docs")
        self.rag.add_document("Other source doc", "notes")

        results = self.rag.search("RAG source", n_results=5, source_filter="docs")

        self.assertGreater(results['total'], 0)
        for result in results['results']:
            self.assertEqual(result['metadata'].get('source'), "docs")
    
    def test_statistics(self):
        """Test RAG statistics."""
        # Add some documents
        self.rag.add_document("Test 1", "test1.txt")
        self.rag.add_document("Test 2", "test2.txt")
        
        stats = self.rag.get_stats()
        
        self.assertIn('total_documents', stats)
        self.assertIn('total_chunks', stats)
        self.assertIn('embedding_model', stats)
        self.assertIn('collection_count', stats)
        self.assertIn('source_distribution', stats)
        
        self.assertGreaterEqual(stats['total_documents'], 2)
        self.assertGreaterEqual(stats['total_chunks'], 2)
        self.assertGreaterEqual(stats['collection_count'], stats['total_chunks'])


class TestRAGCommands(unittest.TestCase):
    """Test RAG CLI commands."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the RAG system
        self.mock_rag = MagicMock()
        
        # Patch get_rag_system
        self.patcher = patch('episodic.commands.rag.get_rag_system')
        self.mock_get_rag = self.patcher.start()
        self.mock_get_rag.return_value = self.mock_rag
        self.init_patcher = patch('episodic.rag.ensure_rag_initialized', return_value=True)
        self.init_patcher.start()
        
        # Enable RAG
        config.set('rag_enabled', True)
    
    def tearDown(self):
        """Clean up."""
        self.patcher.stop()
        self.init_patcher.stop()
    
    def test_search_command(self):
        """Test /search command."""
        from episodic.commands.rag import search
        
        # Mock search results
        self.mock_rag.search.return_value = {
            'results': [{
                'content': 'Test document content',
                'metadata': {'source': 'test.txt', 'doc_id': 'test-id'},
                'relevance_score': 0.9
            }],
            'total': 1
        }
        
        # Run search
        search("test query")
        
        # Verify search was called
        self.mock_rag.search.assert_called_once()
        args = self.mock_rag.search.call_args
        self.assertEqual(args[0][0], "test query")
    
    def test_index_text_command(self):
        """Test /index --text command."""
        from episodic.commands.rag import index_text
        
        # Mock add_document
        self.mock_rag.add_document.return_value = ('doc-id', 1)
        
        # Run index
        index_text("Test content to index")
        
        # Verify document was added
        self.mock_rag.add_document.assert_called_once()
        args = self.mock_rag.add_document.call_args
        self.assertEqual(args[0][0], "Test content to index")
        self.assertEqual(args[0][1], "manual_input")
    
    def test_index_file_command(self):
        """Test /index <file> command."""
        from episodic.commands.rag import index_file
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_file = f.name
        
        try:
            # Mock add_document
            self.mock_rag.add_document.return_value = ('doc-id', 1)
            
            # Run index
            index_file(temp_file)
            
            # Verify document was added
            self.mock_rag.add_document.assert_called_once()
            args = self.mock_rag.add_document.call_args
            self.assertEqual(args[0][0], "Test file content")
            self.assertEqual(args[0][1], os.path.basename(temp_file))
        finally:
            os.unlink(temp_file)
    
    def test_docs_list_command(self):
        """Test /docs list command."""
        from episodic.commands.rag import docs_command
        
        # Mock list_documents
        self.mock_rag.list_documents.return_value = [
            {
                'id': 'doc1',
                'source': 'test1.txt',
                'word_count': 100,
                'indexed_at': '2023-01-01T00:00:00',
                'preview': 'Preview one'
            },
            {
                'id': 'doc2',
                'source': 'test2.txt',
                'word_count': 200,
                'indexed_at': '2023-01-02T00:00:00',
                'preview': 'Preview two'
            }
        ]
        
        # Run docs list
        docs_command("list")
        
        # Verify list was called
        self.mock_rag.list_documents.assert_called_once()
    
    def test_rag_toggle_command(self):
        """Test /rag on/off command."""
        from episodic.commands.rag import rag_toggle
        
        # Test enabling
        rag_toggle(True)
        self.assertTrue(config.get('rag_enabled'))
        
        # Test disabling
        rag_toggle(False)
        self.assertFalse(config.get('rag_enabled'))


class TestRAGCollectionIsolation(unittest.TestCase):
    """Test that user docs and conversation memory are properly isolated.

    This tests the security fix for the bug where conversation content
    was leaking into RAG search results.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_home = os.environ.get('EPISODIC_HOME')
        cls.original_user_home = os.environ.get('HOME')
        os.environ['EPISODIC_HOME'] = cls.temp_dir
        os.environ['HOME'] = cls.temp_dir
        cls.db_path = os.environ.get('EPISODIC_DB_PATH', os.path.join(cls.temp_dir, '.episodic', 'episodic.db'))

        # Patch embedding function in all modules that import it at module level
        cls._embedding_patcher = patch(
            'episodic.rag_utils.SilentSentenceTransformerEmbeddingFunction',
            DummyEmbeddingFunction
        )
        cls._embedding_patcher2 = patch(
            'episodic.rag_collections.SilentSentenceTransformerEmbeddingFunction',
            DummyEmbeddingFunction
        )
        cls._embedding_patcher3 = patch(
            'episodic.rag_migration.SilentSentenceTransformerEmbeddingFunction',
            DummyEmbeddingFunction
        )
        cls._embedding_patcher.start()
        cls._embedding_patcher2.start()
        cls._embedding_patcher3.start()

        # Initialize database
        os.makedirs(os.path.dirname(cls.db_path), exist_ok=True)
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        initialize_db(migrate=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.original_home:
            os.environ['EPISODIC_HOME'] = cls.original_home
        else:
            os.environ.pop('EPISODIC_HOME', None)
        if cls.original_user_home:
            os.environ['HOME'] = cls.original_user_home
        else:
            os.environ.pop('HOME', None)
        cls._embedding_patcher.stop()
        cls._embedding_patcher2.stop()
        cls._embedding_patcher3.stop()

        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_user_doc_search_excludes_conversation(self):
        """Test that searching user docs doesn't return conversation memory."""
        try:
            from episodic.rag_adapter import EpisodicRAGAdapter
            from episodic.rag_collections import CollectionType
        except ImportError as e:
            self.skipTest(f"RAG imports failed (likely SOCKS proxy issue): {e}")

        try:
            adapter = EpisodicRAGAdapter()
        except Exception as e:
            if "SOCKS" in str(e) or "socksio" in str(e):
                self.skipTest(f"Embeddings unavailable due to SOCKS proxy: {e}")
            raise

        # Add a user document
        adapter.add_document(
            content="Python is a programming language used for data science.",
            source="file",  # This routes to USER_DOCS
            metadata={"filename": "python_guide.txt"}
        )

        # Add a conversation memory
        adapter.add_document(
            content="User asked about secret API keys and passwords.",
            source="conversation",  # This routes to CONVERSATION
            metadata={"user_id": "test-user-123"}
        )

        # Search with source_filter='file' should ONLY return user docs
        results = adapter.search("python", n_results=10, source_filter='file')

        # Should find the user document
        self.assertGreater(results['total'], 0)

        # Should NOT contain conversation content
        for result in results['results']:
            content = result.get('content', '')
            self.assertNotIn('secret API keys', content)
            self.assertNotIn('passwords', content)

    def test_conversation_search_excludes_user_docs(self):
        """Test that searching conversation memory doesn't return user docs."""
        try:
            from episodic.rag_adapter import EpisodicRAGAdapter
        except ImportError as e:
            self.skipTest(f"RAG imports failed (likely SOCKS proxy issue): {e}")

        try:
            adapter = EpisodicRAGAdapter()
        except Exception as e:
            if "SOCKS" in str(e) or "socksio" in str(e):
                self.skipTest(f"Embeddings unavailable due to SOCKS proxy: {e}")
            raise

        # Add a user document with sensitive-looking content
        adapter.add_document(
            content="Company confidential: quarterly revenue report.",
            source="file",
            metadata={"filename": "revenue.txt"}
        )

        # Add a conversation memory
        adapter.add_document(
            content="User discussed their favorite programming languages.",
            source="conversation",
            metadata={"user_id": "test-user-456"}
        )

        # Search with source_filter='conversation' should ONLY return memories
        results = adapter.search("confidential revenue", n_results=10, source_filter='conversation')

        # Should NOT contain user document content
        for result in results['results']:
            content = result.get('content', '')
            self.assertNotIn('quarterly revenue', content)
            self.assertNotIn('Company confidential', content)

    def test_no_filter_searches_all_collections_warning(self):
        """Test that searching without filter searches all (and log warning in debug)."""
        try:
            from episodic.rag_adapter import EpisodicRAGAdapter
        except ImportError as e:
            self.skipTest(f"RAG imports failed (likely SOCKS proxy issue): {e}")

        try:
            adapter = EpisodicRAGAdapter()
        except Exception as e:
            if "SOCKS" in str(e) or "socksio" in str(e):
                self.skipTest(f"Embeddings unavailable due to SOCKS proxy: {e}")
            raise

        # Add to both collections
        adapter.add_document("User doc content about testing.", source="file")
        adapter.add_document("Conversation about testing.", source="conversation")

        # Search without filter - this searches ALL collections
        # This is the behavior that was causing the leak; now it's explicit
        results = adapter.search("testing", n_results=10)

        # Without filter, should find results from both collections
        # This test documents the current behavior
        self.assertGreaterEqual(results['total'], 1)


if __name__ == '__main__':
    unittest.main()
