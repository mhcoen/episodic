"""
RAG (Retrieval Augmented Generation) operations for Episodic database.

This module handles database operations for the RAG system.
"""

import logging

from .db_connection import get_connection

# Set up logging
logger = logging.getLogger(__name__)


def create_rag_tables():
    """Create tables for RAG functionality."""
    with get_connection() as conn:
        c = conn.cursor()
        
        # Table for tracking indexed documents
        c.execute('''
            CREATE TABLE IF NOT EXISTS rag_documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                metadata JSON,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT UNIQUE,
                chunk_count INTEGER DEFAULT 1,
                preview TEXT
            )
        ''')
        
        # Table for tracking which documents were used in responses
        c.execute('''
            CREATE TABLE IF NOT EXISTS rag_retrievals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT REFERENCES nodes(id),
                document_id TEXT,
                relevance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                was_helpful BOOLEAN DEFAULT NULL
            )
        ''')
        
        # Create indices for performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_rag_documents_hash ON rag_documents(content_hash)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_rag_documents_source ON rag_documents(source)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_retrieved_at ON rag_retrievals(retrieved_at)')
        
        conn.commit()