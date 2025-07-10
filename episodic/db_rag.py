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
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                content_hash TEXT UNIQUE,
                parent_doc_id TEXT,
                chunk_index INTEGER DEFAULT 0
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
        c.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_doc ON rag_retrievals(document_id)')
        
        conn.commit()