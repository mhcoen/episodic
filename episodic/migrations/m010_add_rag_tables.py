"""
Migration to add RAG (Retrieval Augmented Generation) tables.

This migration creates tables to track indexed documents and retrievals
for the RAG system.
"""

def up(conn):
    """Apply the migration."""
    cursor = conn.cursor()
    
    # Create rag_documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            metadata TEXT,
            content_hash TEXT UNIQUE,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create rag_retrievals table for tracking usage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_retrievals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            query TEXT NOT NULL,
            retrieved_at TIMESTAMP NOT NULL,
            FOREIGN KEY (document_id) REFERENCES rag_documents(id)
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_documents_source ON rag_documents(source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_documents_hash ON rag_documents(content_hash)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_document ON rag_retrievals(document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_time ON rag_retrievals(retrieved_at)')
    
    conn.commit()


def down(conn):
    """Rollback the migration."""
    cursor = conn.cursor()
    
    # Drop indexes
    cursor.execute('DROP INDEX IF EXISTS idx_rag_retrievals_time')
    cursor.execute('DROP INDEX IF EXISTS idx_rag_retrievals_document')
    cursor.execute('DROP INDEX IF EXISTS idx_rag_documents_hash')
    cursor.execute('DROP INDEX IF EXISTS idx_rag_documents_source')
    
    # Drop tables
    cursor.execute('DROP TABLE IF EXISTS rag_retrievals')
    cursor.execute('DROP TABLE IF EXISTS rag_documents')
    
    conn.commit()