"""
Migration 011: Fix rag_retrievals schema to match usage

The rag_retrievals table schema doesn't match what the code expects.
This migration updates the schema to match the actual usage.
"""

def migrate(cursor):
    """Apply migration."""
    
    # Drop the existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS rag_retrievals')
    
    # Create the table with the correct schema that matches the code usage
    cursor.execute('''
        CREATE TABLE rag_retrievals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            retrieved_doc_ids TEXT,  -- JSON array of document IDs
            chunk_texts TEXT,        -- JSON array of chunk texts
            retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_retrieved_at ON rag_retrievals(retrieved_at)')
    
    print("✅ Fixed rag_retrievals table schema")


def rollback(cursor):
    """Rollback migration."""
    # Drop the fixed table
    cursor.execute('DROP TABLE IF EXISTS rag_retrievals')
    
    # Recreate the original (incorrect) schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_retrievals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT REFERENCES nodes(id),
            document_id TEXT,
            relevance_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_doc ON rag_retrievals(document_id)')
    
    print("⏪ Rolled back rag_retrievals table schema")