"""
Document management functionality for RAG system.
"""

import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime


from episodic.db import get_connection


def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content for duplicate detection."""
    return hashlib.sha256(content.encode()).hexdigest()


def check_duplicate(content_hash: str) -> Optional[str]:
    """Check if a document with this content hash already exists."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT doc_id FROM rag_documents WHERE content_hash = ?",
            (content_hash,)
        )
        result = cursor.fetchone()
        return result[0] if result else None


def add_document_to_db(
    doc_id: str, 
    source: str, 
    metadata: Dict[str, Any],
    content_hash: str,
    chunk_count: int,
    preview: str = None
) -> None:
    """Add document metadata to database."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO rag_documents (doc_id, source, metadata, indexed_at, content_hash, chunk_count, preview)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, source, json.dumps(metadata), datetime.now().isoformat(), 
              content_hash, chunk_count, preview))


def list_documents(limit: Optional[int] = None, 
                  source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all indexed documents.
    
    Args:
        limit: Maximum number of documents to return
        source_filter: Filter by source (e.g., 'file', 'text', 'web')
        
    Returns:
        List of document metadata
    """
    with get_connection() as conn:
        query = "SELECT * FROM rag_documents"
        params = []
        
        if source_filter:
            query += " WHERE source = ?"
            params.append(source_filter)
            
        query += " ORDER BY indexed_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
            
        cursor = conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        docs = []
        for row in cursor:
            doc = dict(zip(columns, row))
            # Parse JSON metadata
            doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
            docs.append(doc)
            
    return docs


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific document by ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM rag_documents WHERE doc_id = ?",
            (doc_id,)
        )
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        
        if row:
            doc = dict(zip(columns, row))
            doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
            return doc
            
    return None


def remove_document(doc_id: str) -> bool:
    """Remove a document and its chunks from the index."""
    with get_connection() as conn:
        # Check if document exists
        cursor = conn.execute(
            "SELECT 1 FROM rag_documents WHERE doc_id = ?",
            (doc_id,)
        )
        if not cursor.fetchone():
            return False
            
        # Delete from database
        conn.execute("DELETE FROM rag_documents WHERE doc_id = ?", (doc_id,))
        
    return True


def clear_documents(source_filter: Optional[str] = None) -> int:
    """
    Clear all documents or documents from a specific source.
    
    Args:
        source_filter: If provided, only clear documents from this source
        
    Returns:
        Number of documents cleared
    """
    with get_connection() as conn:
        if source_filter:
            # Get count first
            cursor = conn.execute(
                "SELECT COUNT(*) FROM rag_documents WHERE source = ?",
                (source_filter,)
            )
            count = cursor.fetchone()[0]
            
            # Get all doc_ids to remove from vector store
            cursor = conn.execute(
                "SELECT doc_id FROM rag_documents WHERE source = ?",
                (source_filter,)
            )
            doc_ids = [row[0] for row in cursor]
            
            # Delete from database
            conn.execute(
                "DELETE FROM rag_documents WHERE source = ?",
                (source_filter,)
            )
        else:
            # Get count first
            cursor = conn.execute("SELECT COUNT(*) FROM rag_documents")
            count = cursor.fetchone()[0]
            
            # Get all doc_ids
            cursor = conn.execute("SELECT doc_id FROM rag_documents")
            doc_ids = [row[0] for row in cursor]
            
            # Clear entire table
            conn.execute("DELETE FROM rag_documents")
            
    return count, doc_ids


def get_source_distribution() -> Dict[str, int]:
    """Get distribution of documents by source."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT source, COUNT(*) as count 
            FROM rag_documents 
            GROUP BY source
        """)
        return dict(cursor.fetchall())


def get_document_stats() -> Dict[str, Any]:
    """Get statistics about indexed documents."""
    with get_connection() as conn:
        # Total documents
        cursor = conn.execute("SELECT COUNT(*) FROM rag_documents")
        total_docs = cursor.fetchone()[0]
        
        # Total chunks
        cursor = conn.execute("SELECT SUM(chunk_count) FROM rag_documents")
        total_chunks = cursor.fetchone()[0] or 0
        
        # Source distribution
        source_dist = get_source_distribution()
        
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "source_distribution": source_dist
    }


def record_retrieval(message: str, doc_ids: List[str], chunk_texts: List[str]) -> None:
    """Record which documents were retrieved for a message."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO rag_retrievals (message, retrieved_doc_ids, chunk_texts, retrieved_at)
            VALUES (?, ?, ?, ?)
        """, (
            message, 
            json.dumps(doc_ids), 
            json.dumps(chunk_texts),
            datetime.now().isoformat()
        ))


# Import json at the top of the file
import json