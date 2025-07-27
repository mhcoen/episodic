"""
Adapter to make EpisodicRAG work with the multi-collection system.

This provides backward compatibility while transitioning to the new architecture.
"""

from typing import List, Dict, Any, Optional, Tuple

from episodic.rag_collections import get_multi_collection_rag, CollectionType
from episodic.config import config
from episodic.debug_utils import debug_print


class EpisodicRAGAdapter:
    """
    Adapter class that provides the same interface as EpisodicRAG
    but uses the multi-collection system underneath.
    """
    
    def __init__(self):
        """Initialize the adapter."""
        self.multi_rag = get_multi_collection_rag()
        
        # For backward compatibility, track which collection to use by default
        self._default_collection = CollectionType.USER_DOCS
        
        # Ensure SQL database tables exist
        from episodic.db_rag import create_rag_tables
        create_rag_tables()
    
    def chunk_document(self, content: str, chunk_size: int = None, 
                      overlap: int = None) -> List[Tuple[str, Dict[str, int]]]:
        """Split a document into overlapping chunks."""
        if chunk_size is None:
            chunk_size = config.get("rag_chunk_size", 1000)
        if overlap is None:
            overlap = config.get("rag_chunk_overlap", 200)
        
        # Simple character-based chunking (same as original)
        chunks = []
        
        if len(content) <= chunk_size:
            chunks.append((content, {"start": 0, "end": len(content)}))
            return chunks
        
        start = 0
        while start < len(content):
            end = start + chunk_size
            
            if end < len(content):
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, {"start": start, "end": end}))
            
            start = end - overlap
            if start <= chunks[-1][1]["start"]:
                start = chunks[-1][1]["end"]
                
        return chunks
    
    def add_document(self, 
                    content: str, 
                    source: str, 
                    metadata: Optional[Dict[str, Any]] = None,
                    chunk: bool = True) -> Tuple[str, int]:
        """Add a document to the appropriate collection based on source."""
        # Determine collection based on source
        if source == 'conversation':
            collection_type = CollectionType.CONVERSATION
        else:
            collection_type = CollectionType.USER_DOCS
        
        # Use multi-collection RAG
        return self.multi_rag.add_document(
            content=content,
            source=source,
            metadata=metadata,
            collection_type=collection_type,
            chunk=chunk
        )
    
    def search(self, 
              query: str, 
              n_results: int = None,
              source_filter: Optional[str] = None) -> Dict[str, Any]:
        """Search for relevant documents."""
        if n_results is None:
            n_results = config.get("rag_search_results", 5)
        
        # Determine which collection(s) to search based on source_filter
        collection_type = None
        if source_filter == 'conversation':
            collection_type = CollectionType.CONVERSATION
        elif source_filter and source_filter != 'conversation':
            collection_type = CollectionType.USER_DOCS
        # If no source_filter, search all collections
        
        return self.multi_rag.search(
            query=query,
            n_results=n_results,
            collection_type=collection_type,
            source_filter=source_filter
        )
    
    def list_documents(self, limit: Optional[int] = None, 
                      source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        # This needs to be implemented with SQL queries
        from episodic.rag_document_manager import list_documents as _list_documents
        return _list_documents(limit, source_filter)
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        from episodic.rag_document_manager import get_document as _get_document
        return _get_document(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the index."""
        # This needs both ChromaDB and SQL operations
        # For now, return False to indicate not implemented
        debug_print(f"Document removal not yet implemented in multi-collection system", category="rag")
        return False
    
    def clear_documents(self, source_filter: Optional[str] = None) -> int:
        """Clear documents based on source filter."""
        if source_filter == 'conversation':
            return self.multi_rag.clear_collection(CollectionType.CONVERSATION)
        elif source_filter:
            # Clear only documents with specific source from user docs
            # This would need more complex filtering
            debug_print(f"Source-specific clearing not yet implemented", category="rag")
            return 0
        else:
            # Clear all user documents (not conversations)
            return self.multi_rag.clear_collection(CollectionType.USER_DOCS)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        # Get stats from multi-collection system
        collection_stats = self.multi_rag.get_collection_stats()
        
        # Get document stats from SQL
        from episodic.rag_document_manager import get_document_stats
        doc_stats = get_document_stats()
        
        # Combine stats
        stats = doc_stats.copy()
        
        # Add collection counts
        total_count = 0
        for coll_type, coll_info in collection_stats.items():
            total_count += coll_info['count']
        
        stats['collection_count'] = total_count
        stats['collection_breakdown'] = collection_stats
        stats['embedding_model'] = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
        
        return stats
    
    def get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of documents by source."""
        from episodic.rag_document_manager import get_source_distribution
        return get_source_distribution()
    
    # Add compatibility properties
    @property
    def collection(self):
        """For backward compatibility - returns user docs collection."""
        return self.multi_rag.get_collection(CollectionType.USER_DOCS)
    
    @property
    def embedding_function(self):
        """For backward compatibility."""
        return self.multi_rag.embedding_function