"""
Multi-collection RAG system for Episodic.

This module provides a RAG system that manages multiple ChromaDB collections
for different types of content (conversation memories, user documents, etc).
"""

import os
import warnings
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Suppress ChromaDB warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger('chromadb').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*telemetry.*")

import chromadb
from chromadb.utils import embedding_functions

from episodic.config import config
from episodic.debug_utils import debug_print
from episodic.rag_utils import suppress_chromadb_telemetry


class CollectionType:
    """Constants for collection types."""
    USER_DOCS = 'user_docs'
    CONVERSATION = 'conversation'
    HELP = 'help'  # Managed separately by help.py


class MultiCollectionRAG:
    """RAG system that manages multiple collections for different content types."""
    
    def __init__(self):
        """Initialize the multi-collection RAG system."""
        # Set up ChromaDB client
        db_path = os.path.expanduser("~/.episodic/rag/chroma")
        os.makedirs(db_path, exist_ok=True)
        
        # Configure ChromaDB client
        from chromadb.config import Settings
        with suppress_chromadb_telemetry():
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        
        # Set up embedding function
        embedding_model = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
        with suppress_chromadb_telemetry():
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        
        # Initialize collections
        self.collections = {}
        self._init_collections()
        
        # Check if migration is needed
        if config.get("enable_collection_migration", True):
            self._check_and_migrate()
    
    def _init_collections(self):
        """Initialize all managed collections."""
        # User documents collection
        self.collections[CollectionType.USER_DOCS] = self._get_or_create_collection(
            'episodic_user_docs',
            'User indexed documents'
        )
        
        # Conversation memory collection  
        self.collections[CollectionType.CONVERSATION] = self._get_or_create_collection(
            'episodic_conversation_memory',
            'Conversation memories and exchanges'
        )
        
        debug_print("Initialized multi-collection RAG system", category="rag")
    
    def _get_or_create_collection(self, name: str, description: str):
        """Get or create a collection with the given name."""
        with suppress_chromadb_telemetry():
            try:
                collection = self.client.get_collection(
                    name=name,
                    embedding_function=self.embedding_function
                )
                debug_print(f"Loaded existing collection: {name}", category="rag")
            except:
                collection = self.client.create_collection(
                    name=name,
                    embedding_function=self.embedding_function,
                    metadata={"description": description}
                )
                debug_print(f"Created new collection: {name}", category="rag")
        
        return collection
    
    def _check_and_migrate(self):
        """Check if migration from old single collection is needed."""
        # Check if old collection exists
        try:
            old_collection = self.client.get_collection(
                name="episodic_docs",
                embedding_function=self.embedding_function
            )
            
            # Check if migration has been done
            migration_flag = config.get("collection_migration_completed", False)
            if not migration_flag:
                debug_print("Found old collection, migration may be needed", category="rag")
                # Migration will be implemented in a separate method
                # For now, just log that it's needed
        except:
            # Old collection doesn't exist, no migration needed
            debug_print("No old collection found, using new multi-collection system", category="rag")
    
    def get_collection(self, collection_type: str = CollectionType.USER_DOCS):
        """Get a collection by type."""
        if collection_type not in self.collections:
            raise ValueError(f"Unknown collection type: {collection_type}")
        return self.collections[collection_type]
    
    def add_document(self,
                    content: str,
                    source: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    collection_type: Optional[str] = None,
                    chunk: bool = True) -> Tuple[str, int]:
        """
        Add a document to the specified collection.
        
        Args:
            content: The document content
            source: Source identifier (e.g., 'file', 'conversation', 'web')
            metadata: Optional metadata for the document
            collection_type: Which collection to store in (defaults based on source)
            chunk: Whether to chunk the document
            
        Returns:
            Tuple of (document ID, number of chunks)
        """
        # Determine collection type from source if not specified
        if collection_type is None:
            if source == 'conversation':
                collection_type = CollectionType.CONVERSATION
            else:
                collection_type = CollectionType.USER_DOCS
        
        # Get the appropriate collection
        collection = self.get_collection(collection_type)
        
        # Generate document ID
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "source": source,
            "indexed_at": datetime.now().isoformat(),
            "collection_type": collection_type
        })
        
        # For now, store as single chunk (chunking logic can be added later)
        # This is a simplified implementation
        collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata]
        )
        
        debug_print(f"Added document to {collection_type}: {doc_id[:8]}", category="rag")
        return doc_id, 1
    
    def search(self,
              query: str,
              n_results: int = 5,
              collection_type: Optional[str] = None,
              source_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for relevant documents in specified collection(s).
        
        Args:
            query: The search query
            n_results: Number of results to return
            collection_type: Which collection to search (None = all)
            source_filter: Filter results by source
            
        Returns:
            Dictionary with search results
        """
        # If collection_type specified, search only that collection
        if collection_type:
            collections_to_search = [collection_type]
        else:
            # Search all collections
            collections_to_search = list(self.collections.keys())
        
        all_results = []
        
        for coll_type in collections_to_search:
            collection = self.get_collection(coll_type)
            
            # Build where clause
            where = {}
            if source_filter:
                where["source"] = source_filter
            
            # Perform search
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )
            
            # Format results
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else None
                    
                    # Add collection type to metadata
                    metadata['collection_type'] = coll_type
                    
                    all_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'relevance_score': 1.0 - (distance / 2.0) if distance else None
                    })
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Limit to n_results
        all_results = all_results[:n_results]
        
        return {
            'query': query,
            'results': all_results,
            'total': len(all_results)
        }
    
    def clear_collection(self, collection_type: str) -> int:
        """Clear all documents from a specific collection."""
        collection = self.get_collection(collection_type)
        
        # Get all document IDs
        all_ids = collection.get()['ids']
        
        if all_ids:
            # Delete all documents
            collection.delete(ids=all_ids)
            debug_print(f"Cleared {len(all_ids)} documents from {collection_type}", category="rag")
            return len(all_ids)
        
        return 0
    
    def get_collection_stats(self, collection_type: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for collection(s)."""
        stats = {}
        
        if collection_type:
            collections_to_check = [collection_type]
        else:
            collections_to_check = list(self.collections.keys())
        
        for coll_type in collections_to_check:
            collection = self.get_collection(coll_type)
            count = collection.count()
            
            stats[coll_type] = {
                'count': count,
                'name': collection.name,
                'metadata': collection.metadata
            }
        
        return stats


# Singleton instance
_multi_collection_rag = None


def get_multi_collection_rag() -> MultiCollectionRAG:
    """Get or create the multi-collection RAG instance."""
    global _multi_collection_rag
    if _multi_collection_rag is None:
        _multi_collection_rag = MultiCollectionRAG()
    return _multi_collection_rag