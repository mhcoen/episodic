"""
Retrieval Augmented Generation functionality for Episodic.
"""

import os
import uuid
import warnings
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from io import StringIO

# Disable ChromaDB telemetry to avoid warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress ChromaDB warnings
logging.getLogger('chromadb').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*Failed to send telemetry.*")

# Import chromadb with stderr redirected to suppress telemetry
_stderr_backup = sys.stderr
try:
    sys.stderr = StringIO()
    import chromadb
    from chromadb.utils import embedding_functions
finally:
    sys.stderr = _stderr_backup

import typer

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color
# Import from modular files
from episodic.rag_document_manager import (
    calculate_content_hash,
    check_duplicate,
    add_document_to_db,
    list_documents as _list_documents,
    get_document as _get_document,
    remove_document as _remove_document,
    clear_documents as _clear_documents,
    get_source_distribution,
    get_document_stats,
    record_retrieval
)

# Patch ChromaDB telemetry after import to fix the capture() argument error
try:
    import chromadb.telemetry.posthog
    
    # Replace the Posthog class with a no-op version that handles all signatures
    class NoOpPosthog:
        def __init__(self, *args, **kwargs):
            pass
        
        def capture(self, *args, **kwargs):
            # Accept any number of arguments to handle signature changes
            pass
        
        def __getattr__(self, name):
            # Return a function that accepts any arguments
            return lambda *args, **kwargs: None
    
    # Replace the module's Posthog class
    chromadb.telemetry.posthog.Posthog = NoOpPosthog
    
    # Patch any existing instances
    if hasattr(chromadb.telemetry, 'posthog'):
        if hasattr(chromadb.telemetry.posthog, '_posthog'):
            chromadb.telemetry.posthog._posthog = NoOpPosthog()
    
    # Also patch the capture function directly if it exists
    if hasattr(chromadb.telemetry.posthog, 'capture'):
        chromadb.telemetry.posthog.capture = lambda *args, **kwargs: None
    
    # Disable telemetry product instance if it exists
    try:
        import chromadb.telemetry.product
        if hasattr(chromadb.telemetry.product, '_telemetry_client'):
            chromadb.telemetry.product._telemetry_client = None
        if hasattr(chromadb.telemetry.product, 'TelemetryClient'):
            chromadb.telemetry.product.TelemetryClient = NoOpPosthog
    except:
        pass
    
    # Patch the telemetry event classes to prevent errors
    try:
        import chromadb.telemetry.events
        for attr_name in dir(chromadb.telemetry.events):
            if attr_name.endswith('Event'):
                setattr(chromadb.telemetry.events, attr_name, type(attr_name, (), {
                    '__init__': lambda self, **kwargs: None,
                    '__dict__': {}
                }))
    except:
        pass
    
    # Monkey patch print to suppress telemetry errors
    import builtins
    _original_print = builtins.print
    def _filtered_print(*args, **kwargs):
        # Skip telemetry error messages
        if args and len(args) > 0:
            first_arg = str(args[0])
            if any(pattern in first_arg for pattern in [
                "Failed to send telemetry",
                "capture() takes",
                "ClientStartEvent",
                "CollectionGetEvent", 
                "CollectionQueryEvent"
            ]):
                return
        _original_print(*args, **kwargs)
    builtins.print = _filtered_print
        
except Exception:
    pass  # If the module structure changes, just ignore


class EpisodicRAG:
    """Manages RAG functionality for Episodic conversations."""
    
    def __init__(self):
        """Initialize the RAG system."""
        # Set up ChromaDB client
        db_path = os.path.expanduser("~/.episodic/rag/chroma")
        
        # Validate the path to ensure it's not in the project directory
        from .db_safeguards import validate_db_path
        db_path = validate_db_path(db_path)
        
        os.makedirs(db_path, exist_ok=True)
        
        # Configure ChromaDB client with telemetry disabled
        from chromadb.config import Settings
        from episodic.rag_utils import suppress_chromadb_telemetry
        
        # Suppress telemetry errors during initialization
        with suppress_chromadb_telemetry():
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        
        # Get or create collection with embedding function
        embedding_model = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
        
        with suppress_chromadb_telemetry():
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        
            # Get or create the main collection
            try:
                self.collection = self.client.get_collection(
                    name="episodic_docs",
                    embedding_function=self.embedding_function
                )
            except:
                self.collection = self.client.create_collection(
                    name="episodic_docs",
                    embedding_function=self.embedding_function,
                    metadata={"description": "Episodic conversation knowledge base"}
                )
        
        # Ensure SQL database tables exist for RAG functionality
        from .db_rag import create_rag_tables
        create_rag_tables()
    
    def chunk_document(self, content: str, chunk_size: int = None, 
                      overlap: int = None) -> List[Tuple[str, Dict[str, int]]]:
        """
        Split a document into overlapping chunks for better retrieval.
        
        Args:
            content: The document content to chunk
            chunk_size: Size of each chunk in characters (default from config)
            overlap: Overlap between chunks in characters (default from config)
            
        Returns:
            List of tuples (chunk_text, metadata) where metadata contains start/end positions
        """
        if chunk_size is None:
            chunk_size = config.get("rag_chunk_size", 1000)
        if overlap is None:
            overlap = config.get("rag_chunk_overlap", 200)
        
        # Simple character-based chunking
        chunks = []
        
        # If content is smaller than chunk size, return as single chunk
        if len(content) <= chunk_size:
            chunks.append((content, {"start": 0, "end": len(content)}))
            return chunks
        
        # Create overlapping chunks
        start = 0
        while start < len(content):
            # Calculate end position
            end = start + chunk_size
            
            # Adjust end to not break in the middle of a word if possible
            if end < len(content):
                # Look for the last space before the end
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Extract chunk
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, {"start": start, "end": end}))
            
            # Move start position with overlap
            start = end - overlap
            if start <= chunks[-1][1]["start"]:
                # Avoid infinite loop
                start = chunks[-1][1]["end"]
                
        return chunks
    
    def add_document(self, 
                    content: str, 
                    source: str, 
                    metadata: Optional[Dict[str, Any]] = None,
                    chunk: bool = True) -> Tuple[str, int]:
        """
        Add a document to the RAG index.
        
        Args:
            content: The document content
            source: Source identifier (e.g., 'file', 'text', 'web')
            metadata: Optional metadata for the document
            chunk: Whether to chunk the document (default: True)
            
        Returns:
            Tuple of (document ID, number of chunks)
        """
        # Calculate content hash for duplicate detection
        content_hash = calculate_content_hash(content)
        
        # Check for duplicates
        existing_doc_id = check_duplicate(content_hash)
        if existing_doc_id:
            typer.secho(f"Document already indexed with ID: {existing_doc_id}", 
                       fg=get_text_color())
            # Get the existing document's chunk count
            doc = _get_document(existing_doc_id)
            return existing_doc_id, doc['chunk_count'] if doc else 0
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "source": source,
            "indexed_at": datetime.now().isoformat()
        })
        
        # Generate preview text (first 200 chars, clean it up)
        preview = self._generate_preview(content)
        
        # Process the document
        from episodic.rag_utils import suppress_chromadb_telemetry
        
        if chunk:
            # Chunk the document
            chunks = self.chunk_document(content)
            
            # Add chunks to vector store
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, (chunk_text, chunk_meta) in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)
                
                # Combine document and chunk metadata
                combined_meta = metadata.copy()
                combined_meta.update({
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_start": chunk_meta["start"],
                    "chunk_end": chunk_meta["end"]
                })
                chunk_metadatas.append(combined_meta)
            
            # Add to ChromaDB
            with suppress_chromadb_telemetry():
                self.collection.add(
                    ids=chunk_ids,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas
                )
            
            # Store document metadata in SQLite with preview
            add_document_to_db(doc_id, source, metadata, content_hash, len(chunks), preview)
            
            return doc_id, len(chunks)
        else:
            # Add entire document as single chunk
            with suppress_chromadb_telemetry():
                self.collection.add(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[metadata]
                )
            
            # Store document metadata with preview
            add_document_to_db(doc_id, source, metadata, content_hash, 1, preview)
            
            return doc_id, 1
    
    def search(self, 
              query: str, 
              n_results: int = None,
              source_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for relevant documents.
        
        Args:
            query: The search query
            n_results: Number of results to return (default from config)
            source_filter: Filter results by source
            
        Returns:
            Dictionary with search results and metadata
        """
        if n_results is None:
            n_results = config.get("rag_search_results", 5)
        
        # Build where clause for filtering
        where = None
        if source_filter:
            where = {"source": source_filter}
        
        # Perform search
        from episodic.rag_utils import suppress_chromadb_telemetry
        with suppress_chromadb_telemetry():
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
        
        # Format results
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else None
                
                documents.append({
                    'content': doc,
                    'metadata': metadata,
                    'relevance_score': 1.0 - (distance / 2.0) if distance else None
                })
        
        return {
            'query': query,
            'results': documents,
            'total': len(documents)
        }
    
    def list_documents(self, limit: Optional[int] = None, 
                      source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        return _list_documents(limit, source_filter)
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        return _get_document(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the index."""
        # Remove from vector store
        from episodic.rag_utils import suppress_chromadb_telemetry
        with suppress_chromadb_telemetry():
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results['ids']:
                # Delete all chunks
                self.collection.delete(ids=results['ids'])
        
        # Remove from database
        return _remove_document(doc_id)
    
    def clear_documents(self, source_filter: Optional[str] = None) -> int:
        """Clear all documents or documents from a specific source."""
        count, doc_ids = _clear_documents(source_filter)
        
        # Remove from vector store
        from episodic.rag_utils import suppress_chromadb_telemetry
        with suppress_chromadb_telemetry():
            for doc_id in doc_ids:
                # Get all chunk IDs for this document
                results = self.collection.get(
                    where={"doc_id": doc_id}
                )
                
                if results['ids']:
                    # Delete all chunks
                    self.collection.delete(ids=results['ids'])
        
        return count
    
    def get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of documents by source."""
        return get_source_distribution()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        stats = get_document_stats()
        
        # Add collection stats
        from episodic.rag_utils import suppress_chromadb_telemetry
        with suppress_chromadb_telemetry():
            stats['collection_count'] = self.collection.count()
        
        # Add embedding model info
        stats['embedding_model'] = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
        
        return stats
    
    def enhance_with_context(self, message: str, n_results: int = None,
                           include_web: Optional[bool] = None) -> str:
        """
        Enhance a message with relevant context from the knowledge base.
        
        Args:
            message: The user's message
            n_results: Number of results to include (default from config)
            include_web: Whether to include web search results if local results insufficient
            
        Returns:
            Enhanced message with context prepended
        """
        if n_results is None:
            n_results = config.get("rag_context_results", 3)
            
        # Search for relevant context
        results = self.search(message, n_results=n_results)
        
        # Check if we should do web search
        if self._should_search_web(results, include_web):
            from episodic.web_search import WebSearchManager
            searcher = WebSearchManager()
            
            # Perform web search
            web_results = searcher.search(message, num_results=3)
            
            if web_results:
                typer.echo("\n🌐 Augmenting with web search results...", 
                          fg=get_system_color())
                
                # Add web results to context
                for i, result in enumerate(web_results[:2]):
                    # Create a synthetic document entry
                    results['results'].append({
                        'content': result.snippet,
                        'metadata': {
                            'source': 'web',
                            'title': result.title,
                            'url': result.url
                        },
                        'relevance_score': 0.8 - (i * 0.1)  # Slightly lower than local
                    })
        
        if not results['results']:
            # No relevant context found
            return message
        
        # Build context section
        context_parts = ["### Relevant Context ###"]
        
        # Track which documents were used
        used_doc_ids = []
        chunk_texts = []
        
        for i, result in enumerate(results['results']):
            metadata = result.get('metadata', {})
            content = result['content']
            
            # Add source attribution
            source = metadata.get('source', 'unknown')
            if source == 'file':
                source_info = f"From file: {metadata.get('filename', 'unknown')}"
            elif source == 'web':
                source_info = f"From web: {metadata.get('title', 'Web Page')}"
                if metadata.get('url'):
                    source_info += f" ({metadata['url']})"
            else:
                source_info = f"From {source}"
            
            context_parts.append(f"\n[Context {i+1} - {source_info}]")
            context_parts.append(content)
            
            # Track usage
            if doc_id := metadata.get('doc_id'):
                used_doc_ids.append(doc_id)
            chunk_texts.append(content)
        
        context_parts.append("\n### User Query ###")
        context_parts.append(message)
        
        # Record retrieval for analytics (skip for help RAG)
        if used_doc_ids and not getattr(self, '_is_help_rag', False):
            record_retrieval(message, used_doc_ids, chunk_texts)
        
        return "\n".join(context_parts)
    
    def _should_search_web(self, local_results: Dict, include_web: Optional[bool]) -> bool:
        """Determine if web search should be performed."""
        # Explicit control
        if include_web is not None:
            return include_web
            
        # Check if web search is enabled
        if not config.get("web_search_enabled", False):
            return False
            
        # Auto-detect: search web if local results are insufficient
        if not local_results['results']:
            return True
            
        # Check relevance scores
        avg_score = sum(r.get('relevance_score', 0) for r in local_results['results']) / len(local_results['results'])
        
        # If average relevance is low, search web
        return avg_score < config.get("rag_web_search_threshold", 0.7)
    
    def _generate_preview(self, content: str, max_length: int = 200) -> str:
        """Generate a preview of the content.
        
        Args:
            content: The full content
            max_length: Maximum length of preview (default: 200)
            
        Returns:
            Preview text, cleaned and truncated
        """
        # Remove excessive whitespace
        preview = ' '.join(content.split())
        
        # Truncate to max length
        if len(preview) > max_length:
            # Try to cut at a word boundary
            preview = preview[:max_length]
            last_space = preview.rfind(' ')
            if last_space > max_length * 0.8:  # Only cut at word if we keep 80% of content
                preview = preview[:last_space]
            preview += '...'
        
        return preview


# Global instance
_rag_system: Optional[EpisodicRAG] = None


def get_rag_system() -> Optional[EpisodicRAG]:
    """Get the global RAG system instance."""
    global _rag_system
    
    if _rag_system is None:
        try:
            # Check if we should use the multi-collection system
            if config.get("collection_migration_completed", False):
                # Use the adapter for multi-collection system
                from episodic.rag_adapter import EpisodicRAGAdapter
                _rag_system = EpisodicRAGAdapter()
            else:
                # Use the original single-collection system
                _rag_system = EpisodicRAG()
        except Exception as e:
            if config.get("debug"):
                typer.secho(f"Failed to initialize RAG system: {e}", fg="red")
            return None
    
    return _rag_system


def ensure_rag_initialized() -> bool:
    """Ensure RAG system is initialized."""
    return get_rag_system() is not None