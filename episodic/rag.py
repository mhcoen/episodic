"""
Retrieval Augmented Generation functionality for Episodic.
"""

import os
import uuid
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
import logging
import sys
from contextlib import contextmanager
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
from episodic.db import get_connection

# Patch ChromaDB telemetry after import to fix the capture() argument error
try:
    import chromadb.telemetry.posthog
    # Replace the Posthog class with a no-op version
    class NoOpPosthog:
        def __init__(self, *args, **kwargs):
            pass
        def capture(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    chromadb.telemetry.posthog.Posthog = NoOpPosthog
    
    # Also patch the capture function directly if it exists
    if hasattr(chromadb.telemetry.posthog, 'capture'):
        chromadb.telemetry.posthog.capture = lambda *args, **kwargs: None
    
    # Disable telemetry product instance if it exists
    try:
        import chromadb.telemetry.product
        if hasattr(chromadb.telemetry.product, '_telemetry_client'):
            chromadb.telemetry.product._telemetry_client = None
    except:
        pass
    
    # Monkey patch print to suppress telemetry errors
    import builtins
    _original_print = builtins.print
    def _filtered_print(*args, **kwargs):
        # Skip telemetry error messages
        if args and len(args) > 0:
            first_arg = str(args[0])
            if "Failed to send telemetry" in first_arg or "capture() takes" in first_arg:
                return
        _original_print(*args, **kwargs)
    builtins.print = _filtered_print
        
except Exception:
    pass  # If the module structure changes, just ignore


# Note: suppress_telemetry_errors moved to rag_utils.py


class EpisodicRAG:
    """Manages RAG functionality for Episodic conversations."""
    
    def __init__(self):
        """Initialize the RAG system."""
        # Set up ChromaDB client
        db_path = os.path.expanduser("~/.episodic/rag/chroma")
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
            
            # Use sentence transformers for embeddings
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.get('rag_embedding_model', 'all-MiniLM-L6-v2')
            )
            
            # Get or create main collection
            self.collection = self.client.get_or_create_collection(
                name="episodic_knowledge",
                embedding_function=self.embedding_fn,
                metadata={"created_at": datetime.now().isoformat()}
            )
        
    def chunk_document(self, content: str, chunk_size: int = None, 
                      overlap: int = None, doc_type: str = None) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks for better retrieval."""
        chunk_size = chunk_size or config.get('rag_chunk_size', 500)
        overlap = overlap or config.get('rag_chunk_overlap', 100)
        use_format_preserving = config.get('rag_preserve_formatting', True)
        
        # Validate parameters
        from episodic.rag_utils import validate_chunk_params
        if not validate_chunk_params(chunk_size, overlap):
            raise ValueError("Invalid chunk parameters")
        
        # Use format-preserving chunker if enabled
        if use_format_preserving:
            from episodic.rag_chunker import FormatPreservingChunker
            chunker = FormatPreservingChunker(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            
            # Generate a temporary doc_id
            import uuid
            temp_doc_id = str(uuid.uuid4())[:8]
            
            # Detect document type if not specified
            if doc_type is None:
                if content.strip().startswith('#') or '```' in content:
                    doc_type = 'markdown'
                else:
                    doc_type = 'text'
            
            # Get format-preserving chunks
            doc_chunks = chunker.chunk_document(content, temp_doc_id, doc_type)
            
            # Convert to expected format
            chunks = []
            for chunk in doc_chunks:
                chunks.append({
                    'text': chunk.clean_text,  # For embeddings
                    'original_text': chunk.original_text,  # Preserved formatting
                    'start_idx': chunk.start_idx,
                    'end_idx': chunk.end_idx,
                    'chunk_type': chunk.chunk_type.value,
                    'metadata': chunk.metadata
                })
            
            return chunks
        
        # Fall back to original simple chunking
        words = content.split()
        chunks = []
        
        # If document is small enough, return as single chunk
        if len(words) <= chunk_size:
            return [{'text': content, 'start_idx': 0, 'end_idx': len(words)}]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_idx': i,
                'end_idx': min(i + chunk_size, len(words))
            })
        
        return chunks
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate hash for duplicate detection."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def check_duplicate(self, content_hash: str) -> Optional[str]:
        """Check if document already exists."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id FROM rag_documents WHERE content_hash = ?',
                (content_hash,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def add_document(self, 
                    content: str, 
                    source: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    chunk: bool = True) -> List[str]:
        """Add a document to the knowledge base with optional chunking."""
        # Check for duplicates
        content_hash = self.calculate_content_hash(content)
        existing_id = self.check_duplicate(content_hash)
        if existing_id:
            if config.get('debug'):
                typer.secho(f"Document already exists with ID: {existing_id}", fg="yellow")
            return [existing_id]
        
        # Generate parent document ID
        parent_doc_id = str(uuid.uuid4())
        doc_ids = []
        
        # Chunk document if requested and necessary
        if chunk:
            chunks = self.chunk_document(content)
        else:
            chunks = [{'text': content, 'start_idx': 0, 'end_idx': len(content.split())}]
        
        # Index each chunk
        for idx, chunk_data in enumerate(chunks):
            chunk_id = f"{parent_doc_id}-{idx}" if len(chunks) > 1 else parent_doc_id
            
            doc_metadata = {
                'source': source,
                'indexed_at': datetime.now().isoformat(),
                'word_count': len(chunk_data['text'].split()),
                'content': chunk_data['text'],  # Store content in metadata
                'parent_doc_id': parent_doc_id,
                'chunk_index': idx,
                'total_chunks': len(chunks),
            }
            
            # Add format-preserving metadata if available
            if 'original_text' in chunk_data:
                doc_metadata['original_text'] = chunk_data['original_text']
                doc_metadata['chunk_type'] = chunk_data.get('chunk_type', 'unknown')
                # Merge chunk metadata
                if 'metadata' in chunk_data:
                    doc_metadata.update(chunk_data['metadata'])
            
            # Only add content_hash for first chunk to avoid None values
            if idx == 0:
                doc_metadata['content_hash'] = content_hash
            
            if metadata:
                # Filter out None values from metadata to avoid ChromaDB issues
                filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                doc_metadata.update(filtered_metadata)
            
            self.collection.add(
                documents=[chunk_data['text']],
                metadatas=[doc_metadata],
                ids=[chunk_id]
            )
            
            doc_ids.append(chunk_id)
        
        # Track in SQLite (store full document once)
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO rag_documents (id, content, source, metadata, content_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (parent_doc_id, content, source, json.dumps({
                'indexed_at': datetime.now().isoformat(),
                'word_count': len(content.split()),
                'chunk_count': len(chunks)
            }), content_hash))
            conn.commit()
        
        return doc_ids
    
    def search(self, 
              query: str, 
              n_results: int = 5,
              threshold: float = 0.0) -> Dict[str, Any]:
        """Search for relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Filter by threshold if needed
        filtered_results = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }
        
        # If no results, return empty
        if not results['distances'] or not results['distances'][0]:
            return filtered_results
            
        for i, distance in enumerate(results['distances'][0]):
            # ChromaDB returns L2 distance for normalized embeddings
            # Lower is better, typical range is 0-2
            # For a basic threshold, we'll accept distances < 2.0
            if threshold == 0.0 or distance <= 2.0:
                # Check if we have original formatted text
                metadata = results['metadatas'][0][i]
                if 'original_text' in metadata:
                    # Use original formatted text for display
                    document_text = metadata['original_text']
                else:
                    # Fall back to clean text
                    document_text = results['documents'][0][i]
                
                filtered_results['documents'].append(document_text)
                filtered_results['metadatas'].append(metadata)
                filtered_results['distances'].append(distance)
                filtered_results['ids'].append(results['ids'][0][i])
        
        return filtered_results
    
    def list_documents(self, limit: Optional[int] = None, 
                      source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List documents in the knowledge base."""
        # Get all documents from the collection
        all_docs = self.collection.get()
        
        docs = []
        for i in range(len(all_docs['ids'])):
            metadata = all_docs['metadatas'][i]
            
            # Apply source filter if provided
            if source_filter and source_filter not in metadata.get('source', ''):
                continue
            
            docs.append({
                'id': all_docs['ids'][i],
                'source': metadata.get('source', 'Unknown'),
                'word_count': metadata.get('word_count', 0),
                'indexed_at': metadata.get('indexed_at', 'Unknown'),
                'content': metadata.get('content', '')[:200]  # Preview
            })
        
        # Sort by indexed_at (newest first)
        docs.sort(key=lambda x: x['indexed_at'], reverse=True)
        
        # Apply limit
        if limit:
            docs = docs[:limit]
        
        return docs
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            result = self.collection.get(ids=[doc_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['metadatas'][0].get('content', ''),
                    'metadata': result['metadatas'][0]
                }
        except Exception:
            pass
        return None
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        try:
            self.collection.delete(ids=[doc_id])
            
            # Also remove from SQLite
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM rag_documents WHERE id = ?', (doc_id,))
                conn.commit()
            
            return True
        except Exception:
            return False
    
    def clear_documents(self, source_filter: Optional[str] = None) -> int:
        """Clear documents from the knowledge base."""
        if source_filter:
            # Get documents matching the filter
            docs = self.list_documents(source_filter=source_filter)
            doc_ids = [doc['id'] for doc in docs]
            
            if doc_ids:
                self.collection.delete(ids=doc_ids)
                
                # Remove from SQLite - use separate queries to avoid dynamic SQL
                with get_connection() as conn:
                    cursor = conn.cursor()
                    for doc_id in doc_ids:
                        cursor.execute('DELETE FROM rag_documents WHERE id = ?', (doc_id,))
                    conn.commit()
            
            return len(doc_ids)
        else:
            # Clear all documents
            # Get count before deletion
            count = self.collection.count()
            
            # Delete the collection and recreate it
            self.client.delete_collection(name="episodic_knowledge")
            self.collection = self.client.get_or_create_collection(
                name="episodic_knowledge",
                embedding_function=self.embedding_fn,
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            # Clear SQLite table
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM rag_documents')
                conn.commit()
            
            return count
    
    def get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of documents by source."""
        docs = self.list_documents()
        distribution = {}
        
        for doc in docs:
            source = doc['source']
            distribution[source] = distribution.get(source, 0) + 1
        
        return distribution
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        count = self.collection.count()
        
        return {
            'total_documents': count,
            'embedding_model': config.get('rag_embedding_model', 'all-MiniLM-L6-v2'),
            'collection_name': 'episodic_knowledge'
        }
    
    def enhance_with_context(self, message: str, n_results: int = None,
                           threshold: float = None, include_web: bool = None) -> Tuple[str, List[str]]:
        """Enhance a message with relevant context from the knowledge base.
        
        Args:
            message: The message to enhance
            n_results: Number of results to include
            threshold: Relevance threshold for local results
            include_web: Whether to include web search results
        
        Returns:
            Tuple of (enhanced_message, list_of_sources_used)
        """
        if not config.get('rag_auto_search', True):
            return message, []
        
        n_results = n_results or config.get('rag_max_results', 5)
        threshold = threshold or config.get('rag_search_threshold', 0.7)
        
        # Search for relevant context
        results = self.search(message, n_results=n_results, threshold=threshold)
        
        # Check if we should also search the web
        should_search_web = self._should_search_web(results, include_web)
        
        context_parts = []
        sources_used = []
        
        # Add local results if any
        if results['documents']:
            context_prefix = config.get('rag_context_prefix', 
                                      'CONTEXT FROM DOCUMENTATION:\n')
            context_parts.append(context_prefix)
            
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                source = metadata.get('source', 'Unknown')
                if source not in sources_used:
                    sources_used.append(source)
                
                # Add numbered context
                context_parts.append(f"\n{i+1}. From {source}:")
                context_parts.append(doc)
        
        # Add web results if needed
        if should_search_web and config.get('web_search_enabled', False):
            from episodic.web_search import get_web_search_manager
            
            web_manager = get_web_search_manager()
            web_results = web_manager.search(message, num_results=3)  # Fewer web results
            
            if web_results:
                if context_parts:
                    context_parts.append("\n")
                context_parts.append("\n[Current web information]:\n")
                
                for i, result in enumerate(web_results):
                    source = f"web:{result.title}"
                    sources_used.append(source)
                    
                    context_parts.append(f"\n{len(results['documents']) + i + 1}. {result.title}")
                    context_parts.append(f"   {result.snippet}")
                    
                    # Optionally index for future use
                    if config.get('web_search_index_results', True):
                        try:
                            content = f"{result.title}\n\n{result.snippet}\n\nSource: {result.url}"
                            self.add_document(
                                content=content,
                                source=f"web:{result.url}",
                                metadata={
                                    'title': result.title,
                                    'url': result.url,
                                    'search_query': message,
                                    'search_timestamp': result.timestamp.isoformat()
                                }
                            )
                        except Exception:
                            pass  # Ignore indexing errors
        
        if not context_parts:
            return message, []
        
        # Add citation note if enabled
        if config.get('rag_include_citations', True) and sources_used:
            context_parts.append(f"\n\n[Sources: {', '.join(sources_used)}]")
        
        # Combine context FIRST, then original message
        # This ensures LLM sees relevant context before the question
        enhanced_message = ''.join(context_parts) + '\n\n' + message
        
        # Track retrieval in database if not in help mode
        # Skip tracking for help queries to avoid polluting retrieval history
        if not hasattr(self, '_is_help_rag'):
            try:
                with get_connection() as conn:
                    cursor = conn.cursor()
                    # Use the actual schema which has node_id, not query
                    # For now, just track document_id without node_id since this is a standalone query
                    for i, doc_id in enumerate(results['ids']):
                        relevance_score = 1.0 - results['distances'][i] if results['distances'] else 0.5
                        cursor.execute('''
                            INSERT INTO rag_retrievals (document_id, relevance_score)
                            VALUES (?, ?)
                        ''', (doc_id, relevance_score))
                    conn.commit()
            except Exception as e:
                # Log error but don't fail the whole operation
                if config.get('debug'):
                    typer.secho(f"Warning: Failed to track retrieval: {e}", fg="yellow")
        
        return enhanced_message, sources_used
    
    def _should_search_web(self, local_results: Dict, include_web: Optional[bool]) -> bool:
        """Determine if web search should be performed."""
        # Explicit user preference
        if include_web is not None:
            return include_web
        
        # Check auto-enhance setting
        if not config.get('web_search_auto_enhance', False):
            return False
        
        # No local results or all have low relevance
        if not local_results['documents']:
            return True
        
        # Check if local results are good enough
        if local_results['distances']:
            # ChromaDB distance: lower is better, 0 is perfect match
            best_distance = min(local_results['distances'][0])
            relevance_threshold = 1 - config.get('rag_search_threshold', 0.7)
            
            # If best result is too far (low relevance), search web
            if best_distance > relevance_threshold:
                return True
        
        return False


# Global RAG instance
rag_system = None


def get_rag_system() -> Optional[EpisodicRAG]:
    """Get or create the global RAG system instance with error handling."""
    global rag_system
    
    if rag_system is None:
        try:
            rag_system = EpisodicRAG()
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"⚠️  Failed to initialize RAG system: {e}", fg="yellow")
            # Return None to allow graceful degradation
            return None
    
    return rag_system


def ensure_rag_initialized() -> bool:
    """Ensure RAG system is initialized, with user-friendly error messages."""
    try:
        rag = get_rag_system()
        if rag is None:
            typer.secho("⚠️  RAG system initialization failed.", fg="yellow")
            typer.secho("Try installing required dependencies:", fg="yellow")
            typer.secho("  pip install chromadb sentence-transformers", fg="cyan")
            return False
        return True
    except ImportError as e:
        typer.secho(f"⚠️  Missing dependency: {e}", fg="yellow")
        typer.secho("Install with: pip install chromadb sentence-transformers", fg="cyan")
        return False
    except Exception as e:
        typer.secho(f"⚠️  RAG initialization error: {e}", fg="red")
        return False