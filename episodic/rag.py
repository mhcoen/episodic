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

# Monkey-patch to completely disable ChromaDB telemetry due to compatibility issues
import types

class MockTelemetry:
    """Mock telemetry module to prevent errors."""
    def __getattr__(self, name):
        # Return a no-op function for any attribute access
        return lambda *args, **kwargs: None

# Create mock modules for telemetry
sys.modules['chromadb.telemetry'] = types.ModuleType('telemetry')
sys.modules['chromadb.telemetry.posthog'] = MockTelemetry()
sys.modules['chromadb.telemetry.events'] = MockTelemetry()

# Suppress ChromaDB warnings
logging.getLogger('chromadb').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*telemetry.*")

import chromadb
from chromadb.utils import embedding_functions
import typer

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color
from episodic.db import get_connection


@contextmanager
def suppress_telemetry_errors():
    """Context manager to suppress telemetry error messages."""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


class EpisodicRAG:
    """Manages RAG functionality for Episodic conversations."""
    
    def __init__(self):
        """Initialize the RAG system."""
        # Set up ChromaDB client
        db_path = os.path.expanduser("~/.episodic/rag/chroma")
        os.makedirs(db_path, exist_ok=True)
        
        # Configure ChromaDB client with telemetry disabled
        from chromadb.config import Settings
        
        # Suppress telemetry errors during initialization
        with suppress_telemetry_errors():
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
                      overlap: int = None) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks for better retrieval."""
        chunk_size = chunk_size or config.get('rag_chunk_size', 500)
        overlap = overlap or config.get('rag_chunk_overlap', 100)
        
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
        conn = get_connection()
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
                'content_hash': content_hash if idx == 0 else None
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            self.collection.add(
                documents=[chunk_data['text']],
                metadatas=[doc_metadata],
                ids=[chunk_id]
            )
            
            doc_ids.append(chunk_id)
        
        # Track in SQLite (store full document once)
        conn = get_connection()
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
        
        for i, distance in enumerate(results['distances'][0]):
            # ChromaDB uses cosine distance, lower is better
            if distance <= (1 - threshold):
                filtered_results['documents'].append(results['documents'][0][i])
                filtered_results['metadatas'].append(results['metadatas'][0][i])
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
            conn = get_connection()
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
                
                # Remove from SQLite
                conn = get_connection()
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(doc_ids))
                cursor.execute(f'DELETE FROM rag_documents WHERE id IN ({placeholders})', doc_ids)
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
            conn = get_connection()
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
                           threshold: float = None) -> Tuple[str, List[str]]:
        """Enhance a message with relevant context from the knowledge base.
        
        Returns:
            Tuple of (enhanced_message, list_of_sources_used)
        """
        if not config.get('rag_auto_search', True):
            return message, []
        
        n_results = n_results or config.get('rag_max_results', 5)
        threshold = threshold or config.get('rag_search_threshold', 0.7)
        
        # Search for relevant context
        results = self.search(message, n_results=n_results, threshold=threshold)
        
        if not results['documents']:
            return message, []
        
        # Build context section
        context_prefix = config.get('rag_context_prefix', 
                                  '\n\n[Relevant context from knowledge base]:\n')
        context_parts = [context_prefix]
        sources_used = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            source = metadata.get('source', 'Unknown')
            if source not in sources_used:
                sources_used.append(source)
            
            # Add numbered context
            context_parts.append(f"\n{i+1}. From {source}:")
            context_parts.append(doc)
        
        # Add citation note if enabled
        if config.get('rag_include_citations', True):
            context_parts.append(f"\n\n[Sources: {', '.join(sources_used)}]")
        
        # Combine original message with context
        enhanced_message = message + ''.join(context_parts)
        
        # Track retrieval in database
        conn = get_connection()
        cursor = conn.cursor()
        for doc_id in results['ids']:
            cursor.execute('''
                INSERT INTO rag_retrievals (document_id, query, retrieved_at)
                VALUES (?, ?, ?)
            ''', (doc_id, message, datetime.now().isoformat()))
        conn.commit()
        
        return enhanced_message, sources_used


# Global RAG instance
rag_system = None


def get_rag_system() -> Optional[EpisodicRAG]:
    """Get or create the global RAG system instance with error handling."""
    global rag_system
    
    if rag_system is None:
        try:
            # Apply runtime telemetry patch if needed
            try:
                import chromadb.telemetry.posthog as posthog_module
                if hasattr(posthog_module, 'Posthog'):
                    # Patch the capture method to accept any arguments
                    original_capture = getattr(posthog_module.Posthog, 'capture', None)
                    if original_capture:
                        posthog_module.Posthog.capture = lambda self, *args, **kwargs: None
            except Exception:
                pass  # Ignore patching errors
            
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