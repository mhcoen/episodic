# RAG Implementation Plan for Episodic

## Overview

This plan outlines the implementation of RAG (Retrieval Augmented Generation) capabilities in Episodic using ChromaDB as the vector store and a phased approach that starts simple and grows more sophisticated.

## Command Summary

### Core RAG Commands

- **`/rag [on|off]`** - Enable/disable RAG and show statistics
- **`/search <query>`** - Search the knowledge base for relevant documents
- **`/index <file>`** - Add a file to the knowledge base (supports .txt, .md, .pdf, .rst)
- **`/index --text "<content>"`** - Add text content directly to the knowledge base

### Document Management Commands

- **`/docs`** - List all documents in the knowledge base (default action)
- **`/docs list [--limit N] [--source filter]`** - List documents with optional filters
- **`/docs show <doc_id>`** - Display full content of a specific document
- **`/docs remove <doc_id>`** - Remove a specific document (with confirmation)
- **`/docs clear [--source filter]`** - Clear all or filtered documents (with confirmation)

### Batch Operations

- **`/index-dir <directory> [--pattern '*.txt'] [--recursive]`** - Index all matching files in a directory
- **`/export-kb <output_file>`** - Export knowledge base to JSON file
- **`/import-kb <input_file>`** - Import knowledge base from JSON file

### Testing and Validation

- **`/test-rag <query> [--expect <source>]`** - Test RAG retrieval quality
- **`/benchmark-rag`** - Benchmark RAG performance (search and indexing speed)

### Convenience Shortcuts

- **`/s <query>`** - Shortcut for `/search`
- **`/i <file>`** - Shortcut for `/index`
- **`/d`** - Shortcut for `/docs`

### Key Features

1. **Document Chunking**: Large documents automatically split into overlapping chunks for better retrieval
2. **Duplicate Detection**: Prevents indexing the same content multiple times
3. **Token Budget Management**: Ensures RAG context fits within model limits
4. **Full CRUD Operations**: Create (index), Read (list/show), Update (via remove+add), Delete (remove/clear)
5. **Source Filtering**: Filter documents by source when listing or clearing
6. **Safety Confirmations**: Deletion operations require confirmation to prevent accidents
7. **Progress Indicators**: Visual feedback for batch operations
8. **Error Recovery**: Graceful handling of missing dependencies and initialization failures

### Relationship with Existing Commands

- **`/load`** remains for loading PDFs into current conversation context (session-based)
- **`/index`** adds documents to persistent RAG knowledge base (permanent)
- **`/search`** can be used independently or auto-triggered during conversations
- **`/docs`** manages the knowledge base similar to how `/topics` manages conversation topics

## Phase 1: Basic RAG Infrastructure (Week 1)

### 1.1 Core RAG Module
Create `episodic/rag.py`:
```python
"""
Retrieval Augmented Generation functionality for Episodic.
"""

import os
import uuid
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
import typer

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color
from episodic.db import get_connection


class EpisodicRAG:
    """Manages RAG functionality for Episodic conversations."""
    
    def __init__(self):
        """Initialize the RAG system."""
        # Set up ChromaDB client
        db_path = os.path.expanduser("~/.episodic/rag/chroma")
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        
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
```

### 1.2 Database Schema Updates
Add to `episodic/db.py`:
```python
def create_rag_tables():
    """Create tables for RAG functionality."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for tracking indexed documents
    cursor.execute('''
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
    cursor.execute('''
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
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_documents_hash ON rag_documents(content_hash)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_documents_source ON rag_documents(source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_retrievals_doc ON rag_retrievals(document_id)')
    
    conn.commit()
```

### 1.3 Configuration Updates
Add to `episodic/config_defaults.py`:
```python
# RAG (Retrieval Augmented Generation) settings
RAG_DEFAULTS = {
    'rag_enabled': {
        'default': False,
        'doc': 'Enable RAG for enhanced responses with external knowledge'
    },
    'rag_auto_search': {
        'default': True,
        'doc': 'Automatically search knowledge base for each user message'
    },
    'rag_search_threshold': {
        'default': 0.7,
        'doc': 'Minimum relevance score for including search results'
    },
    'rag_max_results': {
        'default': 5,
        'doc': 'Maximum number of search results to include'
    },
    'rag_embedding_model': {
        'default': 'all-MiniLM-L6-v2',
        'doc': 'Sentence transformer model for embeddings'
    },
    'rag_include_citations': {
        'default': True,
        'doc': 'Include source citations in responses'
    },
    'rag_context_prefix': {
        'default': '\n\n[Relevant context from knowledge base]:\n',
        'doc': 'Prefix for RAG context in prompts'
    },
    'rag_chunk_size': {
        'default': 500,
        'doc': 'Number of words per document chunk'
    },
    'rag_chunk_overlap': {
        'default': 100,
        'doc': 'Number of overlapping words between chunks'
    },
    'rag_max_file_size': {
        'default': 10 * 1024 * 1024,  # 10MB
        'doc': 'Maximum file size for indexing (in bytes)'
    },
    'rag_show_citations': {
        'default': True,
        'doc': 'Show which documents were used in responses'
    },
    'rag_citation_style': {
        'default': 'inline',  # or 'footnote'
        'doc': 'How to display citations in responses'
    },
    'rag_allowed_file_types': {
        'default': ['.txt', '.md', '.pdf', '.rst'],
        'doc': 'Allowed file extensions for indexing'
    }
}
```

## Phase 2: Command Integration (Week 1-2)

### 2.1 RAG Commands Module
Create `episodic/commands/rag.py`:
```python
"""RAG-related commands for Episodic."""

import typer
from typing import Optional, List
from datetime import datetime

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color
from episodic.rag import get_rag_system


def search(query: str, limit: Optional[int] = None):
    """Search the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    n_results = limit or config.get('rag_max_results', 5)
    threshold = config.get('rag_search_threshold', 0.7)
    
    results = rag.search(query, n_results=n_results, threshold=threshold)
    
    if not results['documents']:
        typer.secho("No relevant results found.", fg=get_text_color())
        return
    
    typer.secho(f"\n🔍 Search Results for: '{query}'", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    for i, (doc, metadata, distance, doc_id) in enumerate(zip(
        results['documents'], 
        results['metadatas'], 
        results['distances'],
        results['ids']
    )):
        relevance = 1 - distance  # Convert distance to similarity
        typer.secho(f"\n[{i+1}] ", nl=False, fg=get_system_color(), bold=True)
        typer.secho(f"Relevance: {relevance:.2%}", fg=get_system_color())
        typer.secho(f"Source: {metadata.get('source', 'Unknown')}", fg=get_text_color())
        typer.secho(f"ID: {doc_id[:8]}...", fg=get_text_color())
        
        # Show snippet
        snippet = doc[:200] + "..." if len(doc) > 200 else doc
        typer.secho(f"{snippet}", fg=get_text_color())


def list_documents(limit: Optional[int] = None, source_filter: Optional[str] = None):
    """List all documents in the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    docs = rag.list_documents(limit=limit, source_filter=source_filter)
    
    if not docs:
        typer.secho("No documents in knowledge base.", fg=get_text_color())
        return
    
    typer.secho(f"\n📚 Documents in Knowledge Base", fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    for i, doc in enumerate(docs):
        typer.secho(f"\n[{i+1}] ", nl=False, fg=get_system_color(), bold=True)
        typer.secho(f"{doc['source']}", fg=get_system_color())
        typer.secho(f"   ID: {doc['id'][:8]}...", fg=get_text_color())
        typer.secho(f"   Words: {doc['word_count']}", fg=get_text_color())
        typer.secho(f"   Indexed: {doc['indexed_at']}", fg=get_text_color())
        
        # Show preview
        preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
        typer.secho(f"   Preview: {preview}", fg=get_text_color())


def show_document(doc_id: str):
    """Show full content of a specific document."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    doc = rag.get_document(doc_id)
    
    if not doc:
        typer.secho(f"Document not found: {doc_id}", fg="red")
        return
    
    typer.secho(f"\n📄 Document Details", fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    typer.secho(f"ID: {doc['id']}", fg=get_system_color())
    typer.secho(f"Source: {doc['metadata']['source']}", fg=get_system_color())
    typer.secho(f"Words: {doc['metadata']['word_count']}", fg=get_text_color())
    typer.secho(f"Indexed: {doc['metadata']['indexed_at']}", fg=get_text_color())
    
    typer.secho(f"\nContent:", fg=get_heading_color())
    typer.secho("─" * 60, fg=get_heading_color())
    typer.secho(doc['content'], fg=get_text_color())


def remove_document(doc_id: str, confirm: bool = True):
    """Remove a document from the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    
    # Get document details before deletion
    doc = rag.get_document(doc_id)
    if not doc:
        typer.secho(f"Document not found: {doc_id}", fg="red")
        return
    
    # Confirm deletion
    if confirm:
        typer.secho(f"\nAbout to delete:", fg="yellow")
        typer.secho(f"  Source: {doc['metadata']['source']}", fg=get_text_color())
        typer.secho(f"  Words: {doc['metadata']['word_count']}", fg=get_text_color())
        
        if not typer.confirm("Are you sure you want to delete this document?"):
            typer.secho("Deletion cancelled.", fg=get_text_color())
            return
    
    # Delete the document
    if rag.remove_document(doc_id):
        typer.secho(f"✅ Document deleted successfully", fg=get_system_color())
        typer.secho(f"   Source: {doc['metadata']['source']}", fg=get_text_color())
    else:
        typer.secho(f"❌ Failed to delete document", fg="red")


def clear_documents(source_filter: Optional[str] = None, confirm: bool = True):
    """Clear all or filtered documents from the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    
    # Get count of documents to be deleted
    docs = rag.list_documents(source_filter=source_filter)
    count = len(docs)
    
    if count == 0:
        typer.secho("No documents to delete.", fg=get_text_color())
        return
    
    # Confirm deletion
    if confirm:
        if source_filter:
            typer.secho(f"\n⚠️  About to delete {count} documents with source matching '{source_filter}'", fg="yellow")
        else:
            typer.secho(f"\n⚠️  About to delete ALL {count} documents from the knowledge base", fg="yellow")
        
        if not typer.confirm("Are you sure you want to proceed?"):
            typer.secho("Deletion cancelled.", fg=get_text_color())
            return
    
    # Delete documents
    deleted = rag.clear_documents(source_filter=source_filter)
    typer.secho(f"✅ Deleted {deleted} documents", fg=get_system_color())


def index_text(content: str, source: Optional[str] = None):
    """Index text content into the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    source = source or "manual_input"
    
    doc_ids = rag.add_document(content, source)
    
    if len(doc_ids) == 1:
        typer.secho(f"✅ Document indexed successfully", fg=get_system_color())
        typer.secho(f"   ID: {doc_ids[0][:8]}...", fg=get_text_color())
    else:
        typer.secho(f"✅ Document indexed in {len(doc_ids)} chunks", fg=get_system_color())
        typer.secho(f"   Parent ID: {doc_ids[0].split('-')[0][:8]}...", fg=get_text_color())
    
    typer.secho(f"   Source: {source}", fg=get_text_color())
    typer.secho(f"   Words: {len(content.split())}", fg=get_text_color())


def index_file(filepath: str):
    """Index a file into the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    import os
    if not os.path.exists(filepath):
        typer.secho(f"File not found: {filepath}", fg="red")
        return
    
    # Check file size
    file_size = os.path.getsize(filepath)
    max_size = config.get('rag_max_file_size', 10 * 1024 * 1024)
    if file_size > max_size:
        typer.secho(f"File too large: {file_size / 1024 / 1024:.1f}MB (max: {max_size / 1024 / 1024:.1f}MB)", fg="red")
        return
    
    # Check allowed file types
    file_ext = os.path.splitext(filepath)[1].lower()
    allowed_types = config.get('rag_allowed_file_types', ['.txt', '.md', '.pdf', '.rst'])
    if file_ext not in allowed_types:
        typer.secho(f"Unsupported file type: {file_ext}", fg="red")
        typer.secho(f"Allowed types: {', '.join(allowed_types)}", fg="yellow")
        return
    
    # Determine file type and load accordingly
    if filepath.endswith('.pdf'):
        # Try to use existing PDF loading logic if available
        try:
            from episodic.commands.documents import extract_pdf_content
            content = extract_pdf_content(filepath)
            source = os.path.basename(filepath)
        except ImportError:
            typer.secho("PDF support not available. Install required dependencies.", fg="red")
            return
    else:
        # Read text file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            source = os.path.basename(filepath)
        except Exception as e:
            typer.secho(f"Error reading file: {e}", fg="red")
            return
    
    # Index the content
    index_text(content, source=source)


def rag_toggle(enable: Optional[bool] = None):
    """Enable or disable RAG functionality."""
    if enable is None:
        # Toggle current state
        current = config.get('rag_enabled', False)
        enable = not current
    
    config.set('rag_enabled', enable)
    
    status = "enabled" if enable else "disabled"
    typer.secho(f"RAG {status}", fg=get_system_color())
    
    if enable:
        # Initialize RAG system
        rag = get_rag_system()
        stats = rag.get_stats()
        typer.secho(f"Knowledge base: {stats['total_documents']} documents", fg=get_text_color())


def rag_stats():
    """Show RAG system statistics."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    stats = rag.get_stats()
    
    typer.secho("\n📊 RAG System Statistics", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    typer.secho("Documents indexed: ", nl=False, fg=get_text_color())
    typer.secho(f"{stats['total_documents']}", fg=get_system_color())
    
    typer.secho("Embedding model: ", nl=False, fg=get_text_color())
    typer.secho(f"{stats['embedding_model']}", fg=get_system_color())
    
    typer.secho("Auto-search: ", nl=False, fg=get_text_color())
    typer.secho(f"{config.get('rag_auto_search', True)}", fg=get_system_color())
    
    typer.secho("Search threshold: ", nl=False, fg=get_text_color())
    typer.secho(f"{config.get('rag_search_threshold', 0.7)}", fg=get_system_color())
    
    typer.secho("Max results: ", nl=False, fg=get_text_color())
    typer.secho(f"{config.get('rag_max_results', 5)}", fg=get_system_color())
    
    # Show source distribution
    sources = rag.get_source_distribution()
    if sources:
        typer.secho("\nDocument Sources:", fg=get_heading_color())
        for source, count in sources.items():
            typer.secho(f"  {source}: ", nl=False, fg=get_text_color())
            typer.secho(f"{count} documents", fg=get_system_color())
```

### 2.2 Update CLI Command Handler
Add to `episodic/cli.py` in the `handle_command` function:
```python
# RAG commands
elif cmd == "/search":
    if not args:
        typer.secho("Usage: /search <query>", fg="red")
    else:
        from episodic.commands.rag import search
        query = " ".join(args)
        search(query)

elif cmd == "/docs":
    from episodic.commands.rag import list_documents, show_document, remove_document, clear_documents
    if not args:
        # Default to listing documents
        list_documents()
    else:
        subcommand = args[0].lower()
        if subcommand == "list":
            limit = None
            source_filter = None
            # Parse optional arguments
            if "--limit" in args:
                idx = args.index("--limit")
                if idx + 1 < len(args):
                    limit = int(args[idx + 1])
            if "--source" in args:
                idx = args.index("--source")
                if idx + 1 < len(args):
                    source_filter = args[idx + 1]
            list_documents(limit=limit, source_filter=source_filter)
        elif subcommand == "show":
            if len(args) < 2:
                typer.secho("Usage: /docs show <doc_id>", fg="red")
            else:
                show_document(args[1])
        elif subcommand == "remove":
            if len(args) < 2:
                typer.secho("Usage: /docs remove <doc_id>", fg="red")
            else:
                remove_document(args[1])
        elif subcommand == "clear":
            source_filter = None
            if "--source" in args:
                idx = args.index("--source")
                if idx + 1 < len(args):
                    source_filter = args[idx + 1]
            clear_documents(source_filter=source_filter)
        else:
            typer.secho("Usage: /docs [list|show|remove|clear]", fg="red")

elif cmd == "/index":
    if not args:
        typer.secho("Usage: /index <file_path> or /index --text '<content>'", fg="red")
    else:
        from episodic.commands.rag import index_file, index_text
        if args[0] == "--text" and len(args) > 1:
            content = " ".join(args[1:])
            index_text(content)
        else:
            index_file(args[0])

elif cmd == "/rag":
    from episodic.commands.rag import rag_toggle, rag_stats
    if not args:
        rag_stats()
    elif args[0].lower() in ["on", "off"]:
        enable = args[0].lower() == "on"
        rag_toggle(enable)
    else:
        typer.secho("Usage: /rag [on|off]", fg="red")

elif cmd == "/index-dir":
    if not args:
        typer.secho("Usage: /index-dir <directory> [--pattern '*.txt'] [--recursive]", fg="red")
    else:
        from episodic.commands.rag import index_directory
        directory = args[0]
        pattern = "*"
        recursive = False
        
        if "--pattern" in args:
            idx = args.index("--pattern")
            if idx + 1 < len(args):
                pattern = args[idx + 1]
        
        if "--recursive" in args:
            recursive = True
        
        index_directory(directory, pattern, recursive)

elif cmd == "/export-kb":
    if not args:
        typer.secho("Usage: /export-kb <output_file>", fg="red")
    else:
        from episodic.commands.rag import export_knowledge_base
        export_knowledge_base(args[0])

elif cmd == "/import-kb":
    if not args:
        typer.secho("Usage: /import-kb <input_file>", fg="red")
    else:
        from episodic.commands.rag import import_knowledge_base
        import_knowledge_base(args[0])

elif cmd == "/test-rag":
    if not args:
        typer.secho("Usage: /test-rag <query> [--expect <source>]", fg="red")
    else:
        from episodic.commands.rag import test_rag_retrieval
        query = args[0]
        expected = None
        
        if "--expect" in args:
            idx = args.index("--expect")
            if idx + 1 < len(args):
                expected = args[idx + 1]
        
        test_rag_retrieval(query, expected)
```

## Phase 3: Conversation Integration (Week 2)

### 3.1 Enhance Conversation Manager
Update `episodic/conversation.py`:
```python
def calculate_rag_token_budget(self, model: str, base_message: str, 
                              system_message: str, context_messages: List[str]) -> int:
    """Calculate available tokens for RAG context."""
    from episodic.configuration import get_model_context_limit
    from episodic.llm import count_tokens
    
    # Get model's context limit
    model_limit = get_model_context_limit(model)
    
    # Count existing tokens
    total_tokens = count_tokens(system_message, model)
    total_tokens += count_tokens(base_message, model)
    for msg in context_messages:
        total_tokens += count_tokens(msg, model)
    
    # Reserve tokens for response and safety margin
    response_reserve = 1000
    safety_margin = 500
    
    available_tokens = model_limit - total_tokens - response_reserve - safety_margin
    return max(0, available_tokens)

def enhance_with_rag(self, user_message: str, model: str, 
                    system_message: str, context_messages: List[str]) -> Tuple[str, List[Dict], List[str]]:
    """Enhance user message with RAG context if enabled."""
    if not config.get('rag_enabled', False) or not config.get('rag_auto_search', True):
        return user_message, [], []
    
    from episodic.rag import get_rag_system
    from episodic.llm import count_tokens
    
    rag = get_rag_system()
    if not rag:
        return user_message, [], []
    
    # Calculate token budget for RAG
    token_budget = self.calculate_rag_token_budget(
        model, user_message, system_message, context_messages
    )
    
    if token_budget < 100:  # Not enough space for RAG
        if config.get('debug'):
            typer.secho(f"⚠️  Insufficient token budget for RAG: {token_budget}", fg="yellow")
        return user_message, [], []
    
    # Search for relevant context
    results = rag.search(
        user_message,
        n_results=config.get('rag_max_results', 10),  # Get more, filter by tokens
        threshold=config.get('rag_search_threshold', 0.7)
    )
    
    if not results['documents']:
        return user_message, [], []
    
    # Select documents that fit within token budget
    context_parts = []
    citations = []
    used_doc_ids = []
    total_tokens = 0
    
    for doc, metadata, doc_id in zip(
        results['documents'], 
        results['metadatas'],
        results['ids']
    ):
        doc_tokens = count_tokens(doc, model)
        if total_tokens + doc_tokens <= token_budget:
            context_parts.append(doc)
            citations.append({
                'id': doc_id,
                'source': metadata.get('source', 'Unknown'),
                'snippet': doc[:100] + "..." if len(doc) > 100 else doc
            })
            used_doc_ids.append(doc_id)
            total_tokens += doc_tokens
        else:
            break
    
    if not context_parts:
        return user_message, [], []
    
    # Add context to message
    context_prefix = config.get('rag_context_prefix', '\n\n[Relevant context from knowledge base]:\n')
    enhanced_message = user_message + context_prefix + "\n\n".join(context_parts)
    
    if config.get('debug'):
        typer.secho(f"📚 Added {len(context_parts)} RAG chunks ({total_tokens} tokens)", fg="cyan")
    
    return enhanced_message, citations, used_doc_ids
```

## Phase 4: Advanced Features (Week 3)

### 4.1 Batch Operations
```python
def index_directory(directory_path: str, pattern: str = "*", recursive: bool = False):
    """Index all matching files in a directory."""
    import glob
    import os
    
    if not os.path.isdir(directory_path):
        typer.secho(f"Directory not found: {directory_path}", fg="red")
        return
    
    # Get matching files
    if recursive:
        pattern_path = os.path.join(directory_path, "**", pattern)
        files = glob.glob(pattern_path, recursive=True)
    else:
        pattern_path = os.path.join(directory_path, pattern)
        files = glob.glob(pattern_path)
    
    # Filter by allowed types
    allowed_types = config.get('rag_allowed_file_types', ['.txt', '.md', '.pdf', '.rst'])
    files = [f for f in files if os.path.splitext(f)[1].lower() in allowed_types]
    
    if not files:
        typer.secho("No matching files found.", fg="yellow")
        return
    
    success_count = 0
    with typer.progressbar(files, label="Indexing files") as progress:
        for file_path in progress:
            try:
                index_file(file_path)
                success_count += 1
            except Exception as e:
                if config.get('debug'):
                    typer.secho(f"\nFailed to index {file_path}: {e}", fg="red")
    
    typer.secho(f"\n✅ Indexed {success_count}/{len(files)} files", fg=get_system_color())


def export_knowledge_base(output_path: str):
    """Export entire knowledge base for backup/sharing."""
    rag = get_rag_system()
    if not rag:
        typer.secho("RAG system not available.", fg="red")
        return
    
    docs = rag.list_documents()
    
    export_data = {
        'version': '1.0',
        'exported_at': datetime.now().isoformat(),
        'document_count': len(docs),
        'documents': []
    }
    
    for doc in docs:
        full_doc = rag.get_document(doc['id'])
        if full_doc:
            export_data['documents'].append({
                'id': full_doc['id'],
                'content': full_doc['content'],
                'metadata': full_doc['metadata']
            })
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        typer.secho(f"✅ Exported {len(docs)} documents to {output_path}", fg=get_system_color())
    except Exception as e:
        typer.secho(f"Export failed: {e}", fg="red")


def import_knowledge_base(input_path: str):
    """Import knowledge base from backup."""
    if not os.path.exists(input_path):
        typer.secho(f"File not found: {input_path}", fg="red")
        return
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
    except Exception as e:
        typer.secho(f"Failed to read import file: {e}", fg="red")
        return
    
    # Validate version
    if import_data.get('version') != '1.0':
        typer.secho(f"Incompatible export version: {import_data.get('version')}", fg="red")
        return
    
    rag = get_rag_system()
    if not rag:
        typer.secho("RAG system not available.", fg="red")
        return
    
    # Import documents
    imported = 0
    skipped = 0
    
    with typer.progressbar(import_data['documents'], label="Importing documents") as progress:
        for doc_data in progress:
            # Check if already exists
            content_hash = rag.calculate_content_hash(doc_data['content'])
            if rag.check_duplicate(content_hash):
                skipped += 1
                continue
            
            # Import document
            try:
                rag.add_document(
                    content=doc_data['content'],
                    source=doc_data['metadata'].get('source', 'imported'),
                    metadata=doc_data['metadata']
                )
                imported += 1
            except Exception as e:
                if config.get('debug'):
                    typer.secho(f"\nFailed to import document: {e}", fg="red")
    
    typer.secho(f"\n✅ Imported {imported} documents, skipped {skipped} duplicates", fg=get_system_color())


### 4.2 Web Search Integration
```python
def index_web_page(url: str):
    """Index a web page into the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    typer.secho(f"Fetching content from {url}...", fg=get_text_color())
    
    try:
        # Try to use WebFetch if available
        from episodic.commands.web import fetch_web_content
        content = fetch_web_content(url)
    except ImportError:
        typer.secho("Web fetching not available. Install required dependencies.", fg="red")
        return
    
    if content:
        # Clean up HTML artifacts if any
        import re
        content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
        content = re.sub(r'\s+', ' ', content)     # Normalize whitespace
        
        index_text(content, source=url)
    else:
        typer.secho("Failed to fetch web content.", fg="red")
```

### 4.3 Topic-Aware RAG
```python
def index_conversation_topic(topic_id: str):
    """Index a completed conversation topic into RAG for future reference."""
    from episodic.db import get_topic_content, get_topic_by_id
    
    topic = get_topic_by_id(topic_id)
    if not topic:
        typer.secho(f"Topic not found: {topic_id}", fg="red")
        return
    
    # Get all messages in the topic
    messages = get_topic_content(topic_id)
    
    # Format as Q&A pairs for better retrieval
    qa_pairs = []
    for i in range(0, len(messages) - 1, 2):
        if messages[i]['role'] == 'user' and i + 1 < len(messages):
            qa_pairs.append(f"Q: {messages[i]['content']}\nA: {messages[i+1]['content']}")
    
    if qa_pairs:
        content = "\n\n".join(qa_pairs)
        
        # Index with special metadata
        rag = get_rag_system()
        if rag:
            doc_ids = rag.add_document(
                content=content,
                source=f"topic:{topic['name']}",
                metadata={
                    'type': 'conversation_topic',
                    'topic_id': topic_id,
                    'topic_name': topic['name'],
                    'message_count': len(messages)
                }
            )
            
            typer.secho(f"✅ Indexed topic '{topic['name']}' ({len(qa_pairs)} Q&A pairs)", fg=get_system_color())
```

## Phase 5: Production Enhancements (Week 4)

### 5.1 Performance Optimizations
- Implement async searching
- Add result caching
- Batch embedding generation

### 5.2 Advanced Search
- Implement hybrid search (keyword + semantic)
- Add metadata filtering
- Support complex queries

### 5.3 User Feedback
- Track which results were helpful
- Improve relevance over time
- Allow result voting

## Migration Path

### From Simple to Advanced
1. Start with ChromaDB embedded
2. Add LangChain for complex queries
3. Consider Pinecone/Weaviate for scale
4. Evaluate Haystack for enterprise

### Data Management
1. Export/import functionality
2. Backup and restore
3. Privacy controls
4. Data retention policies

## Success Metrics

1. **Usage Metrics**
   - RAG queries per session
   - Documents indexed
   - Search relevance scores

2. **Performance Metrics**
   - Search latency
   - Indexing speed
   - Memory usage

3. **Quality Metrics**
   - User satisfaction
   - Citation accuracy
   - Response improvement

## Risk Mitigation

1. **Graceful Degradation**
   - RAG failures don't break conversations
   - Fallback to non-RAG mode
   - Clear error messages

2. **Resource Management**
   - Configurable index size limits
   - Memory usage monitoring
   - Cleanup policies

3. **Privacy Protection**
   - Local-only by default
   - No automatic web indexing
   - Clear data ownership

## Testing and Validation

### Test Commands
Add to `episodic/commands/rag.py`:
```python
def test_rag_retrieval(query: str, expected_source: Optional[str] = None):
    """Test RAG retrieval quality."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    if not rag:
        return
    
    # Perform search
    results = rag.search(query, n_results=10)
    
    typer.secho(f"\n🧪 Testing RAG retrieval for: '{query}'", fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    if not results['documents']:
        typer.secho("❌ No results found", fg="red")
        return
    
    # Check if expected source is in results
    if expected_source:
        sources = [m.get('source', '') for m in results['metadatas']]
        if any(expected_source in s for s in sources):
            rank = next(i for i, s in enumerate(sources) if expected_source in s) + 1
            typer.secho(f"✅ Expected source found at rank {rank}", fg="green")
        else:
            typer.secho(f"❌ Expected source '{expected_source}' not found", fg="red")
    
    # Show top results
    typer.secho("\nTop 3 results:", fg=get_heading_color())
    for i in range(min(3, len(results['documents']))):
        relevance = 1 - results['distances'][i]
        typer.secho(f"\n[{i+1}] Relevance: {relevance:.2%}", fg=get_system_color())
        typer.secho(f"Source: {results['metadatas'][i].get('source', 'Unknown')}", fg=get_text_color())
        snippet = results['documents'][i][:150] + "..."
        typer.secho(f"{snippet}", fg=get_text_color())


def benchmark_rag_performance():
    """Benchmark RAG system performance."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    if not rag:
        return
    
    import time
    
    typer.secho("\n⚡ RAG Performance Benchmark", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How to implement error handling?",
        "Best practices for code optimization",
        "Database design patterns",
        "API authentication methods"
    ]
    
    total_time = 0
    for query in test_queries:
        start = time.time()
        results = rag.search(query, n_results=5)
        elapsed = time.time() - start
        total_time += elapsed
        
        typer.secho(f"Query: '{query[:30]}...'", fg=get_text_color())
        typer.secho(f"  Time: {elapsed*1000:.1f}ms", fg=get_system_color())
        typer.secho(f"  Results: {len(results['documents'])}", fg=get_system_color())
    
    avg_time = (total_time / len(test_queries)) * 1000
    typer.secho(f"\nAverage query time: {avg_time:.1f}ms", fg=get_heading_color(), bold=True)
    
    # Test indexing speed
    test_content = "This is a test document. " * 100  # ~100 words
    
    start = time.time()
    doc_ids = rag.add_document(test_content, "benchmark_test")
    index_time = time.time() - start
    
    typer.secho(f"\nIndexing time (100 words): {index_time*1000:.1f}ms", fg=get_heading_color(), bold=True)
    
    # Clean up test document
    for doc_id in doc_ids:
        rag.remove_document(doc_id)
```

### Integration Tests
Create `tests/integration/test_rag.py`:
```python
import unittest
from episodic.rag import EpisodicRAG

class TestRAGIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test RAG instance."""
        self.rag = EpisodicRAG()
        
    def test_document_chunking(self):
        """Test document chunking logic."""
        # Small document (should not chunk)
        small_doc = "This is a small document."
        chunks = self.rag.chunk_document(small_doc)
        self.assertEqual(len(chunks), 1)
        
        # Large document (should chunk)
        large_doc = " ".join(["word"] * 1000)
        chunks = self.rag.chunk_document(large_doc, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        
        # Verify overlap
        for i in range(len(chunks) - 1):
            chunk1_words = chunks[i]['text'].split()
            chunk2_words = chunks[i+1]['text'].split()
            # Last 20 words of chunk1 should be first 20 of chunk2
            self.assertEqual(chunk1_words[-20:], chunk2_words[:20])
    
    def test_duplicate_detection(self):
        """Test duplicate document detection."""
        content = "This is a unique test document."
        
        # First add should succeed
        doc_ids1 = self.rag.add_document(content, "test_source")
        self.assertEqual(len(doc_ids1), 1)
        
        # Second add should detect duplicate
        doc_ids2 = self.rag.add_document(content, "test_source")
        self.assertEqual(doc_ids1[0], doc_ids2[0])
    
    def test_search_relevance(self):
        """Test search relevance scoring."""
        # Add test documents
        docs = [
            ("Python is a programming language", "python_intro"),
            ("JavaScript is used for web development", "js_intro"),
            ("Python is great for data science", "python_ds")
        ]
        
        for content, source in docs:
            self.rag.add_document(content, source)
        
        # Search for Python-related content
        results = self.rag.search("Python programming", n_results=3)
        
        # Python documents should rank higher
        sources = [m['source'] for m in results['metadatas']]
        self.assertIn('python_intro', sources[:2])
        self.assertIn('python_ds', sources[:2])
    
    def tearDown(self):
        """Clean up test data."""
        # Clear all test documents
        self.rag.clear_documents()
```

## Conclusion

This enhanced implementation plan provides a comprehensive approach to adding RAG capabilities to Episodic:

1. **Robust Foundation**: Document chunking, duplicate detection, and error handling
2. **Complete Document Management**: Full CRUD operations with safety measures
3. **Performance Aware**: Token budget management and efficient retrieval
4. **Production Ready**: Export/import, batch operations, and testing infrastructure
5. **User Friendly**: Progress indicators, helpful error messages, and validation

The phased approach allows for incremental development while maintaining system stability and user experience quality.