# RAG Implementation Plan for Episodic

## Overview

This plan outlines the implementation of RAG (Retrieval Augmented Generation) capabilities in Episodic using ChromaDB as the vector store and a phased approach that starts simple and grows more sophisticated.

## Command Summary

### RAG Document Management Commands

- **`/rag [on|off]`** - Enable/disable RAG and show statistics
- **`/search <query>`** - Search the knowledge base for relevant documents
- **`/index <file>`** - Add a file to the knowledge base (supports .txt, .md, .pdf)
- **`/index --text "<content>"`** - Add text content directly to the knowledge base
- **`/docs`** - List all documents in the knowledge base (default action)
- **`/docs list [--limit N] [--source filter]`** - List documents with optional filters
- **`/docs show <doc_id>`** - Display full content of a specific document
- **`/docs remove <doc_id>`** - Remove a specific document (with confirmation)
- **`/docs clear [--source filter]`** - Clear all or filtered documents (with confirmation)

### Key Features

1. **Full CRUD Operations**: Create (index), Read (list/show), Update (via remove+add), Delete (remove/clear)
2. **Source Filtering**: Filter documents by source when listing or clearing
3. **Safety Confirmations**: Deletion operations require confirmation to prevent accidents
4. **Document IDs**: Each document gets a unique ID (shown truncated as 8 chars for display)
5. **Metadata Tracking**: Word count, indexing time, source information

### Relationship with Existing Commands

- **`/load`** remains for loading PDFs into current conversation context
- **`/index`** adds documents to persistent RAG knowledge base
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
from typing import List, Dict, Any, Optional
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
        
    def add_document(self, 
                    content: str, 
                    source: str,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the knowledge base."""
        doc_id = str(uuid.uuid4())
        
        doc_metadata = {
            'source': source,
            'indexed_at': datetime.now().isoformat(),
            'word_count': len(content.split()),
            'content': content  # Store content in metadata for retrieval
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        self.collection.add(
            documents=[content],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )
        
        # Also track in SQLite for additional management
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO rag_documents (id, content, source, metadata)
            VALUES (?, ?, ?, ?)
        ''', (doc_id, content, source, json.dumps(doc_metadata)))
        conn.commit()
        
        return doc_id
    
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


def get_rag_system() -> EpisodicRAG:
    """Get or create the global RAG system instance."""
    global rag_system
    if rag_system is None:
        rag_system = EpisodicRAG()
    return rag_system
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
            metadata JSON
        )
    ''')
    
    # Table for tracking which documents were used in responses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_retrievals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT REFERENCES nodes(id),
            document_id TEXT,
            relevance_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
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
    
    doc_id = rag.add_document(content, source)
    
    typer.secho(f"✅ Document indexed successfully", fg=get_system_color())
    typer.secho(f"   ID: {doc_id[:8]}...", fg=get_text_color())
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
```

## Phase 3: Conversation Integration (Week 2)

### 3.1 Enhance Conversation Manager
Update `episodic/conversation.py`:
```python
def enhance_with_rag(self, user_message: str) -> tuple[str, List[Dict]]:
    """Enhance user message with RAG context if enabled."""
    if not config.get('rag_enabled', False) or not config.get('rag_auto_search', True):
        return user_message, []
    
    from episodic.rag import get_rag_system
    rag = get_rag_system()
    
    # Search for relevant context
    results = rag.search(
        user_message,
        n_results=config.get('rag_max_results', 5),
        threshold=config.get('rag_search_threshold', 0.7)
    )
    
    if not results['documents']:
        return user_message, []
    
    # Build context from search results
    context_parts = []
    citations = []
    
    for doc, metadata in zip(results['documents'], results['metadatas']):
        context_parts.append(doc)
        citations.append({
            'source': metadata.get('source', 'Unknown'),
            'snippet': doc[:100] + "..." if len(doc) > 100 else doc
        })
    
    # Add context to message
    context_prefix = config.get('rag_context_prefix', '\n\n[Relevant context]:\n')
    enhanced_message = user_message + context_prefix + "\n\n".join(context_parts)
    
    return enhanced_message, citations
```

## Phase 4: Advanced Features (Week 3)

### 4.1 Web Search Integration
```python
def index_web_page(url: str):
    """Index a web page into the knowledge base."""
    from episodic.commands.documents import fetch_web_content
    
    content = fetch_web_content(url)
    if content:
        rag = get_rag_system()
        doc_id = rag.add_document(content, source=url)
        return doc_id
```

### 4.2 Document Loaders
```python
def index_pdf(filepath: str):
    """Index a PDF file."""
    # Use existing PDF loading from document commands
    pass

def index_markdown(filepath: str):
    """Index markdown files."""
    pass
```

### 4.3 Topic-Aware RAG
```python
def index_conversation_topic(topic_name: str, start_node: str, end_node: str):
    """Index a conversation topic into RAG for future reference."""
    # Extract topic content and index it
    pass
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

## Conclusion

This implementation plan provides a practical path to adding RAG capabilities to Episodic while maintaining its simplicity and reliability. The phased approach allows for testing and refinement at each stage.