# RAG Implementation Plan for Episodic

## Overview

This plan outlines the implementation of RAG (Retrieval Augmented Generation) capabilities in Episodic using ChromaDB as the vector store and a phased approach that starts simple and grows more sophisticated.

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
            'word_count': len(content.split())
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        self.collection.add(
            documents=[content],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )
        
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
from typing import Optional

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
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'], 
        results['metadatas'], 
        results['distances']
    )):
        relevance = 1 - distance  # Convert distance to similarity
        typer.secho(f"\n[{i+1}] ", nl=False, fg=get_system_color(), bold=True)
        typer.secho(f"Relevance: {relevance:.2%}", fg=get_system_color())
        typer.secho(f"Source: {metadata.get('source', 'Unknown')}", fg=get_text_color())
        
        # Show snippet
        snippet = doc[:200] + "..." if len(doc) > 200 else doc
        typer.secho(f"{snippet}", fg=get_text_color())


def index_text(content: str, source: Optional[str] = None):
    """Index text content into the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    rag = get_rag_system()
    source = source or "manual_input"
    
    doc_id = rag.add_document(content, source)
    
    typer.secho(f"✅ Document indexed successfully", fg=get_system_color())
    typer.secho(f"   ID: {doc_id}", fg=get_text_color())
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
    
    # Read file content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        typer.secho(f"Error reading file: {e}", fg="red")
        return
    
    # Index with filename as source
    index_text(content, source=os.path.basename(filepath))


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