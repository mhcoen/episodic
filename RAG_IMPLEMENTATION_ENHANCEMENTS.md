# RAG Implementation Enhancements and Considerations

After reviewing the RAG implementation plan, here are critical enhancements and considerations that should be addressed:

## 1. Document Chunking Strategy

The current plan indexes entire documents, which will cause problems:
- Large documents will exceed embedding model limits
- Poor retrieval precision (returning entire documents instead of relevant sections)
- Inefficient token usage in LLM context

### Solution: Implement Smart Chunking
```python
def chunk_document(content: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """Split document into overlapping chunks for better retrieval."""
    words = content.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'text': chunk_text,
            'start_idx': i,
            'end_idx': min(i + chunk_size, len(words))
        })
    
    return chunks
```

## 2. Error Handling and Resilience

### Missing Error Scenarios:
- Embedding model download failures
- ChromaDB initialization errors
- Corrupt or malformed documents
- Network issues during web indexing

### Enhanced Error Handling:
```python
class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass

class EmbeddingError(RAGError):
    """Error during embedding generation."""
    pass

class IndexingError(RAGError):
    """Error during document indexing."""
    pass

def safe_init_rag():
    """Initialize RAG with proper error handling."""
    try:
        return EpisodicRAG()
    except Exception as e:
        typer.secho(f"⚠️  RAG initialization failed: {e}", fg="yellow")
        typer.secho("RAG features will be disabled for this session.", fg="yellow")
        config.set('rag_enabled', False)
        return None
```

## 3. Performance Optimizations

### Batch Processing:
```python
def index_directory(directory_path: str, pattern: str = "*"):
    """Index all matching files in a directory."""
    import glob
    files = glob.glob(os.path.join(directory_path, pattern))
    
    with typer.progressbar(files, label="Indexing files") as progress:
        for file_path in progress:
            try:
                index_file(file_path)
            except Exception as e:
                typer.secho(f"Failed to index {file_path}: {e}", fg="red")
```

### Embedding Cache:
```python
# Add to config_defaults.py
'rag_cache_embeddings': {
    'default': True,
    'doc': 'Cache generated embeddings to speed up re-indexing'
},
'rag_embedding_cache_size': {
    'default': 1000,
    'doc': 'Maximum number of cached embeddings'
}
```

## 4. Token Management

### Context Window Awareness:
```python
def calculate_rag_token_budget(model: str, base_context: str) -> int:
    """Calculate how many tokens available for RAG context."""
    model_limit = get_model_context_limit(model)
    base_tokens = count_tokens(base_context, model)
    safety_margin = 500  # Reserve tokens for response
    
    return max(0, model_limit - base_tokens - safety_margin)

def select_rag_results(results: Dict, token_budget: int, model: str) -> List[str]:
    """Select RAG results that fit within token budget."""
    selected = []
    total_tokens = 0
    
    for doc in results['documents']:
        doc_tokens = count_tokens(doc, model)
        if total_tokens + doc_tokens <= token_budget:
            selected.append(doc)
            total_tokens += doc_tokens
        else:
            break
    
    return selected
```

## 5. Security and Privacy

### Sensitive Data Handling:
```python
# Add to config_defaults.py
'rag_sanitize_content': {
    'default': True,
    'doc': 'Remove potential PII before indexing'
},
'rag_allowed_file_types': {
    'default': ['.txt', '.md', '.pdf', '.rst'],
    'doc': 'Allowed file extensions for indexing'
}

def sanitize_content(content: str) -> str:
    """Remove potential sensitive information."""
    # Remove email addresses
    content = re.sub(r'\S+@\S+', '[EMAIL]', content)
    
    # Remove phone numbers (basic pattern)
    content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', content)
    
    # Remove SSN-like patterns
    content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', content)
    
    return content
```

## 6. Duplicate Detection

```python
def calculate_content_hash(content: str) -> str:
    """Calculate hash for duplicate detection."""
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()

def check_duplicate(content_hash: str) -> Optional[str]:
    """Check if document already exists."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id FROM rag_documents WHERE content_hash = ?',
        (content_hash,)
    )
    result = cursor.fetchone()
    return result[0] if result else None
```

## 7. Export/Import Functionality

```python
def export_knowledge_base(output_path: str):
    """Export entire knowledge base for backup/sharing."""
    rag = get_rag_system()
    docs = rag.list_documents()
    
    export_data = {
        'version': '1.0',
        'exported_at': datetime.now().isoformat(),
        'documents': []
    }
    
    for doc in docs:
        full_doc = rag.get_document(doc['id'])
        export_data['documents'].append(full_doc)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

def import_knowledge_base(input_path: str):
    """Import knowledge base from backup."""
    with open(input_path, 'r') as f:
        import_data = json.load(f)
    
    # Validate version compatibility
    if import_data.get('version') != '1.0':
        raise ValueError("Incompatible export version")
    
    # Import documents
    for doc_data in import_data['documents']:
        # Re-index each document
        pass
```

## 8. Usage Analytics

```python
# Track which documents are most useful
def record_document_usage(node_id: str, doc_ids: List[str], relevance_scores: List[float]):
    """Track which documents were used in responses."""
    conn = get_connection()
    cursor = conn.cursor()
    
    for doc_id, score in zip(doc_ids, relevance_scores):
        cursor.execute('''
            INSERT INTO rag_usage_analytics 
            (node_id, document_id, relevance_score, used_at)
            VALUES (?, ?, ?, ?)
        ''', (node_id, doc_id, score, datetime.now()))
    
    conn.commit()
```

## 9. Advanced Search Features

```python
def search_with_filters(
    query: str,
    source_filter: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_relevance: float = 0.0
) -> Dict[str, Any]:
    """Advanced search with multiple filters."""
    # First get semantic search results
    results = rag.search(query)
    
    # Then apply filters
    filtered = {
        'documents': [],
        'metadatas': [],
        'distances': [],
        'ids': []
    }
    
    for i, metadata in enumerate(results['metadatas']):
        # Apply source filter
        if source_filter and source_filter not in metadata.get('source', ''):
            continue
        
        # Apply date filter
        indexed_at = datetime.fromisoformat(metadata.get('indexed_at', ''))
        if date_from and indexed_at < date_from:
            continue
        if date_to and indexed_at > date_to:
            continue
        
        # Apply relevance filter
        if 1 - results['distances'][i] < min_relevance:
            continue
        
        # Add to filtered results
        filtered['documents'].append(results['documents'][i])
        filtered['metadatas'].append(metadata)
        filtered['distances'].append(results['distances'][i])
        filtered['ids'].append(results['ids'][i])
    
    return filtered
```

## 10. Configuration Additions

```python
# Add to config_defaults.py
RAG_DEFAULTS = {
    # ... existing configs ...
    
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
    'rag_auto_index_topics': {
        'default': False,
        'doc': 'Automatically index completed conversation topics'
    }
}
```

## 11. Integration with Existing Features

### Topic-Aware Indexing:
```python
def index_topic_conversation(topic_id: str):
    """Index a completed topic into RAG for future reference."""
    # Get topic content
    topic_data = get_topic_content(topic_id)
    
    # Format as Q&A pairs
    content = format_topic_as_qa(topic_data)
    
    # Index with special metadata
    rag.add_document(
        content=content,
        source=f"topic:{topic_data['name']}",
        metadata={
            'type': 'conversation_topic',
            'topic_id': topic_id,
            'message_count': topic_data['message_count']
        }
    )
```

### Visual Feedback:
```python
def show_rag_status():
    """Show RAG status in the prompt or status line."""
    if config.get('rag_enabled'):
        stats = rag.get_stats()
        return f"📚 RAG: {stats['total_documents']} docs"
    return ""
```

## 12. Testing Considerations

### Test Commands:
```python
# Add test mode for RAG
def test_rag_retrieval(query: str, expected_doc_id: str):
    """Test if RAG retrieves expected document."""
    results = rag.search(query)
    
    if expected_doc_id in results['ids']:
        rank = results['ids'].index(expected_doc_id) + 1
        typer.secho(f"✅ Found at rank {rank}", fg="green")
    else:
        typer.secho(f"❌ Not found in top results", fg="red")
```

## 13. Migration Path

### Database Migrations:
```sql
-- Add new columns to support enhanced features
ALTER TABLE rag_documents ADD COLUMN content_hash TEXT;
ALTER TABLE rag_documents ADD COLUMN chunk_index INTEGER DEFAULT 0;
ALTER TABLE rag_documents ADD COLUMN parent_doc_id TEXT;

-- Create analytics table
CREATE TABLE rag_usage_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT REFERENCES nodes(id),
    document_id TEXT,
    relevance_score REAL,
    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    was_helpful BOOLEAN DEFAULT NULL
);

-- Create indices for performance
CREATE INDEX idx_rag_documents_hash ON rag_documents(content_hash);
CREATE INDEX idx_rag_documents_source ON rag_documents(source);
CREATE INDEX idx_rag_analytics_doc ON rag_usage_analytics(document_id);
```

## Summary of Critical Additions

1. **Document Chunking** - Essential for handling real-world documents
2. **Token Budget Management** - Prevents context overflow
3. **Duplicate Detection** - Avoids redundant storage
4. **Export/Import** - Enables backup and sharing
5. **Advanced Search** - More powerful retrieval options
6. **Usage Analytics** - Learn from user interactions
7. **Error Resilience** - Graceful handling of failures
8. **Security Features** - PII sanitization
9. **Performance Optimizations** - Batch operations, caching
10. **Testing Support** - Validation of RAG quality

These enhancements transform the basic RAG implementation into a production-ready system that can handle real-world usage patterns while maintaining Episodic's focus on simplicity and reliability.