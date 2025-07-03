# RAG Integration Analysis for Episodic

## Executive Summary

This document analyzes the integration of Retrieval Augmented Generation (RAG) packages into Episodic to enable live search results within conversations. Based on research of current RAG frameworks in 2024, we recommend a phased approach using **ChromaDB** for vector storage and **LangChain** for orchestration, with the option to integrate **Haystack** for production deployments.

## Current Episodic Architecture Review

Episodic's current architecture:
- **Conversation Management**: Linear DAG-based conversation flow (currently no branching)
- **LLM Integration**: Multi-provider support via LiteLLM
- **Topic Detection**: Hybrid sliding window approach with drift detection
- **Storage**: SQLite database for conversation nodes and metadata
- **Compression**: Background async compression of topic segments

## RAG Package Comparison

### 1. LangChain
**Pros:**
- General-purpose framework with extensive LLM integration
- Strong community support and extensive documentation
- Flexible chaining capabilities for complex workflows
- Works well with ChromaDB and other vector stores
- Good for prototyping and experimentation

**Cons:**
- Can be over-engineered for simple use cases
- Performance overhead for basic RAG operations
- Steeper learning curve for advanced features

**Best for Episodic when:**
- Building complex search pipelines with multiple steps
- Need flexibility to experiment with different approaches
- Want to leverage existing LangChain integrations

### 2. LlamaIndex
**Pros:**
- Optimized specifically for indexing and retrieval
- Better performance for document-heavy workloads
- Simpler API for basic RAG operations
- Advanced features like sentence window retrieval
- Good integration with various data sources

**Cons:**
- Less flexible for non-retrieval workflows
- Smaller ecosystem compared to LangChain
- More focused scope may limit future extensibility

**Best for Episodic when:**
- Primary focus is on efficient document retrieval
- Working with large document collections
- Need optimized performance for search operations

### 3. Haystack
**Pros:**
- Production-ready with enterprise focus
- Built-in REST API for easy integration
- Most stable and mature of the frameworks
- Excellent for document-centric Q&A systems
- Strong support for different backends (Elasticsearch, FAISS)

**Cons:**
- Heavier framework with more dependencies
- May be overkill for simpler use cases
- Less flexible than LangChain for experimental features

**Best for Episodic when:**
- Planning for production deployment at scale
- Need enterprise-grade stability
- Want built-in API endpoints

### 4. ChromaDB (Vector Store)
**Pros:**
- Lightweight, embedded vector database
- Easy to integrate with Python applications
- Works well with all major RAG frameworks
- Good performance for small to medium datasets
- Simple API and minimal setup

**Cons:**
- Limited scalability compared to dedicated vector databases
- Fewer advanced features than specialized solutions

**Best for Episodic:**
- Ideal as the vector storage backend
- Can be used with any of the above frameworks
- Good starting point that can be migrated later

## Recommended Integration Approach

### Phase 1: Minimal RAG Implementation
```python
# Use ChromaDB directly with existing LiteLLM integration
import chromadb
from chromadb.utils import embedding_functions

class EpisodicRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="~/.episodic/chroma")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="episodic_knowledge",
            embedding_function=self.embedding_fn
        )
    
    def add_document(self, text: str, metadata: dict):
        """Add a document to the knowledge base."""
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[metadata.get('id', str(uuid.uuid4()))]
        )
    
    def search(self, query: str, n_results: int = 5):
        """Search for relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
```

### Phase 2: LangChain Integration
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

class EpisodicLangChainRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory="~/.episodic/chroma",
            embedding_function=self.embeddings
        )
        
    def create_qa_chain(self, llm):
        """Create a question-answering chain."""
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
        )
```

### Phase 3: Integration with Episodic Commands

New commands to add:
- `/search <query>` - Search knowledge base
- `/index <url/file>` - Add document to knowledge base
- `/rag on/off` - Enable/disable RAG for conversations
- `/rag-stats` - Show RAG usage statistics

### Phase 4: Conversation Enhancement

Modify `conversation.py` to:
1. Check if RAG is enabled for the conversation
2. For each user message, perform semantic search
3. Include relevant context in the LLM prompt
4. Track which documents were used for each response

## Implementation Recommendations

### 1. Start Simple
- Begin with ChromaDB + direct integration
- Add basic search and indexing commands
- Test with small document sets

### 2. Gradual Enhancement
- Add LangChain for more complex queries
- Implement document loaders for various formats
- Add web search integration

### 3. Production Considerations
- Consider Haystack for production deployments
- Plan for vector database migration path
- Implement proper document versioning

### 4. Integration Points

**Database Schema Changes:**
```sql
-- New tables for RAG integration
CREATE TABLE rag_documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE TABLE rag_retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT REFERENCES nodes(id),
    document_id TEXT REFERENCES rag_documents(id),
    relevance_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Configuration Options:**
```python
# Add to config_defaults.py
RAG_DEFAULTS = {
    'rag_enabled': False,
    'rag_auto_search': True,
    'rag_search_threshold': 0.7,
    'rag_max_results': 5,
    'rag_embedding_model': 'all-MiniLM-L6-v2',
    'rag_include_citations': True
}
```

## Benefits for Episodic

1. **Enhanced Context**: Conversations can reference external knowledge
2. **Dynamic Information**: Access to up-to-date information beyond training data
3. **Personalization**: Users can build their own knowledge bases
4. **Topic Coherence**: Better topic detection with semantic understanding
5. **Compression Quality**: Improved summaries using relevant context

## Risks and Mitigation

1. **Performance Impact**: 
   - Mitigation: Async search, caching, selective RAG activation

2. **Storage Growth**:
   - Mitigation: Document limits, compression, cleanup policies

3. **Relevance Quality**:
   - Mitigation: Tunable thresholds, user feedback, citation tracking

4. **Privacy Concerns**:
   - Mitigation: Local-only storage, encrypted vectors, clear data policies

## Conclusion

For Episodic's use case, we recommend:

1. **Immediate**: Implement Phase 1 with ChromaDB for basic RAG functionality
2. **Short-term**: Add LangChain for enhanced retrieval capabilities
3. **Long-term**: Evaluate Haystack for production deployments
4. **Optional**: Consider LlamaIndex if document-heavy workloads become primary

This approach provides flexibility while maintaining Episodic's architectural simplicity and allows for gradual enhancement based on user needs.