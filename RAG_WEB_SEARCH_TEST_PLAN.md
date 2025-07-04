# RAG and Web Search Testing Plan

## Overview
This document outlines the comprehensive testing approach for the newly implemented RAG (Retrieval Augmented Generation) and Web Search features in Episodic.

## Test Structure

### 1. Unit Tests
Created comprehensive unit test suites in:
- `tests/integration/test_rag_integration.py` - RAG functionality tests
- `tests/integration/test_web_search_integration.py` - Web search tests

#### RAG Unit Tests Cover:
- Document indexing and retrieval
- Duplicate detection using content hashing
- Document chunking for large files
- Search functionality with relevance scoring
- Context enhancement for messages
- Document management (list, remove, clear)
- Source filtering
- Threshold-based filtering
- Statistics tracking

#### Web Search Unit Tests Cover:
- Search result caching with TTL
- Rate limiting implementation
- DuckDuckGo provider integration
- Search manager with caching/rate limiting
- CLI command functionality
- Statistics tracking
- Error handling

### 2. Interactive Test Scripts

#### `scripts/test_rag_web_search.py`
Main interactive test runner with options to:
1. Test RAG features independently
2. Test web search features independently
3. Test RAG + web search integration
4. Run all unit tests
5. Run comprehensive test suite

#### `scripts/quick_rag_test.txt`
Quick smoke test script that:
- Initializes fresh database
- Tests basic RAG indexing and search
- Tests basic web search
- Tests integration between both features

### 3. Test Scenarios

#### RAG Testing:
1. **Document Indexing**
   - Index plain text content
   - Index files (txt, pdf)
   - Handle duplicates
   - Chunk large documents

2. **Search Operations**
   - Basic keyword search
   - Relevance scoring
   - Threshold filtering
   - Multi-document retrieval

3. **Context Enhancement**
   - Automatic context injection
   - Source citation tracking
   - Integration with chat messages

4. **Document Management**
   - List all documents
   - Filter by source
   - Remove specific documents
   - Clear entire collection

#### Web Search Testing:
1. **Basic Search**
   - Query DuckDuckGo
   - Parse HTML results
   - Format for display

2. **Caching**
   - Store results with TTL
   - Retrieve from cache
   - Cache statistics
   - Clear cache

3. **Rate Limiting**
   - Enforce request limits
   - Calculate wait times
   - Handle limit exceeded

4. **Integration**
   - Auto-enhance when RAG insufficient
   - Index web results for future use
   - Combine local and web sources

### 4. Integration Testing

Tests the synergy between RAG and web search:
- Fallback to web when local knowledge insufficient
- Automatic indexing of web results
- Combined context from both sources
- Proper source attribution

## Running Tests

### Quick Validation:
```bash
# Run quick smoke test
python -m episodic < scripts/quick_rag_test.txt

# Run interactive test suite
python scripts/test_rag_web_search.py
```

### Unit Tests:
```bash
# Run all RAG tests
python -m pytest tests/integration/test_rag_integration.py -v

# Run all web search tests  
python -m pytest tests/integration/test_web_search_integration.py -v

# Run specific test
python -m pytest tests/integration/test_rag_integration.py::TestRAGIntegration::test_document_indexing -v
```

### Manual Testing:
1. Start Episodic: `python -m episodic`
2. Enable features: `/rag on` and `/websearch on`
3. Test commands:
   - `/index <file>` - Index documents
   - `/search <query>` - Search knowledge base
   - `/websearch <query>` - Search the web
   - `/docs list` - List indexed documents
   - Ask questions to test context enhancement

## Expected Outcomes

### RAG System:
- ✅ Documents indexed successfully with deduplication
- ✅ Search returns relevant results with scoring
- ✅ Context automatically enhances chat responses
- ✅ Documents can be managed effectively
- ✅ Statistics track usage accurately

### Web Search:
- ✅ DuckDuckGo queries return relevant results
- ✅ Results cached to reduce API calls
- ✅ Rate limiting prevents abuse
- ✅ Errors handled gracefully
- ✅ Integration with RAG seamless

### Integration:
- ✅ Web search triggers when local knowledge insufficient
- ✅ Web results can be indexed for future use
- ✅ Sources properly attributed in responses
- ✅ Configuration controls behavior

## Configuration Options

### RAG Configuration:
- `rag_enabled` - Enable/disable RAG
- `rag_auto_search` - Auto-enhance messages
- `rag_chunk_size` - Document chunk size
- `rag_chunk_overlap` - Overlap between chunks
- `rag_max_results` - Max search results
- `rag_search_threshold` - Relevance threshold
- `rag_include_citations` - Show sources

### Web Search Configuration:
- `web_search_enabled` - Enable/disable web search
- `web_search_provider` - Search provider (duckduckgo)
- `web_search_auto_enhance` - Auto-search when needed
- `web_search_cache_duration` - Cache TTL in seconds
- `web_search_rate_limit` - Requests per hour
- `web_search_index_results` - Index results in RAG

## Known Limitations

1. **ChromaDB Telemetry**: Suppressed but may show warnings
2. **PDF Support**: Requires additional dependencies
3. **Rate Limiting**: DuckDuckGo has implicit limits
4. **Async Operations**: Some tests mock async behavior

## Next Steps

1. Add more search providers (Google, Bing)
2. Implement PDF and other document formats
3. Add embedding model configuration
4. Implement privacy features (Tor support)
5. Add result ranking algorithms
6. Create performance benchmarks