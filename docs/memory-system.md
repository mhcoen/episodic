# Episodic Memory System Documentation

## Overview

Episodic has a dual-memory architecture that provides both persistent conversation history and optional document-based knowledge enhancement:

1. **System Memory** (Always On): Automatically stores and retrieves your conversation history
2. **User RAG** (Optional): Indexes your documents for knowledge-base queries

This separation ensures conversations are always remembered while keeping document search optional.

## Understanding the Dual-Memory System

### System Memory (Always On)
Think of this as Episodic's "conversation memory" - it remembers everything you've discussed:
- **What it stores**: All your conversations, organized by topic
- **When it's used**: Automatically provides context from past discussions
- **Why it's always on**: Ensures continuity across sessions
- **Example**: "Remember when we discussed that Python bug?" - Episodic will find it

### User RAG (Optional)
Think of this as your "personal knowledge base" - for your documents and files:
- **What it stores**: Documents you explicitly index (PDFs, markdown, code)
- **When it's used**: When you enable it and index documents
- **Why it's optional**: Not everyone needs document search
- **Example**: Index your project docs, then ask questions about them

## Key Features

### System Memory (Conversation History)
- **Always Active**: Cannot be disabled - ensures continuity
- **Automatic Storage**: Every conversation is preserved
- **Smart Retrieval**: Finds relevant past discussions
- **Topic-Aware**: Organized by detected conversation topics
- **Persistent**: Survives application restarts

### User RAG (Document Knowledge Base)
- **Optional**: Enable with `/rag on`
- **Document Indexing**: Add PDFs, markdown, text files
- **Semantic Search**: Find information across documents
- **Web Integration**: Index web search results
- **Privacy-First**: All data stays local

## Configuration

### System Memory Configuration
```bash
# System memory is always on by default
/set system_memory_enabled true        # Cannot be disabled in normal use
/set system_memory_auto_store true     # Auto-store conversations
/set system_memory_auto_context true   # Auto-use in responses
/set memory_relevance_threshold 0.3    # Minimum relevance (0.0-1.0)
```

### User RAG Configuration
```bash
# Enable/disable document RAG
/rag on                                # Enable document search
/rag off                               # Disable document search

# Configure RAG behavior
/set rag_auto_search true              # Auto-search on queries
/set rag_search_threshold 0.7          # Relevance threshold
/set rag_max_results 5                 # Max results to include
```

### Configure Auto-Enhancement
```bash
# Enable automatic context enhancement (default)
/set rag-auto-enhance true

# Disable automatic enhancement
/set rag-auto-enhance false
```

### Adjust Search Parameters
```bash
# Set number of search results (default: 5)
/set rag-search-results 10

# Set chunk size for document splitting (default: 1000)
/set rag-chunk-size 500

# Set chunk overlap (default: 200)
/set rag-chunk-overlap 100
```

## Memory Commands

### System Memory Commands

#### `/memory` - Search Conversation History

List recent memories:
```bash
/memory
# Shows 10 most recent memories with previews
```

Search for specific content:
```bash
/memory search python async
# Searches all memories for "python async"
```

Show detailed memory entry:
```bash
/memory show abc12345
# Shows full details of memory with ID abc12345
```

List more memories:
```bash
/memory list 25
# Shows 25 most recent memories
```

#### `/forget` - Remove Conversation Memories

Remove specific memory:
```bash
/forget abc12345
# Removes memory with ID abc12345
```

Remove memories containing text:
```bash
/forget --contains "outdated info"
# Removes all memories containing "outdated info"
```

Remove memories by source:
```bash
/forget --source web
# Removes all web search memories
```

Clear all memories (with confirmation):
```bash
/forget --all
# Removes all memories after confirmation
```

#### `/memory-stats` - View Memory Statistics

```bash
/memory-stats
# Shows total documents, chunks, storage usage, etc.
```

## Example Interactions

### Example 1: Automatic Context from Previous Conversations

```bash
> Can you help me implement a binary search tree in Python?

[Assistant provides implementation]

> I need to add a delete method to it

💭 Using memory context from previous conversation
[Assistant continues with the previous BST implementation and adds delete method]
```

### Example 2: Learning from Indexed Documentation

```bash
> /index project-guidelines.md
✅ Document indexed with ID: a1b2c3d4 (15 chunks)

> How should I name my functions according to our guidelines?

💭 Using memory from: project-guidelines.md
According to your project guidelines, functions should:
- Use snake_case naming convention
- Start with a verb (get_, set_, calculate_, etc.)
- Be descriptive but concise
[...]
```

### Example 3: Web Search Memory

```bash
> /web search best practices for REST API design

🌐 Searching web...
[Results shown and stored]

[Later in a different session:]

> What were those REST API principles we looked at?

💭 Using memory from: web search results
Based on our previous search, the key REST API principles were:
- Use proper HTTP methods (GET, POST, PUT, DELETE)
- Resource-based URLs
- Stateless communication
[...]
```

### Example 4: Complex Project Context

```bash
> I'm working on a Django e-commerce site with React frontend

[Conversation continues about the project...]

[Days later:]

> How should I handle authentication between the frontend and backend?

💭 Using memory context from your Django/React e-commerce project
For your Django backend with React frontend, I recommend:
- Use Django REST Framework's token authentication
- Store JWT tokens securely in httpOnly cookies
- Implement refresh token rotation
[Based on previous context about your specific setup...]
```

### Example 5: Memory Management Workflow

```bash
> /memory
📚 Memory Entries
──────────────────────────────
📌 [a1b2c3d4] 2024-01-15 10:30
   Project guidelines including naming conventions, code style, and architecture decisions...
   
💬 [e5f6g7h8] 2024-01-15 09:15  
   Discussion about implementing binary search tree with insertion and traversal methods...
   
🌐 [i9j0k1l2] 2024-01-14 16:45
   REST API best practices including HTTP methods, status codes, and resource design...

> /memory search authentication

🔍 Searching memories for: authentication
Found 3 matches:
1. 💬 [m3n4o5p6] (relevance: 0.92)
   Django REST Framework authentication setup with JWT tokens and refresh rotation...
   
2. 📄 [q7r8s9t0] (relevance: 0.85)
   Security guidelines covering authentication, authorization, and session management...

> /forget i9j0k1l2
✅ Removed memory: i9j0k1l2

> /memory-stats
📊 Memory System Statistics
──────────────────────────
General:
  Total documents: 42
  Total chunks: 156
  Avg chunks/doc: 3.7
  Total retrievals: 89

Documents by Source:
  💬 conversation: 28
  📄 file: 10
  🌐 web: 4
```

## How Memory Context Works

1. **Query Analysis**: When you send a message, the system analyzes it for key concepts
2. **Memory Search**: Relevant memories are retrieved using semantic search
3. **Context Injection**: If relevant memories are found, they're included as context
4. **Response Generation**: The LLM uses both your query and the memory context
5. **Memory Storage**: The conversation is stored for future reference

## Best Practices

### 1. Index Important Documents
```bash
# Index project documentation
/index README.md
/index docs/architecture.md
/index requirements.txt
```

### 2. Regularly Review Memories
```bash
# Check what's being stored
/memory list 20

# Remove outdated information
/forget --contains "old API version"
```

### 3. Use Descriptive Queries
- More specific queries retrieve better context
- Include project names, technologies, or unique identifiers
- The system learns from how you describe things

### 4. Manage Memory Growth
```bash
# Monitor memory usage
/memory-stats

# Clean up periodically
/forget --source web  # If web results accumulate
```

### Document RAG Commands

#### `/index` or `/i` - Index Documents
```bash
/index document.pdf              # Index a PDF
/i research_paper.md             # Index markdown (short form)
/index ~/Documents/notes.txt     # Index from path
```

#### `/search` or `/s` - Search Documents
```bash
/search quantum computing        # Search indexed documents
/s API authentication           # Short form search
```

#### `/docs` - Manage Documents
```bash
/docs                           # List all indexed documents
/docs show 1                    # Show document details
/docs remove 1                  # Remove a document
/docs clear                     # Remove all documents
```

## Privacy and Security

- All data stored locally in `~/.episodic/`
- System memory in SQLite database
- User RAG in ChromaDB vector store
- No external data sharing (except LLM API calls)
- Clear conversation memory: `/forget --all`
- Clear document index: `/docs clear`

## Troubleshooting

### Memory not being retrieved
1. Check if RAG is enabled: `/set rag-enabled`
2. Verify auto-enhance is on: `/set rag-auto-enhance`
3. Search manually: `/memory search <terms>`

### Too much irrelevant context
1. Reduce search results: `/set rag-search-results 3`
2. Remove irrelevant memories: `/forget <id>`
3. Be more specific in queries

### Performance issues
1. Check memory stats: `/memory-stats`
2. Clear old web searches: `/forget --source web`
3. Reduce chunk size: `/set rag-chunk-size 500`

## Advanced Features

### Custom Metadata
When indexing files, metadata is automatically extracted:
- File type and size
- Creation/modification dates
- Document structure (headings, code blocks)

### Chunk Management
Documents are split into overlapping chunks for better retrieval:
- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Adjustable via settings

### Relevance Scoring
- Memories are ranked by semantic similarity
- Recent memories may be weighted higher
- Source type affects ranking (conversation > file > web)