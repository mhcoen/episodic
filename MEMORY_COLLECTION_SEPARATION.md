# Memory Collection Separation - Implementation Plan

## Current State
- Single collection `episodic_docs` stores everything (conversations, user docs)
- Help system already uses separate `episodic_help` collection
- Unused `episodic_conversation_memory` collection exists in `rag_memory_sqlite.py`

## Target State
Three separate collections:
1. `episodic_help` - System documentation (✓ already separate)
2. `episodic_user_docs` - User-indexed documents only
3. `episodic_conversation_memory` - Conversation memories only

## Implementation Steps

### Step 1: Refactor RAG System Architecture

#### 1.1 Create Multi-Collection RAG Class
```python
class MultiCollectionRAG:
    """RAG system that manages multiple collections."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(...)
        self.collections = {}
        self._init_collections()
    
    def _init_collections(self):
        """Initialize all collections."""
        # User documents collection
        self.collections['user_docs'] = self._get_or_create_collection(
            'episodic_user_docs',
            'User indexed documents'
        )
        
        # Conversation memory collection
        self.collections['conversation'] = self._get_or_create_collection(
            'episodic_conversation_memory',
            'Conversation memories'
        )
        
        # Help collection (already exists separately)
        # Not managed here - handled by help.py
```

#### 1.2 Update EpisodicRAG Methods
- Add `collection_type` parameter to all methods
- Default to 'user_docs' for backward compatibility
- Route operations to appropriate collection

### Step 2: Update Storage Logic

#### 2.1 Conversation Storage
Update `conversation.py`:
```python
def store_conversation_to_memory(self, ...):
    # Store in conversation collection
    rag.add_document(
        content=conversation_text,
        source='conversation',
        metadata=metadata,
        collection_type='conversation'  # NEW
    )
```

#### 2.2 User Document Storage
Update RAG commands:
```python
def index_file(filepath: str):
    # Store in user_docs collection
    rag.add_document(
        content=content,
        source=source,
        collection_type='user_docs'  # NEW
    )
```

### Step 3: Update Search Logic

#### 3.1 Memory Commands
```python
def search_memories(query: str):
    # Search only conversation collection
    results = rag.search(
        query,
        collection_type='conversation'
    )
```

#### 3.2 RAG Commands
```python
def search(query: str):
    # Search only user_docs collection
    results = rag.search(
        query,
        collection_type='user_docs'
    )
```

#### 3.3 Context Builder
```python
def _add_rag_context(self, ...):
    # Search both collections if enabled
    conversation_results = []
    user_doc_results = []
    
    if config.get("system_memory_auto_context", True):
        conversation_results = rag.search(
            query,
            collection_type='conversation'
        )
    
    if config.get("rag_enabled", False):
        user_doc_results = rag.search(
            query,
            collection_type='user_docs'
        )
    
    # Combine results with source indicators
```

### Step 4: Migration Strategy

#### 4.1 Auto-Migration on Startup
```python
def migrate_to_separate_collections():
    """One-time migration of existing data."""
    # 1. Check if migration needed
    if migration_completed():
        return
    
    # 2. Get all documents from episodic_docs
    old_docs = get_all_from_collection('episodic_docs')
    
    # 3. Move to appropriate collections
    for doc in old_docs:
        if doc.source == 'conversation':
            move_to_collection('episodic_conversation_memory')
        else:
            move_to_collection('episodic_user_docs')
    
    # 4. Mark migration complete
    set_migration_flag()
```

#### 4.2 Migration Safety
- Keep original data until migration verified
- Add rollback capability
- Log all migration actions

### Step 5: Configuration Updates

#### 5.1 New Settings
```python
# config_defaults.py
'memory_collection': 'episodic_conversation_memory',
'user_docs_collection': 'episodic_user_docs',
'enable_collection_migration': True
```

### Step 6: Testing Plan

1. **Unit Tests**
   - Test each collection independently
   - Test migration logic
   - Test backward compatibility

2. **Integration Tests**
   - Test memory commands with new collections
   - Test RAG commands with new collections
   - Test context enhancement with both

3. **Migration Tests**
   - Test migration with mixed data
   - Test migration rollback
   - Test idempotency

### Step 7: Rollout Plan

1. **Phase 1**: Implement multi-collection support
2. **Phase 2**: Add migration logic (disabled by default)
3. **Phase 3**: Test extensively
4. **Phase 4**: Enable migration with user confirmation
5. **Phase 5**: Remove old collection references

## Benefits

1. **Clear Separation**: Each data type in its own collection
2. **Better Performance**: Targeted searches are faster
3. **Easier Maintenance**: Clear boundaries between systems
4. **Future Flexibility**: Easy to add new collection types

## Risks & Mitigation

1. **Data Loss**: Mitigated by keeping backups, testing migration
2. **Breaking Changes**: Mitigated by backward compatibility layer
3. **Performance**: Mitigated by optimizing collection sizes

## Success Criteria

- [ ] All memory commands work with conversation collection
- [ ] All RAG commands work with user docs collection
- [ ] Context enhancement searches appropriate collections
- [ ] Migration completes without data loss
- [ ] Performance improves or stays same
- [ ] No breaking changes for users