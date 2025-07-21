# Memory System Separation Plan

## Current State
The memory system is currently mixed with user RAG settings and old POC code. Conversations are stored based on `enable_memory_poc` or `enable_memory_rag` settings.

## Goal
Create a clean separation between:
1. **System Memory** - Always-on conversation history (like help system)
2. **User RAG** - Optional user document indexing via `/index`

## Implementation Steps

### 1. Configuration Changes ✅
- Added `system_memory_enabled` (default: true)
- Added `system_memory_auto_store` (default: true) 
- Added `system_memory_auto_context` (default: true)
- Keep `rag_enabled` for user documents only

### 2. Code Changes

#### ConversationManager Changes ✅
- Added `store_conversation_to_memory()` method
- Stores all conversations when `system_memory_auto_store` is true
- Independent of user RAG settings

#### Context Builder Changes ✅
- Updated `_add_rag_context()` to check both system memory and user RAG
- System memory provides conversation context
- User RAG provides document context

### 3. Cleanup Needed
- Remove old POC code checking `enable_memory_poc`
- Remove old code checking `enable_memory_rag` 
- Update memory commands to show source clearly

### 4. Testing
- Verify conversations are stored automatically
- Verify context works with system memory disabled
- Verify user RAG works independently
- Clear test data from memory

## Benefits
- Conversations always stored (unless explicitly disabled)
- Memory commands always work
- Clear separation of concerns
- No confusion about what memories are