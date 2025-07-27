# Memory System Manual Testing Guide

## Overview
This guide provides comprehensive manual testing procedures for the Episodic memory system. These tests complement the automated test suite and verify real-world usage scenarios.

## Prerequisites
- Episodic installed and running
- Access to the CLI
- Ability to monitor SQLite and ChromaDB databases

## Test Categories

### 1. Automatic Conversation Storage
**Goal**: Verify conversations are automatically stored in memory

**Test Steps**:
1. Start a fresh conversation
2. Send message: "What is the capital of France?"
3. Wait for response
4. Send message: "Tell me more about it"
5. Wait for response
6. Run `/memory list` - should show the conversation
7. Run `/memory search France` - should find the conversation

**Expected Results**:
- Both messages and responses stored
- Timestamps are correct
- Topic detection works
- Both SQLite and ChromaDB updated

### 2. Memory Search Functionality
**Goal**: Test search accuracy and relevance filtering

**Test Cases**:
1. **Exact match**: Search for exact phrase from conversation
2. **Partial match**: Search for related keywords
3. **No match**: Search for unrelated term
4. **Relevance threshold**:
   - Set threshold to 0.5: `/set memory_relevance_threshold 0.5`
   - Search for marginal match
   - Verify filtering works

**Expected Results**:
- High relevance for exact matches (>0.5)
- Medium relevance for partial matches (0.3-0.5)
- Low/no results for unrelated searches
- Threshold filtering works correctly

### 3. Memory Commands
**Goal**: Test all memory management commands

**Test `/memory list`**:
1. Run with no arguments
2. Run with limit: `/memory list 5`
3. Verify ordering (newest first)
4. Check metadata display

**Test `/memory show <id>`**:
1. Get an ID from list
2. Show full memory details
3. Try with partial ID (first 8 chars)
4. Try with invalid ID

**Test `/forget`**:
1. `/forget <id>` - specific memory
2. `/forget --contains "test"` - by content
3. `/forget --source conversation` - by source
4. `/forget --all` - clear all (with confirmation)

### 4. Memory Persistence
**Goal**: Verify memories persist across sessions

**Test Steps**:
1. Create several conversations
2. Note memory IDs
3. Exit Episodic completely
4. Restart Episodic
5. Run `/memory list`
6. Search for previous conversations

**Expected Results**:
- All memories still present
- Search still works
- IDs unchanged

### 5. Context Injection
**Goal**: Test memory-enhanced responses

**Test Steps**:
1. Have conversation about a specific topic (e.g., "Python programming")
2. Exit conversation
3. Start new conversation
4. Reference the topic indirectly: "What did we discuss about that programming language?"
5. Check if context was injected

**Expected Results**:
- System finds relevant past conversation
- Includes context in response
- Response acknowledges previous discussion

### 6. Edge Cases

**Empty/Special Queries**:
- Search for empty string: `/memory search ""`
- Search with special chars: `/memory search @#$%`
- Very long search query (>100 chars)

**Large Conversations**:
- Create very long conversation (20+ exchanges)
- Verify storage works
- Test search performance

**Concurrent Operations**:
- Rapidly send multiple messages
- Run memory commands while conversation active

### 7. Performance Testing
**Goal**: Verify system performs well with many memories

**Test Steps**:
1. Create 100+ conversation memories
2. Time search operations
3. Time list operations
4. Monitor memory usage

**Benchmarks**:
- Search should complete in <1 second
- List should complete in <500ms
- Memory usage should be reasonable

### 8. Integration Testing
**Goal**: Test complete workflows

**Workflow 1 - Research Assistant**:
1. Ask about a topic
2. Have follow-up questions
3. Change topic
4. Return to original topic later
5. Verify context is maintained

**Workflow 2 - Memory Management**:
1. Build up 20+ memories
2. Search for specific ones
3. Delete some selectively
4. Verify search still works
5. Check stats with `/memory-stats`

## Troubleshooting Checklist

If memories aren't being stored:
- [ ] Check `system_memory_auto_store` is true
- [ ] Check `collection_migration_completed` is true
- [ ] Verify SQLite database exists
- [ ] Check ChromaDB collections exist
- [ ] Look for errors in debug mode

If search isn't working:
- [ ] Verify memories exist with `/memory list`
- [ ] Check relevance threshold setting
- [ ] Try different search terms
- [ ] Check collection type routing

## Reporting Issues

When reporting memory system issues, include:
1. Exact commands/messages used
2. Output of `/memory-stats`
3. Relevant config settings
4. Debug output if available
5. SQLite row counts
6. ChromaDB collection counts