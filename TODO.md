# Episodic Project TODO

## Completed ✅

### Topic Detection & Management
- [x] **Implement LLM-based topic detection** - Uses ollama/llama3 for efficient detection
- [x] **Add topic extraction function** - `extract_topic_ollama()` implemented
- [x] **Database schema** - Topics table with name, start/end nodes, confidence
- [x] **Implement `/topics` command** - Shows topics with ranges and node counts
- [x] **Topic storage/retrieval** - Full CRUD operations for topics
- [x] **Optional topic display** - `/set show_topics true` shows topic evolution
- [x] **Fix topic naming** - Topics named based on content, not trigger message
- [x] **Topic detection prompt optimization** - Simplified for ollama/llama3 compatibility
- [x] **Add `/rename-topics` command** - Renames placeholder "ongoing-*" topics
- [x] **Fix overlapping topics** - Topics now have proper boundaries
- [x] **Current topic tracking** - ConversationManager tracks current topic
- [x] **Fix JSON parsing errors** - Added robust fallback for Ollama responses

### Compression System
- [x] **Async background compression** - Thread-based worker with priority queue
- [x] **Auto-compression on topic change** - Topics compressed when closed
- [x] **Compression statistics** - `/compression-stats` command
- [x] **Configurable compression** - Model, min nodes, notifications via `/set`
- [x] **Fix compression nodes in tree** - Compressions stored separately, not as nodes

### UI/UX Improvements
- [x] **Color scheme adaptation** - Supports light/dark terminals
- [x] **Colored help display** - Commands and descriptions with proper formatting
- [x] **Benchmark system** - Performance tracking with `/benchmark` command
- [x] **Summary command** - `/summary [N|all]` for conversation summaries
- [x] **Benchmark display after commands** - Shows benchmarks after commands when enabled
- [x] **Fix streaming duplication** - Fixed text appearing twice in constant-rate mode
- [x] **Word wrapping improvements** - Proper word-by-word streaming with wrapping
- [x] **Markdown bold support** - **text** now appears as bold in terminal
- [x] **List indentation** - 6-space indentation for wrapped lines in lists

### Code Quality & Testing
- [x] **CLI code cleanup** - Removed unused imports and duplicate code
- [x] **Fix test suite** - Fixed cache and config initialization tests
- [x] **Add missing Config.delete()** - Added delete method for configuration values
- [x] **Simplify test infrastructure** - Removed over-engineered test setup
- [x] **Update documentation** - Cleaned up outdated testing docs

### Navigation & State Management
- [x] **Fix `/init --erase`** - Now properly resets conversation manager state
- [x] **Fix topics starting from node 02** - Query first user node directly

## In Progress 🚧

### Critical Issues
- [ ] **Fix dynamic topic threshold behavior** - Document and make configurable the behavior where first 2 topics use half threshold
- [ ] **Test configuration isolation** - Tests modify production config file (~/.episodic/config.json)

### Topic Management Enhancements
- [ ] **Running topic guess** - Update tentative topic names periodically
- [ ] **Fix first topic creation** - Initial topic not always created properly
- [ ] **Move topic detection to background** - Eliminate response delay (as noted in todo list)

## Pending 📋

### High Priority
- [ ] **Manual compression trigger** - `/compress topic-name` command
- [ ] **Topic-based navigation** - Jump to specific topics in history
- [ ] **Fix remaining test failures** - 7 tests still failing (93% passing)

### Medium Priority  
- [ ] **Improve drift accuracy** - Current embedding approach shows high drift for similar sentences
- [ ] **Async drift processing** - Calculate drift in background to reduce delays

### Low Priority
- [ ] **Move debug messages** - Topic detection debug should appear after LLM response
- [ ] **Topic refinement UI** - Allow manual topic name editing
- [ ] **Export topics** - Export topic summaries to markdown

## Technical Debt 🔧
- [ ] **Reduce drift calculation overhead** - Consider faster embedding model
- [ ] **Better embedding cache strategy** - Optimize to reduce redundant calculations
- [ ] **Test coverage for ML modules** - Add tests for drift.py, peaks.py, embeddings/
- [ ] **Update deprecated datetime.utcnow()** - Use timezone-aware datetime.datetime.now(datetime.UTC)

## Notes
- Topic detection uses configurable model (default: ollama/llama3)
- Compression happens automatically when topics close
- Current active topic shows as "ongoing-discussion" until closed
- Running topic guess is configurable via `/set running_topic_guess`
- **IMPORTANT**: Topic detection has dynamic threshold - first 2 topics need 4+ messages, subsequent topics need 8+ messages
- Compressions are stored in separate tables (compressions_v2, compression_nodes) and don't pollute conversation tree
- Unit tests need isolation - currently modify user's production config file