# Episodic Project Memory

Last Updated: 2025-07-10

## Testing
- **Framework**: pytest (not unittest)
- **Location**: tests/ directory  
- **Types**: Unit and integration tests
- **Command**: `python tests/run_all_tests.py`

## Recent Session (2025-07-10)
### Major Code Cleanup
- **Removed unused imports**: Used autoflake to clean 56+ unused imports across 41 files
- **Deleted deprecated code**: Removed `conversation_original.py`, `settings_old.py`, and no-op `close_connection()` function
- **Fixed empty exception blocks**: Added proper error logging instead of silent failures
- **Consolidated duplicate functions**: Created `debug_utils.py` to unify 3 duplicate `debug_print()` implementations
- **Created CLEANUP_SUMMARY.md**: Documented all cleanup changes

### Completed Refactoring Tasks
- ✅ **Enforced 500-600 line limit**: All active files now under 600 lines
- ✅ **Fixed compression commands**: Removed confusing unified command structure
- ✅ **Conversation.py refactoring**: Successfully split from 1,872 lines into:
  - `topic_management.py` (508 lines) - Topic detection and management
  - `response_streaming.py` (410 lines) - Streaming implementations  
  - `text_formatter.py` (385 lines) - Text formatting and wrapping
  - `context_builder.py` (226 lines) - Context preparation with RAG/web
  - `conversation.py` (545 lines) - Core conversation flow
  - `unified_streaming.py` (411 lines) - Unified streaming logic
  - Extended existing modules for web synthesis

### Visualization.py Status
- **IGNORE**: User is replacing visualization.py entirely (1,278 lines)
- Do not attempt to refactor or modify this file

## Recent Session (2025-01-09 continued)
### Embedding Model Configuration and Topic Detection
- **Issue Fixed**: BGE embedding model wasn't being used by all topic detectors
- Updated all `ConversationalDrift` instantiations to read config settings
- Fixed detectors: RealtimeWindowDetector, SlidingWindowDetector, SimpleDriftDetector, HybridTopicDetector
- **Key Finding**: BGE models produce lower drift scores (0.65-0.7) vs default model (0.9+)
- Implemented `/mset embedding` command system for easy model configuration
- Available models: paraphrase-mpnet, BGE family, GTE family, MiniLM variants
- Documented that topic naming uses same model as topic detection

### Python Library Update and Fixes (earlier)
- Fixed Python 3.13 regex syntax warning by replacing `\S` and `\s` with explicit character classes
- Fixed /set command to show short curated list with descriptions
- Changed /set and /help to use 'all' instead of '--all'
- Fixed color-mode documentation (full/basic/none, not dark/light/none)
- Removed non-functional stream-char-mode settings
- Implemented centralized cost tracking in LLMManager
- Fixed help system file indexing

### Git History Cleanup (earlier)
- **CRITICAL**: Removed all Claude/Anthropic references from entire git history
- Rewrote 328 commits using git filter-branch
- Force pushed to private remote repository
- Installed local commit-msg hook to prevent future occurrences
- Hook blocks commits with "claude" or "anthropic" (case-insensitive)

## Recent Decisions
- Fixed unified streaming bold formatting using raw ANSI codes
- Headers (###) now display without markdown markers but remain bold
- Separated memory: Claude Desktop uses general memory, Claude Code uses project-specific
- All LLM cost tracking now centralized through LLMManager
- Git commits must never mention Claude/Anthropic per CLAUDE.md
- BGE embedding models need lower thresholds (0.65-0.75) vs default (0.9)
- Topic naming uses same model as topic detection (not separately configurable)

## Current Focus
- ✅ **COMPLETED**: Enforced 500 line cap per file (all files now under 600 lines)
- ✅ **COMPLETED**: Fixed compression command structure
- ✅ **COMPLETED**: Major code cleanup (unused imports, deprecated code, duplicate functions)
- **Next priorities**:
  - Add previous history to /muse mode for follow-up questions
  - Add support for other web search providers beyond DuckDuckGo

## User Preferences
- 80x24 terminal, needs proper word wrapping
- Bold formatting for numbered/bullet lists up to colon
- Prefers explicit over implicit behavior
- No AI attribution in commits
- Prefers simple solutions over complex ones
- Wants clear, modular configuration options
- Values understanding why features work the way they do

## Code Quality Guidelines
- **File Length**: Maximum 500 lines per file (absolute cap at 600)
- Files exceeding limit must be refactored or modularized
- Keep modules focused on single responsibilities

## Architecture Notes
- Unified streaming in `episodic/unified_streaming.py`
- RAG system with ChromaDB
- Web search with DuckDuckGo provider
- Topic detection with configurable models
- Python 3.13.5 (has stricter syntax warnings)
- Embedding configuration: drift_embedding_provider, drift_embedding_model, drift_threshold
- All topic detectors now properly read embedding configuration from config