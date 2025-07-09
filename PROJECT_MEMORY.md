# Episodic Project Memory

Last Updated: 2025-01-09

## Testing
- **Framework**: pytest (not unittest)
- **Location**: tests/ directory  
- **Types**: Unit and integration tests
- **Command**: `python tests/run_all_tests.py`

## Recent Session (2025-01-09)
### Python Library Update and Fixes
- Fixed Python 3.13 regex syntax warning by replacing `\S` and `\s` with explicit character classes
- Fixed /set command to show short curated list with descriptions
- Changed /set and /help to use 'all' instead of '--all'
- Fixed color-mode documentation (full/basic/none, not dark/light/none)
- Removed non-functional stream-char-mode settings
- Implemented centralized cost tracking in LLMManager
- Fixed help system file indexing

### Git History Cleanup
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

## Current Focus
- Next priority: Add previous history to /muse mode for follow-up questions (see TODO.md)
- Fix compression command structure (confusing /compression unified command)
- Add support for other web search providers beyond DuckDuckGo

## User Preferences
- 80x24 terminal, needs proper word wrapping
- Bold formatting for numbered/bullet lists up to colon
- Prefers explicit over implicit behavior
- No AI attribution in commits
- Prefers simple solutions over complex ones

## Architecture Notes
- Unified streaming in `episodic/unified_streaming.py`
- RAG system with ChromaDB
- Web search with DuckDuckGo provider
- Topic detection with configurable models
- Python 3.13.5 (has stricter syntax warnings)