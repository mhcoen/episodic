# Episodic Project Memory

## Testing
- **Framework**: pytest (not unittest)
- **Location**: tests/ directory  
- **Types**: Unit and integration tests
- **Command**: `python tests/run_all_tests.py`

## Recent Decisions
- Fixed unified streaming bold formatting using raw ANSI codes
- Headers (###) now display without markdown markers but remain bold
- Separated memory: Claude Desktop uses general memory, Claude Code uses project-specific

## Current Focus
- Next priority: Add previous history to /muse mode for follow-up questions (see TODO.md)

## User Preferences
- 80x24 terminal, needs proper word wrapping
- Bold formatting for numbered/bullet lists up to colon
- Prefers explicit over implicit behavior

## Architecture Notes
- Unified streaming in `episodic/unified_streaming.py`
- RAG system with ChromaDB
- Web search with DuckDuckGo provider
- Topic detection with configurable models