# CLAUDE.md

This file provides guidance for AI assistants when working with code in this repository.

**IMPORTANT**: Always read `PROJECT_MEMORY.md` at the start of each session for current context, recent decisions, and user preferences.

**DEBUGGING**: See `DEBUG.md` for comprehensive debugging guide, including debug flags, troubleshooting steps, and development utilities.

## Project Overview

Episodic is a conversational DAG-based memory agent that creates persistent, navigable conversations with language models. It stores conversation history as a directed acyclic graph where each node represents a conversational exchange.

## Architecture

### Core Components
- **Node/ConversationDAG** (`core.py`): Core data structures representing conversation nodes and the DAG
- **Database Layer** (`db_*.py`): Modularized SQLite persistence with specialized modules
- **LLM Integration** (`llm.py`): Multi-provider interface using LiteLLM
- **Topic Detection** (`topics/`): Modular topic detection with multiple strategies
  - Default: Dual-window detector using (4,1) and (4,2) windows
  - Alternative: Sliding window, hybrid, LLM-based detectors
- **CLI Interface** (`cli_*.py`): Typer-based command-line interface
- **Memory System**: Two-part system for intelligent context
  - **System Memory**: Always-on conversation storage (like help system)
  - **User RAG** (`rag.py`): Optional vector search for user's indexed documents
- **Web Search** (`web_search.py`): Multi-provider web search with synthesis

### Key Design Principles
- **Modular Architecture**: Each file under 600 lines (target: 500)
- **Talk-First Interface**: Natural conversation is the primary interaction mode
- **Provider Agnostic**: Works with 20+ LLM providers through LiteLLM
- **Context Aware**: Automatic topic detection and compression
- **Extensible**: Plugin-friendly architecture for new features

## Development Guidelines

### Code Style
- Maximum 500 lines per file (hard limit: 600)
- Use type hints for all function signatures
- Follow existing patterns in the codebase
- Keep imports organized and minimal
- No comments unless explicitly requested
- Prefer editing existing files over creating new ones

### Database
- Default location: `~/.episodic/episodic.db`
- Use migrations for schema changes (`migrations/`)
- All database operations through `db_*.py` modules

### Testing
- Run tests with: `python tests/run_all_tests.py`
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Test new features with appropriate coverage

### Commands
- New commands go in `episodic/commands/`
- Use the command registry for registration
- Follow unified command pattern for subcommands

## Important Notes

- **No hardcoded values**: Use `config_defaults.py` for defaults
- **User data location**: `~/.episodic/` not the project directory
- **Streaming by default**: All LLM responses stream unless disabled
- **Cost tracking**: Built-in token usage tracking for all providers
- **Memory System**: System memory is always on by default (separate from user RAG)
- **Config Persistence**: Only specific settings are saved to disk (use `config.save_setting()`)
- **Model Configuration**: Models are defined in `~/.episodic/models.json` (created from template on first run)
- **Model Pricing**: Update with `scripts/update_model_pricing.py` - never guess pricing
- **Assistant Message Limits**: Assistant message limits reset in 5-hour blocks, with start times rounded down to the nearest hour
- **Topic Detection**: Topics must be properly closed when new ones start - check database for open topics, not just memory state
- **Topic Names**: New topics should use detected names immediately, not placeholder names
- **Dual-Window Detection**: Default topic detection uses (4,1) + (4,2) windows for optimal accuracy
  - High precision (4,1) runs first, safety net (4,2) only runs if needed
  - Debug with `/debug on topic` to see detection details
- **API Keys**: Load from environment or config, never hardcode
- **File Size Violations**: Several files exceed 600 lines and need refactoring

## Common Tasks

### Adding a New Command
1. Create command file in `episodic/commands/`
2. Register in `episodic/commands/registry.py`
3. Add routing in `episodic/cli_command_router.py`
4. Update help documentation

### Adding a New LLM Provider
1. Update `episodic/llm_config.py` with provider details
2. Add any special parameter handling in `llm.py`
3. Test with multiple model types (chat, instruct, etc.)

### Debugging Issues
1. Enable debug mode: `/set debug true`
2. Check `DEBUG.md` for specific debugging guides
3. Use `debug_utils.py` for debug output
4. Review logs for detailed error traces

## Current State & Known Issues

### Performance
- **Startup Time**: Heavy imports (ChromaDB, sentence-transformers) loaded eagerly
- **Lazy Loading**: Partially implemented for LiteLLM, needs expansion
- **Caching**: Limited use of @lru_cache decorators

### Code Organization  
- **Large Files Needing Split**:
  - `visualization.py` (1278 lines)
  - `cli_command_router.py` (939 lines)
  - `conversation.py` (787 lines)
  - `web_search.py` (905 lines)

### Testing
- **Test Files**: 39 test files in `tests/` directory
- **Coverage**: Core functionality tested but edge cases need work
- **Run Tests**: `python tests/run_all_tests.py`

Remember: The codebase is designed to be clean, modular, and maintainable. When in doubt, follow the existing patterns.