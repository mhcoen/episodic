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
- **CLI Interface** (`cli_*.py`): Typer-based command-line interface
- **RAG System** (`rag.py`): Vector search with ChromaDB for knowledge base
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

Remember: The codebase is designed to be clean, modular, and maintainable. When in doubt, follow the existing patterns.