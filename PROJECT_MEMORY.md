# Episodic Project Memory

## Project Overview

Episodic is a conversational DAG-based memory agent that creates persistent, navigable conversations with language models. It automatically organizes conversations into topics and provides advanced features like RAG, web search, and multi-model support.

## Architecture Highlights

- **Modular Design**: Split into focused modules under 600 lines each
- **Database**: SQLite with migration system, default location `~/.episodic/episodic.db`
- **LLM Integration**: Unified interface via LiteLLM supporting 20+ providers
- **Model Configuration**: JSON-based model definitions in `~/.episodic/models.json`
- **Topic Detection**: Multiple algorithms including sliding window and hybrid detection
- **RAG System**: Vector database using ChromaDB for document similarity search
- **Web Search**: Pluggable provider system with automatic fallback

## Key Components

### Core Modules
- `conversation.py` - Core conversation management (545 lines)
- `topic_management.py` - Topic detection and handling (508 lines)
- `response_streaming.py` - Streaming implementations (410 lines)
- `context_builder.py` - Context preparation with RAG/web (226 lines)
- `text_formatter.py` - Text formatting and wrapping (385 lines)
- `unified_streaming.py` - Centralized streaming output (411 lines)
- `model_config.py` - Model configuration loader (manages models.json)

### Database Modules
- `db_connection.py` - Connection management
- `db_nodes.py` - Node operations
- `db_topics.py` - Topic operations
- `db_scoring.py` - Topic detection scoring
- `db_compression.py` - Compression operations
- `db_rag.py` - RAG database operations

### Memory System
- **System Memory** (always on by default):
  - Conversations automatically stored as searchable memories
  - Smart context detection for relevant past conversations
  - Independent of user RAG settings
- **User RAG** (optional):
  - Index your own documents with `/index`
  - Controlled by `/set rag-enabled`
- **Memory Commands**: `/memory`, `/forget`, `/memory-stats` work regardless of RAG setting
- **Preview Support**: Shows content previews in memory listings

### Command System
- Unified commands with subactions (e.g., `/topics list|rename|compress`)
- Command registry for better organization
- Tab completion support with context-aware suggestions

## Testing

- **Framework**: pytest
- **Test Runner**: `python tests/run_all_tests.py`
- **Categories**: unit, integration, quick, topics, coverage
- **CLI Testing**: Comprehensive command validation suite

## Development Guidelines

- **File Length**: Maximum 500 lines per file (hard cap at 600)
- **Code Organization**: Follow established module structure
- **Commands**: Use command registry for new commands
- **Configuration**: Add defaults to `config_defaults.py`
- **Model Configuration**: Models defined in `~/.episodic/models.json` (auto-created from template)
- **Database Changes**: Create migrations in `episodic/migrations/`
- **Debug Output**: Use `debug_utils.py` for debug functions

## Recent Changes

- **Memory Collection Separation (January 2025)**:
  - Implemented multi-collection RAG architecture
  - Created separate collections for conversation memories and user documents
  - Added `/migrate` command for user-controlled migration
  - Maintains full backward compatibility with automatic detection
  - See `MEMORY_COLLECTION_SEPARATION.md` for implementation details
- **Model Configuration System**: Replaced hardcoded models with JSON configuration
  - Groq provider replaced with Google (Gemini models)
  - Models now loaded from `~/.episodic/models.json`
  - Support for user customization and overrides
  - Fixed OpenRouter Claude 4 Opus model ID to `openrouter/anthropic/claude-opus-4`
- **Memory System Improvements**: System memory always on, independent of user RAG
- **Config Persistence**: Only specific settings saved (via `save_setting()`)
- **Assistant Message Limits**: Documented 5-hour reset blocks with start times rounded down to nearest hour

## Future Development

- **Adaptive Topic Detection**: Dynamic context management for non-linear conversations
- **DAG Branching**: Support for conversation trees and topic returns
- **Enhanced Embeddings**: Multiple embedding providers for different use cases
- **Running Topic Prediction**: Real-time topic detection during conversation