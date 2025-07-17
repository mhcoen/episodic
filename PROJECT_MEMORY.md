# Episodic Project Memory

## Project Overview

Episodic is a conversational DAG-based memory agent that creates persistent, navigable conversations with language models. It automatically organizes conversations into topics and provides advanced features like RAG, web search, and multi-model support.

## Architecture Highlights

- **Modular Design**: Split into focused modules under 600 lines each
- **Database**: SQLite with migration system, default location `~/.episodic/episodic.db`
- **LLM Integration**: Unified interface via LiteLLM supporting 20+ providers
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

### Database Modules
- `db_connection.py` - Connection management
- `db_nodes.py` - Node operations
- `db_topics.py` - Topic operations
- `db_scoring.py` - Topic detection scoring
- `db_compression.py` - Compression operations
- `db_rag.py` - RAG database operations

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
- **Database Changes**: Create migrations in `episodic/migrations/`
- **Debug Output**: Use `debug_utils.py` for debug functions

## Future Development

- **Adaptive Topic Detection**: Dynamic context management for non-linear conversations
- **DAG Branching**: Support for conversation trees and topic returns
- **Enhanced Embeddings**: Multiple embedding providers for different use cases
- **Running Topic Prediction**: Real-time topic detection during conversation