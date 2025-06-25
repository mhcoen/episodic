# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Episodic is a conversational DAG-based memory agent that creates persistent, navigable conversations with language models. It stores conversation history as a directed acyclic graph where each node represents a conversational exchange.

## Current Session Context

### Last Working Session (2025-06-25)
- Fixed streaming output duplication in constant-rate mode
- Improved word wrapping and list indentation
- Added markdown bold (**text**) support
- Cleaned up CLI code and removed unused imports
- Fixed test suite issues (cache tests, config initialization)
- Simplified testing approach - removed over-engineered test infrastructure
- Updated documentation to reflect simplified testing

### Key System Understanding

#### Topic Detection Flow
1. User sends message → Topic detection runs (ollama/llama3)
2. If topic change detected → Close previous topic with proper name
3. Previous topic's content is analyzed to extract appropriate name
4. New topic starts as "ongoing-discussion" until it too is closed

#### Database Functions
- `store_topic()` - Creates new topic entry
- `update_topic_end_node()` - Extends topic boundary
- `update_topic_name()` - Renames topic (newly added)
- `get_recent_topics()` - Retrieves topic list

#### Important Code Locations
- Topic detection: `episodic/topics.py:detect_topic_change_separately()`
- Topic naming: `episodic/conversation.py:387-442` (in handle_chat_message)
- Summary command: `episodic/cli.py:1593-1701`
- Command parsing: `episodic/cli.py:2039-2056`

### Configuration Options
- `topic_detection_model` - Default: ollama/llama3
- `running_topic_guess` - Default: True (not yet implemented)
- `min_messages_before_topic_change` - Default: 8
- `show_topics` - Shows topic evolution in responses
- `debug` - Shows detailed topic detection info

### Recent Discoveries
- Topic boundary issues occur when nodes branch (non-linear history)
- The `--` prefix in topic names (like "--space") comes from the prompt response
- First topic creation has timing issues - may not trigger properly
- `get_ancestry()` returns nodes in reverse chronological order

### Test Scripts
- `scripts/test-complex-topics.txt` - 21 queries across multiple topics
- `scripts/test-topic-naming.txt` - Simple topic transitions
- `scripts/test-final-topic.txt` - Tests final topic handling

### Common Development Commands

#### Installation & Setup
```bash
# Install in development mode
pip install -e .

# Install required dependencies
pip install typer  # Required for CLI functionality
```

#### Running the Application
```bash
# Start the main CLI interface
python -m episodic

# Within the CLI, initialize database
> /init

# Start visualization server
> /visualize
```

#### Testing
```bash
# Run all unit tests
python -m unittest discover episodic "test_*.py"

# Run specific test modules
python -m unittest episodic.test_core
python -m unittest episodic.test_db
python -m unittest episodic.test_integration

# Run interactive/manual tests
python -m episodic.test_interactive_features

# Test coverage
pip install coverage
coverage run -m unittest discover episodic "test_*.py"
coverage report -m
```

### Architecture

#### Core Components
- **Node/ConversationDAG** (`core.py`): Core data structures representing conversation nodes and the DAG
- **Database Layer** (`db.py`): SQLite-based persistence with thread-safe connection handling
- **LLM Integration** (`llm.py`): Multi-provider LLM interface using LiteLLM with context caching
- **CLI Interface** (`cli.py`): Typer-based command-line interface with talk-first design
- **Visualization** (`visualization.py`): NetworkX and Plotly-based graph visualization with real-time updates
- **Configuration** (`config.py`): Application configuration management

#### Key Design Patterns
- **Thread-safe database operations**: Uses thread-local connections and context managers
- **Provider-agnostic LLM calls**: Abstracts different LLM providers (OpenAI, Anthropic, Ollama, etc.) through LiteLLM
- **Short node IDs**: Human-readable 2-character IDs for easy navigation
- **Branching conversations**: DAG structure allows conversation branching and merging
- **Real-time visualization**: HTTP polling for live graph updates

#### Database Schema
- Nodes table with id, short_id, message, timestamp, parent_id, model_name, system_prompt, response
- SQLite with full-text search capabilities
- Configurable database path via EPISODIC_DB_PATH environment variable

#### LLM Integration Details
- Prompt caching enabled by default for performance (using LiteLLM prompt caching)
- Cost tracking for token usage with cache discount calculations
- Multiple provider support via LiteLLM
- Model selection via numbered list or direct specification
- Configurable context depth for conversation history

### Development Notes
- Entry point is `episodic/__main__.py` which delegates to `cli.py`
- Tests include both automated unit tests and interactive manual tests
- HTTP polling-based real-time functionality verification
- Prompt management system with role-based prompts in `prompts/` directory
- Configuration stored in episodic.db alongside conversation data