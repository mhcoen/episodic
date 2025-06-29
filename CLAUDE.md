# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Episodic is a conversational DAG-based memory agent that creates persistent, navigable conversations with language models. It stores conversation history as a directed acyclic graph where each node represents a conversational exchange.

## Current Session Context

### Last Working Session (2025-06-29)
- Fixed /rename-topics command to handle ongoing topics (topics with NULL end_node_id)
- Fixed finalize_current_topic() to properly rename ongoing topics when conversation ends
- Root cause: Both functions were trying to get_ancestry(NULL) which returns empty
- Solution: Check if end_node_id is NULL and use get_head() instead
- Fixed bold formatting for numbered lists in streaming output
- Now bolds only the first line of each numbered item (e.g., "**1. Life Support Systems: description here**")
- Continuation lines under the same item are not bolded
- Fixed Google Gemini model configuration to use "gemini/" prefix for Google AI Studio
- Added GOOGLE_API_KEY to provider API keys mapping

### Previous Session (2025-06-28)
- Fixed JSON parsing errors in topic detection for Ollama models
- Added robust fallback parsing for various response formats (Yes/No/JSON)
- Created simplified topic_detection_ollama.md prompt for better compatibility
- Topic detection now handles malformed JSON responses gracefully
- Fixed critical `stop: ["\n"]` parameter causing GPT-3.5 to return truncated responses
- Created topic_detection_v3.md prompt for domain-agnostic detection
- Discovered GPT-3.5 is over-sensitive (6-7 topics) while Ollama is under-sensitive (1 topic)
- Created comprehensive test suite in scripts/topic/ for validating topic detection
- Verified Ollama topic detection IS working but being too conservative

#### Current Issue
Topic detection sensitivity varies drastically by model:
- **GPT-3.5**: Creates too many topics (splits related concepts like "pasta recipes" vs "Italian pantry")
- **Ollama**: Creates too few topics (keeps everything together, even explicit transitions)
- **Target**: 3 topics for the standard test (Mars, Italian cooking, Neural networks)

#### Test Results Summary
| Test | Expected | GPT-3.5 | Ollama |
|------|----------|---------|--------|
| Python progression | 1 | 4 ❌ | 1 ✅ |
| Explicit transitions | 4 | 6 ❌ | 1 ❌ |
| ML deep dive | 1 | 4 ❌ | 1 ✅ |
| Natural flow | 3 | 4 ❌ | 1 ❌ |

### Previous Session (2025-06-27)
- Created centralized LLM manager for accurate API call tracking
- Fixed initial topic extraction to require minimum 3 user messages
- Added /api-stats and /reset-api-stats commands
- Fixed benchmark system operation-specific counting (no longer shows cumulative)
- Fixed streaming response cost calculation (was showing $0.00)
- Fixed topic detection to count user messages only, not total nodes
- Added validation to prevent premature topic creation
- Fixed multiple indentation errors in conversation.py

### Previous Session (2025-06-25)
- Fixed streaming output duplication in constant-rate mode
- Improved word wrapping and list indentation
- Added markdown bold (**text**) support
- Cleaned up CLI code and removed unused imports
- Fixed test suite issues (cache tests, config initialization)
- Simplified testing approach - removed over-engineered test infrastructure
- Updated documentation to reflect simplified testing

### Key System Understanding

#### Topic Detection Flow
1. User sends message → Topic detection runs (ollama/llama3 with JSON output)
2. If topic change detected → Close previous topic at last assistant response
3. Previous topic's content is analyzed to extract appropriate name
4. New topic starts as "ongoing-TIMESTAMP" placeholder
5. After 2+ user messages, topic is automatically renamed based on content
6. Topics remain "open" (end_node_id=NULL) until closed on topic change

#### Database Functions
- `store_topic()` - Creates new topic entry (end_node_id now optional)
- `update_topic_end_node()` - Closes topic by setting end boundary
- `update_topic_name()` - Renames topic
- `get_recent_topics()` - Retrieves topic list
- `migrate_topics_nullable_end()` - Migration to allow NULL end_node_id

#### Important Code Locations
- Topic detection: `episodic/topics.py:detect_topic_change_separately()`
- Topic threshold behavior: `episodic/topics.py:_should_check_for_topic_change()` (lines 75-89)
- Topic user message counting: `episodic/topics.py:count_user_messages_in_topic()` (NEW)
- Topic naming: `episodic/conversation.py:387-442` (in handle_chat_message)
- Topic creation validation: `episodic/conversation.py:876-903` (NEW validation logic)
- LLM Manager: `episodic/llm_manager.py` (centralized API call tracking)
- Compression storage: `episodic/db_compression.py` (new compression mapping system)
- Summary command: `episodic/commands/summary.py`
- Command parsing: `episodic/cli.py:handle_command()`

### Configuration Options
- `topic_detection_model` - Default: ollama/llama3
- `running_topic_guess` - Default: True (not yet implemented)
- `min_messages_before_topic_change` - Default: 8
- `show_topics` - Shows topic evolution in responses
- `debug` - Shows detailed topic detection info
- `main_params` - Model parameters for main conversation
- `topic_params` - Model parameters for topic detection (e.g., temperature=0)
- `compression_params` - Model parameters for compression
- Model params support: temperature, max_tokens, top_p, presence_penalty, frequency_penalty

### Recent Discoveries
- **IMPORTANT**: All conversations are currently completely linear - the DAG is a straight line that is never modified. There is no branching implemented yet.
- **CRITICAL**: Topic detection has undocumented threshold behavior - first 2 topics use half threshold (4 messages), then full threshold (8 messages) applies
- **FIXED**: Topics now properly include all their messages (was missing messages due to premature end_node_id setting)
- **FIXED**: Topic detection now uses JSON output format for consistency
- **FIXED**: Topics automatically rename from "ongoing-XXXX" after 2 user messages
- Compression system stores summaries separately in compressions_v2 and compression_nodes tables
- `/init --erase` properly resets conversation manager state (current_node_id, current_topic, session costs)
- ConversationManager tracks current topic with `set_current_topic()` and `get_current_topic()`
- Topics must remain "open" (end_node_id=NULL) until explicitly closed
- `get_ancestry()` returns nodes from oldest to newest (root to current)
- Topic extraction looks at beginning of conversation for better topic names
- Model parameters can be configured per context (main, topic, compression)

### Test Scripts
- `scripts/test-complex-topics.txt` - 21 queries across multiple topics
- `scripts/test-topic-naming.txt` - Simple topic transitions
- `scripts/test-final-topic.txt` - Tests final topic handling
- `scripts/three-topics-test.txt` - Tests three topic changes accounting for threshold behavior

### New Commands
- `/rename-topics` - Renames all placeholder "ongoing-*" topics by analyzing their content
- `/api-stats` - Shows actual LLM API call statistics
- `/reset-api-stats` - Resets API call counter
- `/compress <topic-name>` - Manually trigger compression for a specific topic
- `/model-params` or `/mp` - Show/set model parameters for different contexts

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
- **Linear conversation structure**: Currently all conversations are completely linear - the DAG is a straight line that is never modified (no branching implemented yet)
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