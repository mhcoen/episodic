# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Always read `PROJECT_MEMORY.md` at the start of each session for current context, recent decisions, and user preferences. Also check `TODO.md` for project todos and upcoming work.

## Project Overview

Episodic is a conversational DAG-based memory agent that creates persistent, navigable conversations with language models. It stores conversation history as a directed acyclic graph where each node represents a conversational exchange.

## Recent Major Refactoring (2025-01)

The codebase underwent significant cleanup and reorganization through 8 pull requests:

1. **PR #1: File Organization** - Moved ~40 test/analysis scripts to proper directories
2. **PR #2: Configuration Consolidation** - Created config_defaults.py, centralized all defaults
3. **PR #3: Topic Detection Module Restructure** - Created episodic.topics module with submodules
4. **PR #4: Database Schema Cleanup** - Added migrations, renamed tables, created indexes
5. **PR #5: Command Consolidation** - Created unified commands with subactions
6. **PR #6: Test Infrastructure** - Organized tests into unit/integration/fixtures
7. **PR #7: Dead Code Removal** - Removed unused imports, created deprecation tracking
8. **PR #8: Documentation Update** - Updated all docs to reflect changes

## Current Architecture

### Project Structure (Updated 2025-07-10)
```
episodic/
├── __main__.py              # Entry point
├── cli.py                   # Main CLI interface (delegated functionality)
├── cli_main.py              # Main loop and entry point
├── cli_command_router.py    # Command routing logic
├── cli_display.py           # Display and UI functions
├── cli_helpers.py           # CLI utility functions
├── cli_registry.py          # Enhanced command handling with registry
├── cli_session.py           # Session management
├── config.py                # Configuration management
├── config_defaults.py       # Centralized default values
├── configuration.py         # Configuration constants
├── conversation.py          # Core conversation flow (545 lines)
├── context_builder.py       # Context preparation with RAG/web (226 lines)
├── core.py                  # Core data structures (Node, ConversationDAG)
├── db.py                    # Database interface (imports from specialized modules)
├── db_*.py                  # Specialized database modules:
│   ├── db_connection.py     # Connection management (now defaults to ~/.episodic/)
│   ├── db_nodes.py          # Node operations
│   ├── db_topics.py         # Topic operations
│   ├── db_scoring.py        # Topic detection scoring
│   ├── db_compression.py    # Compression operations
│   ├── db_ids.py            # ID generation
│   ├── db_migrations.py     # Migration functions
│   └── db_rag.py            # RAG database operations
├── debug_utils.py           # Consolidated debug utilities
├── llm.py                   # LLM integration via LiteLLM
├── llm_manager.py           # Centralized API call tracking
├── rag.py                   # RAG system with ChromaDB (519 lines)
├── rag_utils.py             # RAG utilities and decorators
├── rag_chunker.py           # Document chunking for RAG
├── rag_document_manager.py  # Document management for RAG
├── response_streaming.py    # Streaming implementations (410 lines)
├── text_formatter.py        # Text formatting and wrapping (385 lines)
├── text_formatting.py       # Additional text utilities
├── topic_management.py      # Topic detection and management (508 lines)
├── topic_boundary_analyzer.py # Topic boundary analysis
├── unified_streaming.py     # Unified streaming logic (411 lines)
├── unified_streaming_format.py # Streaming format utilities
├── web_search.py            # Web search functionality (536 lines)
├── web_synthesis.py         # Web synthesis for muse mode
├── web_extract.py           # Web content extraction
├── commands/                # All CLI commands
│   ├── registry.py          # Command registry system
│   ├── unified_topics.py    # Unified topic management
│   ├── unified_compression.py # Unified compression management
│   ├── unified_model.py     # Model management commands
│   ├── mset.py              # Model parameter setting
│   ├── help.py              # Help system (515 lines)
│   ├── settings.py          # Settings commands
│   ├── rag.py               # RAG commands
│   ├── web_search.py        # Web search commands (524 lines)
│   └── ...                  # Other command modules
├── topics/                  # Topic detection module
│   ├── __init__.py          # Public API
│   ├── detector.py          # Main TopicManager class
│   ├── boundaries.py        # Boundary detection logic
│   ├── hybrid.py            # Hybrid detector implementation
│   ├── keywords.py          # Keyword-based detection
│   ├── windows.py           # Sliding window implementation
│   └── utils.py             # Shared utility functions
├── migrations/              # Database migrations
│   ├── __init__.py          # Migration runner
│   └── m00X_*.py            # Individual migrations
└── prompts/                 # System prompts
```

**Note**: All Python files are now under 600 lines (target: 500 lines) after major refactoring

### Recent Changes From Refactoring

#### File Organization (PR #1)
- Test scripts moved to `tests/scripts/`
- Analysis scripts moved to `scripts/analysis/`
- Defunct files moved to `defunct/` (later removed in PR #7)

#### Configuration (PR #2)
- All defaults now in `config_defaults.py`
- Organized into logical groups: CORE, TOPIC, LLM, etc.
- No more hardcoded values throughout codebase

#### Topic Detection Module (PR #3)
- Moved from single `topics.py` to organized module
- Clear separation: detector, boundaries, keywords, windows
- Public API in `__init__.py`

#### Database Schema (PR #4)
- Migration system in `migrations/`
- Table renamed: `manual_index_scores` → `topic_detection_scores`
- Added performance indexes

#### Command Consolidation (PR #5)
- Unified commands with subactions:
  - `/topics [list|rename|compress|index|scores|stats]`
  - `/compression [stats|queue|compress|api-stats|reset-api]`
- Command registry for better organization
- Deprecated old commands still work with warnings

#### Test Infrastructure (PR #6)
```
tests/
├── unit/                    # Unit tests
│   ├── topics/              # Topic-related tests
│   └── commands/            # Command tests
├── integration/             # Integration tests
├── fixtures/                # Reusable test data
│   ├── conversations.py     # Test conversations
│   └── test_utils.py        # Test utilities
└── run_all_tests.py         # Enhanced test runner
```

## Current Session Context

### Working Session (2025-07-11)
- **Fixed comprehensive CLI command breakages after refactoring**
  - Found 37 out of 60 commands were broken (38% pass rate)
  - Fixed all critical issues without user intervention:
    - Model commands: Fixed 'module' object is not callable error
    - RAG system: Fixed import errors (rag_toggle, index_file, docs_command)
    - Compression: Fixed import to use compression_command
    - Web search: Fixed function signature mismatch with websearch_command
    - Muse mode: Created missing muse.py module
    - Topic commands: Fixed get_current_topic AttributeError and Typer decorator issues
  - Added 9 missing command implementations (/h, /about, /welcome, /config, /history, /tree, /graph, /summary)
  - Result: ~100% pass rate for all critical commands
- **Created comprehensive test infrastructure**
  - `test_all_commands.py` - Tests every CLI command systematically
  - `analyze_test_results.py` - Parses test output and categorizes failures
  - `BUG_REPORT.md` - Detailed bug documentation with priorities
  - `FIXES_SUMMARY.md` - Summary of all fixes applied
  - `error_summary.md` - Checklist of all errors found
- **Fixed streaming response duplication**
  - Issue: Last word printed twice in natural rhythm mode
  - Root cause: Queued printer calling finish() which re-printed already queued words
  - Solution: Removed processor.finish() call in _queued_printer thread

### Working Session (2025-07-10)
- **Major conversation.py refactoring completed** (1,872 lines → 545 lines)
  - Split into specialized modules: topic_management.py, response_streaming.py, text_formatter.py, context_builder.py
  - Successfully enforced 500-600 line limit across all active files
  - Fixed compression command structure (removed confusing unified command)
- **Comprehensive code cleanup**
  - Removed 56+ unused imports across 41 files using autoflake
  - Deleted deprecated files: `conversation_original.py`, `settings_old.py`
  - Removed deprecated no-op `close_connection()` function
  - Fixed empty exception blocks with proper error logging
  - Created `debug_utils.py` to consolidate 3 duplicate `debug_print()` implementations
  - Created CLEANUP_SUMMARY.md documenting all changes
- **Database path fix**
  - Changed default database location from project directory to `~/.episodic/episodic.db`
  - Automatically creates ~/.episodic directory if needed
  - Prevents mixing user data with code
- **Updated memory files**
  - Updated PROJECT_MEMORY.md and CLAUDE.md to reflect all changes
- **Remaining tasks**:
  - Add previous query history to muse mode for context-aware follow-ups
  - Add support for additional web search providers beyond DuckDuckGo

### Working Session (2025-01-07 continued)
- Fixed unified streaming output formatting issues
  - Headers (###) now display without markdown markers while keeping text bold
  - Fixed typer/click ANSI code conflicts preventing bold+color combinations
  - Solution: Use raw ANSI escape codes for bold text instead of click.style
  - Bold formatting now works correctly for headers, numbered lists (1., 2.), and bullet lists (-)
  - Lists are bold up to and including the colon
- Diagnosed word wrap issues
  - Terminal width detection working correctly (80 columns)
  - Issue was hard line breaks in source text, not wrapping logic
  - Created test files to verify formatting and terminal width detection
- Added todo items for future muse enhancements:
  - Incorporate previous query history into muse mode for context-aware responses
  - Add support for additional web search providers beyond DuckDuckGo

### Working Session (2025-01-07)
- Implemented RAG (Retrieval Augmented Generation) functionality
  - Core RAG module (`episodic/rag.py`) with ChromaDB vector database integration
  - Document chunking with configurable size and overlap for better retrieval
  - Duplicate detection using SHA-256 content hashing
  - Commands: `/rag [on|off]`, `/search` (alias `/s`), `/index` (alias `/i`), `/docs [list|show|remove|clear]`
  - Automatic context enhancement for chat messages when RAG enabled
  - Graceful degradation when dependencies (chromadb, sentence-transformers) missing
  - Fixed ChromaDB telemetry errors with multiple suppression approaches
  - Created `episodic/rag_utils.py` to consolidate common patterns (@requires_rag decorator, validation utilities)
  - Fixed /index command collision (topic indexing deprecated and shows warning)
  - Improved error handling with context managers for all database operations
  - Fixed SQL injection risk by avoiding dynamic SQL construction
  - Database tables: `rag_documents` for indexed content, `rag_retrievals` for tracking usage
  - Configuration in `config_defaults.py` under `RAG_DEFAULTS`
- Implemented Web Search Integration
  - Web search module (`episodic/web_search.py`) with provider architecture
  - DuckDuckGo provider for free web search (no API key required)
  - SearchCache for result caching (1 hour default)
  - RateLimiter to prevent API abuse (60 searches/hour default)
  - Commands: `/websearch <query>` (alias `/ws`), `/websearch on/off`, `/websearch config/stats/cache clear`
  - Integration with RAG: automatic web search when local results insufficient
  - Web results can be indexed into RAG for future use
  - Configuration in `config_defaults.py` under `WEB_SEARCH_DEFAULTS`
  - Graceful degradation when dependencies (aiohttp, beautifulsoup4) missing
- Fixed color display issues
  - Root cause: Warp terminal was overriding ANSI color codes with its theme
  - Solution: Simplified color handling to use click.echo(color=True) consistently
  - Removed complex reset parameter logic that was added during debugging
  - Colors now work correctly in both standard Terminal.app and Warp with proper themes

### Previous Session (2025-01-01)
- Fixed topic message count showing 0 for ongoing topics
  - Modified count_nodes_in_topic() to use get_head() for ongoing topics
- Fixed excessive topic creation due to min_messages_before_topic_change=2
  - Updated configuration to use recommended value of 8
- Fixed topic boundary assignment bug
  - Issue: Topic boundaries were set at detection point instead of actual transition
  - Solution: Added topic_boundary_analyzer.py to find where topics actually change
  - Analyzes recent messages when topic change detected to find true transition point
  - Supports both LLM-based analysis and heuristic fallback
  - Configuration: analyze_topic_boundaries (default: True), use_llm_boundary_analysis (default: True)
- Restored dynamic threshold behavior for topic detection
  - First 2 topics: Use min_messages/2 threshold (4 when min=8)
  - Subsequent topics: Use full threshold (8)
  - Fixed regression from June 27 that removed dynamic thresholds
  - Now matches three-topics-test.txt expectations

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
- Filter out unsupported parameters (presence_penalty, frequency_penalty) for Google Gemini models

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

#### Four LLM Contexts
The system supports different models and parameters for four distinct contexts:
1. **Chat** - Main conversation model (set with `/model chat <name>`)
2. **Detection** - Topic detection model (set with `/model detection <name>`)
3. **Compression** - Compression/summarization model (set with `/model compression <name>`)
4. **Synthesis** - Web search synthesis model (set with `/model synthesis <name>`)

Each context can have independent model parameters set with `/mset`:
- `/mset chat.temperature 0.7`
- `/mset detection.temperature 0`
- `/mset compression.max_tokens 500`
- `/mset synthesis.temperature 0.3`

#### Topic Detection Flow
1. User sends message → Topic detection runs (ollama/llama3 with JSON output)
2. If topic change detected → Close previous topic at last assistant response
3. Previous topic's content is analyzed to extract appropriate name
4. New topic starts as "ongoing-TIMESTAMP" placeholder
5. After 2+ user messages, topic is automatically renamed based on content
6. Topics remain "open" (end_node_id=NULL) until closed on topic change

**Note on Topic Naming**: Both topic detection and topic name extraction use the same model configured via `/model detection <name>`. The `topic_detection_model` setting controls both:
- Detecting when topics change (JSON yes/no response)
- Extracting descriptive names from conversation content (1-3 word summary)
This means if you change the detection model, it affects both detection accuracy and naming quality.

#### Database Functions
- `store_topic()` - Creates new topic entry (end_node_id now optional)
- `update_topic_end_node()` - Closes topic by setting end boundary
- `update_topic_name()` - Renames topic
- `get_recent_topics()` - Retrieves topic list
- `migrate_topics_nullable_end()` - Migration to allow NULL end_node_id

#### Important Code Locations (Updated 2025-07-10)
- **Conversation flow**: `episodic/conversation.py` - Core conversation management (545 lines)
- **Topic management**: `episodic/topic_management.py` - Topic detection and handling (508 lines)
- **Response streaming**: `episodic/response_streaming.py` - Streaming implementations (410 lines)
- **Context building**: `episodic/context_builder.py` - Context preparation with RAG/web (226 lines)
- **Text formatting**: `episodic/text_formatter.py` - Text wrapping and formatting (385 lines)
- **Topic detection**: `episodic/topics/detector.py` - Main TopicManager class
- **Database operations**: `episodic/db_*.py` - Modularized database functions
- **Debug utilities**: `episodic/debug_utils.py` - Consolidated debug functions
- **Command registry**: `episodic/commands/registry.py` - Command management
- **Unified commands**: `episodic/commands/unified_*.py` - New command structure
- **Configuration**: `episodic/config_defaults.py` - All default values
- **LLM Manager**: `episodic/llm_manager.py` - API call tracking
- **Command routing**: `episodic/cli_command_router.py` - Command routing logic

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
- `scripts/testing/test-complex-topics.txt` - 21 queries across multiple topics
- `scripts/testing/test-topic-naming.txt` - Simple topic transitions
- `scripts/testing/test-final-topic.txt` - Tests final topic handling
- `scripts/testing/three-topics-test.txt` - Tests three topic changes accounting for threshold behavior

### Current Commands (Post-Refactoring)

#### Unified Commands
- `/topics` - Topic management with subactions:
  - `list` (default) - List all topics
  - `rename` - Rename ongoing topics
  - `compress` - Compress current topic
  - `index <n>` - Manual topic detection
  - `scores` - Show detection scores
  - `stats` - Topic statistics
- `/compression` - Compression management:
  - `stats` (default) - Show statistics
  - `queue` - Show pending jobs
  - `compress` - Manual compression
  - `api-stats` - API usage stats
  - `reset-api` - Reset API stats
#### Model and Configuration Commands
- `/model` - Show current chat model
- `/model list` - Show all models for all contexts
- `/model chat <name>` - Set chat (main conversation) model
- `/model detection <name>` - Set topic detection model (also used for topic naming)
- `/model compression <name>` - Set compression model
- `/model synthesis <name>` - Set web synthesis model
- `/mset` - Show all model parameters
- `/mset chat` - Show parameters for chat model
- `/mset chat.temperature 0.7` - Set specific parameter
- `/set <param> <value>` - Set other configuration parameters
- `/verify` - Verify configuration
- `/cost` - Show session costs
- `/config-docs` - Show configuration documentation
- `/reset` - Reset configuration to defaults

#### RAG (Knowledge Base) Commands
- `/rag [on|off]` - Enable/disable RAG or show stats
- `/search <query>` or `/s <query>` - Search the knowledge base
- `/index <file>` or `/i <file>` - Index a file into knowledge base
- `/index --text "<content>"` - Index text directly
- `/docs` - Document management with subactions:
  - `list` (default) - List all documents
  - `show <doc_id>` - Show document content
  - `remove <doc_id>` - Remove a document
  - `clear [source]` - Clear documents

#### Web Search Commands
- `/websearch <query>` or `/ws <query>` - Search the web
- `/websearch on/off` - Enable/disable web search
- `/websearch config` - Show web search configuration
- `/websearch stats` - Show search statistics
- `/websearch cache clear` - Clear search cache

#### Deprecated Commands (still work with warnings)
- `/rename-topics` → `/topics rename`
- `/compress-current-topic` → `/topics compress`
- `/api-stats` → `/compression api-stats`
- `/reset-api-stats` → `/compression reset-api`

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
# Start the main CLI interface (interactive mode)
python -m episodic

# Execute a script non-interactively
python -m episodic --execute scripts/test-script.txt
python -m episodic -e scripts/test-script.txt

# Within the CLI, initialize database
> /init

# Start visualization server
> /visualize
```

#### Testing
```bash
# Run all tests with new test runner
python tests/run_all_tests.py all

# Run specific test categories
python tests/run_all_tests.py unit        # Unit tests only
python tests/run_all_tests.py integration # Integration tests
python tests/run_all_tests.py quick       # Quick stable tests
python tests/run_all_tests.py topics      # Topic-related tests
python tests/run_all_tests.py coverage    # With coverage report

# Run specific test files
python -m unittest tests.unit.topics.test_topic_detection -v
python -m unittest tests.unit.commands.test_unified_commands -v

# Using pytest (if installed)
pytest tests/ -v
pytest tests/unit -m "not slow"
```

### Architecture

#### Core Components
- **Node/ConversationDAG** (`core.py`): Core data structures representing conversation nodes and the DAG
- **Database Layer** (`db.py`): SQLite-based persistence with thread-safe connection handling
- **LLM Integration** (`llm.py`): Multi-provider LLM interface using LiteLLM with context caching
- **CLI Interface** (`cli.py`): Typer-based command-line interface with talk-first design
- **Visualization** (`visualization.py`): NetworkX and Plotly-based graph visualization with real-time updates
- **Configuration** (`config.py`): Application configuration management
- **RAG System** (`rag.py`): Retrieval Augmented Generation with ChromaDB vector database
- **RAG Utilities** (`rag_utils.py`): Common patterns and utilities for RAG functionality
- **Web Search** (`web_search.py`): Web search integration with provider abstraction
- **Web Synthesis** (`web_synthesis.py`): Muse mode for Perplexity-like web search synthesis
- **Unified Streaming** (`unified_streaming.py`): Centralized streaming output with markdown formatting

#### Key Design Patterns
- **Thread-safe database operations**: Uses thread-local connections and context managers
- **Provider-agnostic LLM calls**: Abstracts different LLM providers (OpenAI, Anthropic, Ollama, etc.) through LiteLLM
- **Short node IDs**: Human-readable 2-character IDs for easy navigation
- **Linear conversation structure**: Currently all conversations are completely linear - the DAG is a straight line that is never modified (no branching implemented yet)
- **Real-time visualization**: HTTP polling for live graph updates

#### Database Schema
- **Default location**: `~/.episodic/episodic.db` (changed from project directory)
- **nodes**: Conversation nodes with id, short_id, message, response, etc.
- **topics**: Topic tracking with nullable end_node_id for ongoing topics
- **topic_detection_scores**: Detection scores and metadata (renamed from manual_index_scores)
- **compressions_v2**: Compression summaries
- **compression_nodes**: Mapping of compressed nodes
- **configuration**: Key-value configuration storage
- **conversations**: Conversation metadata
- **migration_history**: Applied migrations tracking
- **rag_documents**: Indexed documents with content hash for duplicate detection
- **rag_retrievals**: Tracks which documents were used in responses
- Indexes on frequently queried columns for performance
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

### Development Guidelines

#### Code Organization
- **File size limit**: Maximum 500 lines per file (absolute cap at 600)
- Keep imports organized and remove unused ones (use `scripts/cleanup_unused_imports.py`)
- Follow the module structure established in the refactoring
- Use the command registry for new commands
- Add new defaults to `config_defaults.py`, not hardcoded
- Use `debug_utils.py` for debug output instead of creating new debug functions

#### Adding New Features
1. For new commands: Add to appropriate unified command or create new one
2. For topic detection: Add to `episodic/topics/` module
3. For database changes: Create a migration in `episodic/migrations/`
4. For tests: Add to appropriate directory in `tests/`
5. Check file size after additions - refactor if approaching 500 lines

#### Deprecation Process
1. Mark old code as deprecated in command registry
2. Add to `DEPRECATED.md` with removal timeline
3. Show warning when deprecated feature is used
4. Remove in specified future version

#### Database Location
- Default: `~/.episodic/episodic.db` (NOT in project directory)
- Override with `EPISODIC_DB_PATH` environment variable
- Directory is created automatically if it doesn't exist

### Future Development Plans

#### Completed: Codebase Cleanup ✅
- All 8 PRs from `CLEANUP_PLAN.md` have been completed
- Codebase is now well-organized and maintainable

#### Next: Adaptive Topic Detection
- **Adaptive Topic Detection**: See `ADAPTIVE_TOPIC_DETECTION_PLAN.md` for detailed implementation plan
  - Enable dynamic context management when users return to previous topics
  - Support non-linear DAG-based conversations with branching
  - Implement with graceful failure and progressive enhancement
  - Phase 1: Foundation and embedding infrastructure
  - Phase 2: Passive suggestions for topic similarity
  - Phase 3: Smart context inclusion
  - Phase 4: Topic return detection
  - Phase 5: Progressive DAG branching
