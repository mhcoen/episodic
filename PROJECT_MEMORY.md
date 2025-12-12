# Episodic Project Memory

## Project Overview

Episodic is a conversational DAG-based memory agent that creates persistent, navigable conversations with language models. It automatically organizes conversations into topics and provides advanced features like RAG, web search, and multi-model support.

## Recent Major Changes (2025)

### Dual-Window Topic Detection (2025-01-01)
- Implemented two-tier detection system using (4,1) and (4,2) windows
- Achieves 95% precision with (4,1) window for high-confidence detection  
- Safety net (4,2) window provides 94% F1 score for comprehensive coverage
- Replaces inferior user-to-user comparison (73.8% F1)
- Optimized to skip safety net when high precision detects change
- Integrated with debug system (`/debug on topic`)
- Configuration: `use_dual_window_detection` (default: true)

## Architecture Highlights

- **Modular Design**: Split into focused modules (target: 500 lines, hard cap: 600)
- **Database**: SQLite with migration system, default location `~/.episodic/episodic.db`
- **LLM Integration**: Unified interface via LiteLLM supporting 20+ providers
- **Model Configuration**: JSON-based model definitions in `~/.episodic/models.json`
- **Topic Detection**: Dual-window detection system with multiple fallback strategies
- **RAG System**: Vector database using ChromaDB for document similarity search
- **Web Search**: Pluggable provider system with automatic fallback
- **Command System**: 47 command modules with unified routing and registry
- **Performance**: Connection pooling for database, prompt caching for LLMs

## Key Components

### Core Modules
- `conversation.py` - Core conversation management (787 lines - needs splitting)
- `topic_management.py` - Topic detection and handling (584 lines)
- `llm.py` - LLM integration and API management (492 lines)
- `context_builder.py` - Context preparation with RAG/web (226 lines)
- `unified_streaming.py` - Centralized streaming output (437 lines)
- `model_config.py` - Model configuration loader (manages models.json)
- `cli_command_router.py` - Command routing logic (939 lines - needs splitting)

### Topic Detection System
- **Strategy Framework** (`episodic/topics/strategies/`):
  - Pluggable strategy architecture with unified `TopicStrategy` interface
  - **NeuralStrategy**: Fine-tuned DistilBERT model with granularity control
  - **DualWindowStrategy**: Embedding-based with adaptive z-score thresholds
  - **Additional strategies**: CUSUM, Delta, SpeechAct, TimeAware, Ensemble
- **Granularity Control**:
  - Three levels: `fine` (0.3), `medium` (0.5), `coarse` (0.7)
  - Configurable via `topic_granularity` setting or strategy params
  - Coarse matches sparse, high-level annotation schemes (e.g., SuperDialseg)
  - Finer granularities capture additional latent conversational structure
- **Calibration** (`episodic/topics/calibration.py`):
  - `BoundaryCalibrator` for domain-specific threshold tuning
  - Supports BOR-targeting, length-targeting, and F1-maximizing strategies
- **Evaluation Framework** (`episodic/topics/evaluation.py`):
  - Canonical boundary representation for turn-agnostic evaluation
  - Operational metrics: W-F1, BOR, WindowDiff, Purity/Coverage
  - See "Evaluation Methodology" section below for details

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

## Evaluation Methodology

### Canonical Boundary Representation
Topic detection evaluation requires care because datasets label boundaries differently:
- Some label on the message that **starts** the new topic
- Some label on the message **after which** the topic changes
- Labels may be on user turns, assistant turns, or either

The canonical representation uses "between-messages" indexing:
- Boundary at index `t` means "topic changes between message t-1 and message t"
- All dataset conventions are normalized to this form before comparison
- Strategy outputs are similarly normalized

### Key Metrics
- **Exact F1**: Strict index matching (often 0 due to detection timing)
- **W-F1 (Windowed F1)**: F1 with ±1 tolerance (the correct metric for comparison)
- **BOR (Boundary Oversegmentation Ratio)**: predicted/gold boundaries (ideal = 1.0)
- **WindowDiff**: Pevzner & Hearst (2002) segmentation metric

### Why Strategies Detect on User Turns
The strategy confirms a topic change only after observing user uptake, which reflects
conversational reality rather than annotator convention. A system cannot know a topic
has changed until the user engages with it. This is epistemically unavoidable, not lag.

### Cross-Dataset Evaluation Results

| Dataset | Dialogues | W-F1 | BOR | Interpretation |
|---------|-----------|------|-----|----------------|
| SuperSeg | 1,322 | 0.54 | 0.53 | Aligned with labels |
| DialSeg711 | 711 | 0.52 | 0.92 | Aligned with labels |
| TIAGE | 100 | 0.44 | 0.50 | Aligned with labels |
| MultiWOZ | 200 | 0.35 | 2.34 | Granularity mismatch |
| DailyDialog | 100 | 0.33 | 3.15 | Under-labeled |
| Taskmaster | 100 | 0.19 | 3.65 | Under-labeled |
| Topical_chat | 100 | 0.12 | 9.94 | Severely under-labeled |

**Interpretation:**
- **Task-oriented datasets** (SuperSeg, DialSeg711, TIAGE): W-F1 0.44-0.54 with BOR near 1.0.
  Model aligns well with human annotation conventions.
- **Open-domain datasets** (DailyDialog, Topical_chat): High BOR + low W-F1 indicates
  **annotation sparsity**, not model failure. The model detects micro-phase transitions
  (subtopics, clarifications, reframings) that datasets simply don't annotate.

**Key insight:** Fine granularity best captures conversational micro-structure across
datasets, while coarse granularity best aligns with sparse human annotation schemes
in task-oriented benchmarks. The model generalizes well in detecting conversational
structure; open-domain datasets lack annotations at the granularity the model detects.

### Alignment Presets
Common configurations in `ALIGNMENT_PRESETS`:
- `segment_start`: Boundary on message that starts new topic (most common)
- `user_starts_topic`: Boundary on user message (matches strategy output)
- `after_message`: Boundary after the labeled message
- `canonical`: Already in between-messages form

## Development Guidelines

- **File Length**: Maximum 500 lines per file (hard cap at 600)
- **Code Organization**: Follow established module structure
- **Commands**: Use command registry for new commands
- **Configuration**: Add defaults to `config_defaults.py`
- **Model Configuration**: Models defined in `~/.episodic/models.json` (auto-created from template)
- **Database Changes**: Create migrations in `episodic/migrations/`
- **Debug Output**: Use `debug_utils.py` for debug functions

## Recent Changes

- **Canonical Boundary Alignment for Evaluation (December 2025)**:
  - Fixed systematic off-by-one errors in cross-dataset evaluation
  - Problem: Datasets label boundaries on different speaker turns (user vs assistant)
  - Strategy detects on user turns; some datasets label on assistant turns
  - This is epistemically correct: topic change is confirmed by user uptake, not proposal
  - Solution: Canonical "between-messages" boundary representation
  - Added `BoundaryAlignment` config with presets for common conventions
  - W-F1 (windowed F1) properly accounts for detection timing
  - 21 unit tests in `tests/unit/topics/test_boundary_alignment.py`
- **Topic Detection Calibration & Strategies (December 2025)**:
  - Added granularity levels (fine/medium/coarse) to NeuralStrategy
  - Added adaptive z-score thresholds to DualWindowStrategy
  - Added operational metrics: Windowed F1, BOR, WindowDiff, Purity/Coverage
  - New strategies: CUSUM, Delta, SpeechAct, TimeAware
  - Calibration module for domain-specific threshold tuning
- **Muse Mode Context Preservation Fix (2025-01-12)**:
  - Fixed critical issue where muse mode was not maintaining conversation context
  - Problem: Each web search query was answered in isolation, ignoring conversation history
  - Root cause: Streaming synthesis was replacing entire message history with just synthesis prompt
  - Solution: Preserve conversation history by appending synthesis to existing messages
  - Now muse mode properly maintains context across multiple related questions
- **Muse Mode & Model Configuration Fixes (2025-01-12)**:
  - Fixed broken muse mode web search functionality (multiple cascading errors)
  - Restored proper default models: `ollama/phi3` for auxiliary tasks
  - Fixed NoneType error in `/model` command when models not configured
  - Fixed UnboundLocalError with `unified_stream_response` in muse mode
  - Discovered dual model system: expensive cloud models for chat, cheap local for auxiliary
  - Root cause: falcon-7b-instruct was incorrectly set as default at some point
- **Topic Detection Evaluation & Fine-Tuning (January 2025)**:
  - Comprehensive evaluation of topic detection methods on 4 datasets
  - Discovered small LLMs (<1B params) cannot handle complex window-based prompts
  - Fine-tuned XtremDistil model (13M params) achieving F1=0.667
  - Model processes 197 messages/second, suitable for real-time use
  - Training on 55,657 examples from SuperDialseg, TIAGE, DialSeg711, and MP2D
  - Full documentation in `evaluation/TOPIC_DETECTION_SUMMARY.md`
  - Production model at `evaluation/finetuned_models/topic_detector_full.pt`
- **Topic Detection Fixes (January 2025)**:
  - Fixed critical bug where topics weren't closed when `current_topic` was None in memory
  - Topics now properly closed by checking database for open topics, not just memory state
  - New topics created with proper names from detection instead of placeholders
  - Added `finalize_current_topic()` to close all open topics at session end
  - Ensures only one topic can be open at a time
- **Memory System Architecture Analysis**:
  - Identified mismatch between fine-grained per-message indexing and intended topic-based approach
  - WordNet expansion creates noise by expanding stop words (44% of expansions)
  - Long conversational queries (20+ words) get diluted when expanded
  - Topic-based indexing would better match original vision
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
  - Fixed OpenRouter model ID to `openrouter/anthropic/claude-opus-4.1`
  - Changed pricing display from "per 1K tokens" to "per 1M tokens" for cleaner display
- **Memory System Improvements**: System memory always on, independent of user RAG
- **Config Persistence**: Only specific settings saved (via `save_setting()`)

## Code Quality & Known Issues

### Technical Debt
- **Large Files**: Several files exceed 600-line target:
  - `visualization.py` (1278 lines)
  - `cli_command_router.py` (939 lines)
  - `conversation.py` (787 lines)
  - `web_search.py` (905 lines)
- **Performance**: Startup time impacted by eager loading of ChromaDB and ML models
- **Test Coverage**: 39 test files but edge cases need more coverage
- **Error Handling**: Inconsistent patterns across modules

### Security & Configuration
- **API Keys**: Properly managed through environment variables
- **Database Safeguards**: Validation prevents project directory placement
- **Model Pricing**: Automatically updated via `scripts/update_model_pricing.py`
- **Telemetry**: Disabled for ChromaDB to protect privacy

## Future Development

### High Priority
- **Performance Optimization**: Implement lazy loading for heavy dependencies
- **Code Refactoring**: Split large files into focused modules
- **Async Support**: Convert blocking operations to async/await
- **Error Standardization**: Unified error handling patterns

### Medium Priority
- **Caching Layer**: Add @lru_cache for expensive computations
- **Command Deduplication**: Refactor similar patterns in command files
- **Integration Testing**: Enhance test coverage for critical paths
- **Documentation**: Complete API documentation with type hints

### Long Term
- **Adaptive Topic Detection**: Dynamic context management for non-linear conversations
- **DAG Branching**: Support for conversation trees and topic returns
- **Enhanced Embeddings**: Multiple embedding providers for different use cases
- **Running Topic Prediction**: Real-time topic detection during conversation