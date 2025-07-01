# Episodic Architecture

## Overview

Episodic is a conversational memory system that stores conversations as a directed acyclic graph (DAG), enabling persistent and navigable dialogue history with language models. The system provides automatic topic detection, conversation compression, and multi-provider LLM support.

## Core Architecture

### Data Model

The conversation is stored as a DAG where:
- Each **node** represents a message exchange (user input + assistant response)
- Nodes are connected by **parent-child relationships**
- Currently, the DAG is **linear** (no branching implemented yet)
- Each node has a unique ID and a human-readable short ID (e.g., "n1", "ab")

### Key Components

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   CLI Interface │────▶│ Conversation    │────▶│    Database     │
│    (cli.py)     │     │   Manager       │     │    (SQLite)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Command Registry│     │ Topic Detection │     │   Migrations    │
│   & Handlers    │     │    Module       │     │    System       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        
         │                       │                        
         ▼                       ▼                        
┌─────────────────┐     ┌─────────────────┐     
│  LLM Manager    │     │  Compression    │     
│  (API Calls)    │     │    Manager      │     
└─────────────────┘     └─────────────────┘     
```

## Module Structure

### Core Modules

1. **cli.py** - Main entry point and command handling
   - Talk-first interface (default mode is conversation)
   - Command parsing and routing
   - Session management

2. **conversation.py** - ConversationManager class
   - Manages conversation flow
   - Handles topic detection triggers
   - Coordinates with database and LLM

3. **db.py** - Database operations
   - Thread-safe SQLite connections
   - CRUD operations for nodes, topics, configuration
   - Transaction management

4. **llm.py** - LLM integration
   - Multi-provider support via LiteLLM
   - Prompt caching
   - Cost tracking
   - Streaming responses

### Topic Detection System

Located in `episodic/topics/`:

```
topics/
├── detector.py      # Main TopicManager class
├── boundaries.py    # Boundary detection algorithms
├── hybrid.py        # Hybrid detection approach
├── keywords.py      # Keyword-based detection
├── windows.py       # Sliding window analysis
└── utils.py         # Shared utilities
```

**Detection Flow:**
1. User message triggers detection check
2. Multiple approaches analyze conversation:
   - Embedding-based semantic drift (primary)
   - Keyword/transition phrase detection
   - LLM-based analysis (fallback)
3. Topic boundaries are refined to find actual transition point
4. Topics are created/closed in database

### Command System

Post-refactoring command structure:

```
commands/
├── registry.py           # Command registration & discovery
├── unified_topics.py     # /topics command with subactions
├── unified_compression.py # /compression command
├── unified_settings.py   # /settings command
└── [legacy commands]     # Individual command implementations
```

**Unified Commands:**
- Commands now have subactions (e.g., `/topics list`, `/topics rename`)
- Registry tracks all commands with metadata
- Deprecated commands show warnings

### Configuration System

1. **config.py** - Runtime configuration management
2. **config_defaults.py** - All default values centralized
3. **configuration.py** - Constants and environment settings

Configuration is stored in the SQLite database and can be modified via `/settings` command.

## Database Schema

### Main Tables

1. **nodes** - Conversation nodes
   ```sql
   - id: UUID primary key
   - short_id: Human-readable ID
   - message: User input
   - response: Assistant response
   - parent_id: Reference to parent node
   - timestamp: Creation time
   - model_name: LLM model used
   - system_prompt: Active prompt
   ```

2. **topics** - Conversation topics
   ```sql
   - id: UUID primary key
   - name: Topic name
   - start_node_id: First node in topic
   - end_node_id: Last node (NULL for ongoing)
   - created_at: Creation timestamp
   ```

3. **topic_detection_scores** - Detection metadata
   ```sql
   - user_node_id: Node that triggered detection
   - topic_changed: Boolean result
   - detection_method: Method used
   - final_score: Combined score
   - [various score components]
   ```

4. **compressions_v2** - Compression summaries
5. **configuration** - Key-value settings
6. **migration_history** - Applied migrations

## Key Design Decisions

### 1. Linear DAG Structure
- Currently, conversations are linear (no branching)
- Foundation laid for future branching support
- Simplifies initial implementation while maintaining flexibility

### 2. Topic Detection Approach
- Hybrid system combining multiple signals
- Configurable thresholds and models
- Graceful degradation if methods fail

### 3. Command Organization
- Unified commands reduce top-level clutter
- Subactions provide logical grouping
- Registry enables easy discovery and deprecation

### 4. Database Migrations
- Version-controlled schema changes
- Automatic migration on startup
- Rollback capability for safety

### 5. Modular Architecture
- Clear separation of concerns
- Pluggable components (LLM providers, detection methods)
- Easy to extend and maintain

## Data Flow

### Conversation Flow
1. User input → CLI → ConversationManager
2. ConversationManager → Topic Detection (if enabled)
3. ConversationManager → LLM (with context)
4. LLM response → Streaming output → User
5. Store nodes and update topics → Database

### Topic Detection Flow
1. Check if detection should run (message count, settings)
2. Gather recent messages for analysis
3. Run detection algorithms in parallel
4. Combine scores and make decision
5. If topic changed, analyze boundary and update database

### Compression Flow
1. Topic marked complete or manually triggered
2. Gather all nodes in topic range
3. Generate summary via LLM
4. Store compression and mark nodes as compressed
5. Optional: Queue for async processing

## Extension Points

### Adding New LLM Providers
1. LiteLLM handles most providers automatically
2. Add provider-specific configuration in `llm_config.py`
3. Update model selection logic if needed

### Adding New Commands
1. Create command function in appropriate module
2. Register in `commands/registry.py`
3. For complex commands, create unified command with subactions

### Adding Topic Detection Methods
1. Create new detector in `topics/` module
2. Integrate with HybridTopicDetector
3. Add configuration options in `config_defaults.py`

### Database Schema Changes
1. Create new migration in `migrations/`
2. Implement `up()` and `down()` methods
3. Test migration thoroughly

## Performance Considerations

1. **Database Indexes** - Added for frequently queried columns
2. **Connection Pooling** - Thread-local connections for safety
3. **Prompt Caching** - Reduces API calls for repeated contexts
4. **Lazy Loading** - Context loaded only as needed
5. **Async Compression** - Background processing for large topics

## Security Considerations

1. **SQL Injection** - Parameterized queries throughout
2. **API Keys** - Stored in environment variables
3. **File Access** - Restricted to application directory
4. **Input Validation** - Commands and parameters validated

## Future Architecture Plans

1. **DAG Branching** - Support conversation branches
2. **Semantic Search** - Find similar past conversations
3. **Multi-User Support** - Separate conversation spaces
4. **Plugin System** - External extensions
5. **Real-time Sync** - WebSocket for live updates

## Testing Architecture

```
tests/
├── unit/          # Isolated component tests
├── integration/   # Full flow tests
├── fixtures/      # Reusable test data
└── run_all_tests.py # Test runner with categories
```

Tests are organized by type and can be run selectively based on needs.