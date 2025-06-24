# Episodic Codebase Structure

Detailed documentation of the Episodic project architecture and file organization.

## Core Architecture

Episodic is built around a conversational DAG (Directed Acyclic Graph) model where each node represents a message in the conversation. The system supports branching conversations, topic detection, compression, and multi-model LLM integration.

## File Structure Overview

```
episodic/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for python -m episodic
├── cli.py                   # Command-line interface and interaction loop
├── core.py                  # Core data structures (Node, ConversationDAG)
├── db.py                    # Database operations and persistence
├── llm.py                   # LLM integration and model management
├── topics.py                # Topic detection and management
├── compression.py           # Async compression system
├── visualization.py         # Graph visualization utilities
├── config.py               # Configuration management
├── prompt_manager.py       # System prompt management
├── llm_config.py          # LLM provider configuration
├── server.py              # HTTP server for visualization
└── ml/                    # Machine learning features
    ├── drift.py           # Semantic drift detection
    ├── embeddings/        # Embedding providers
    └── summarization/     # Summarization strategies
```

## Detailed File Documentation

### Core System Files

#### `__main__.py`
**Purpose**: Application entry point
- Imports and delegates to CLI main function
- Handles command-line argument parsing
- Sets up the execution environment

#### `cli.py` (~2000 lines)
**Purpose**: Command-line interface and user interaction
- **Main Components**:
  - `talk_loop()`: Main interaction loop for conversational mode
  - `_handle_chat_message()`: Processes user messages and LLM responses
  - `process_command()`: Routes slash commands to handlers
  - Command implementations (init, add, show, compress, etc.)
- **Key Features**:
  - Rich terminal UI with syntax highlighting
  - Command history and auto-suggestions
  - Session cost tracking
  - Integration with all other modules
- **Dependencies**: Uses Typer for CLI framework, Prompt Toolkit for rich input

#### `core.py`
**Purpose**: Core data structures and abstractions
- **Classes**:
  - `Node`: Represents a single message/response in the conversation
    - Attributes: id, content, parent_id, role, timestamp
    - Methods: to_dict(), from_dict()
  - `ConversationDAG`: Manages the conversation graph structure
    - Methods: add_node(), get_ancestors(), get_descendants()
    - Supports branching and merging conversations
- **Design Pattern**: Immutable nodes with parent references create the DAG

#### `db.py`
**Purpose**: Database persistence and operations
- **Database**: SQLite with thread-safe connection handling
- **Schema**:
  - `nodes` table: Stores conversation nodes
  - `topics` table: Stores detected topics with ranges
  - `compression` table: Stores compression metadata
- **Key Functions**:
  - `insert_node()`: Add new conversation node
  - `get_node()`: Retrieve node by ID
  - `get_ancestry()`: Get conversation history
  - `store_topic()`: Save detected topic
  - `update_topic_end_node()`: Extend topic boundaries
- **Features**:
  - Short ID generation (2-char identifiers)
  - Full-text search capabilities
  - Migration support

### LLM Integration

#### `llm.py`
**Purpose**: Multi-provider LLM integration
- **Core Functions**:
  - `query_llm()`: Direct LLM queries
  - `query_with_context()`: Queries with conversation history
  - `get_context_messages()`: Build context from conversation DAG
- **Features**:
  - Provider-agnostic interface via LiteLLM
  - Token counting and cost tracking
  - Prompt caching support
  - Context window management
  - Error handling and retries
- **Supported Providers**: OpenAI, Anthropic, Ollama, Google, etc.

#### `llm_config.py`
**Purpose**: LLM provider configuration and model management
- Model availability checking
- Provider-specific settings
- Default model selection
- API key validation

### Topic Management

#### `topics.py` (NEW)
**Purpose**: Topic detection and management system
- **Classes**:
  - `TopicManager`: Encapsulates all topic-related functionality
- **Key Functions**:
  - `detect_topic_change_separately()`: Analyzes messages for topic changes
  - `extract_topic_ollama()`: Extracts topic names using LLM
  - `should_create_first_topic()`: Determines when to create initial topic
  - `build_conversation_segment()`: Formats conversation for analysis
  - `count_nodes_in_topic()`: Counts messages in a topic range
- **Features**:
  - Separate LLM calls for topic detection
  - Configurable sensitivity thresholds
  - Topic boundary management
  - Integration with compression system

### Compression System

#### `compression.py`
**Purpose**: Async background compression of conversation segments
- **Classes**:
  - `CompressionJob`: Represents a queued compression task
  - `AsyncCompressionManager`: Manages background compression
- **Components**:
  - Background worker thread
  - Priority queue for compression jobs
  - Topic-aware compression boundaries
  - Automatic triggering on topic changes
- **Features**:
  - Non-blocking compression
  - Configurable compression strategies
  - Compression statistics tracking
  - Error handling and retries

### Visualization

#### `visualization.py`
**Purpose**: Graph visualization of conversation DAG
- Creates interactive visualizations using Plotly
- Generates NetworkX graphs from conversation data
- Supports filtering and layout algorithms
- Real-time updates via HTTP polling

#### `server.py`
**Purpose**: HTTP server for visualization interface
- Flask-based web server
- Serves visualization HTML/JS
- Provides REST endpoints for graph data
- Supports real-time updates

### Configuration

#### `config.py`
**Purpose**: Application configuration management
- **Features**:
  - JSON-based configuration storage
  - User-specific settings (~/.episodic/config.json)
  - Default values with override support
  - Runtime configuration changes
- **Key Settings**:
  - Model preferences
  - Display options
  - Compression settings
  - Topic detection parameters

#### `prompt_manager.py`
**Purpose**: System prompt management
- Loads prompts from markdown files
- Supports prompt metadata (name, description, tags)
- Active prompt switching
- Prompt template rendering
- Custom prompt creation

### Machine Learning Features

#### `ml/drift.py`
**Purpose**: Semantic drift detection between messages
- Calculates semantic similarity using embeddings
- Tracks conversation topic evolution
- Provides drift metrics and visualization
- Supports multiple embedding providers

#### `ml/embeddings/providers.py`
**Purpose**: Embedding generation for semantic analysis
- Multiple provider support (OpenAI, Sentence Transformers)
- Caching for performance
- Fallback handling
- Batch processing support

#### `ml/summarization/strategies.py`
**Purpose**: Different summarization approaches
- **Strategies**:
  - Simple: Basic summary
  - Detailed: Comprehensive summary
  - Bullets: Bullet-point format
  - Structured: Sectioned summary
  - Topic-aware: Preserves topic boundaries
- Configurable per use case

## Data Flow

1. **User Input** → CLI captures message
2. **Topic Detection** → TopicManager analyzes for changes
3. **LLM Query** → Message sent to selected model
4. **Response Processing** → Format and display response
5. **Persistence** → Save nodes to database
6. **Topic Extension** → Update topic boundaries
7. **Compression Queue** → Queue old topics for compression
8. **Background Compression** → Async worker compresses topics

## Key Design Patterns

### 1. **DAG Structure**
- Immutable nodes with parent references
- Enables conversation branching
- Supports non-linear exploration

### 2. **Provider Abstraction**
- LiteLLM for model-agnostic interface
- Easy addition of new providers
- Consistent error handling

### 3. **Async Processing**
- Background compression doesn't block UI
- Thread-safe database operations
- Queue-based job management

### 4. **Modular Architecture**
- Clear separation of concerns
- Each module has single responsibility
- Easy to test and maintain

### 5. **Configuration-Driven**
- Behavior controlled by configuration
- Runtime adjustments without code changes
- User preferences persistence

## Extension Points

### Adding New Commands
1. Add command handler in `cli.py`
2. Update help text
3. Add to command routing

### Adding New LLM Providers
1. Configure in LiteLLM
2. Add to model list in `llm_config.py`
3. Handle provider-specific errors

### Adding Compression Strategies
1. Create new strategy in `ml/summarization/strategies.py`
2. Register in strategy map
3. Add configuration option

### Custom Topic Detection
1. Modify prompt in `prompts/topic_detection.md`
2. Adjust detection thresholds
3. Implement custom detection logic

## Testing

### Test Structure
```
tests/
├── test_core.py         # Core data structure tests
├── test_db.py          # Database operation tests
├── test_topics.py      # Topic detection tests
├── test_compression.py # Compression system tests
└── test_integration.py # End-to-end tests
```

### Manual Testing
- Interactive test scripts in `scripts/`
- Test different conversation patterns
- Verify topic detection accuracy
- Check compression effectiveness

## Performance Considerations

### Database
- Indexes on frequently queried columns
- Connection pooling for concurrent access
- Periodic vacuum for optimization

### LLM Queries
- Prompt caching reduces API calls
- Context window management prevents errors
- Batch operations where possible

### Memory Usage
- Streaming responses for large content
- Compression reduces storage needs
- Configurable context depth

### Async Operations
- Non-blocking compression
- Thread-safe database access
- Efficient queue management

## Security Considerations

- API keys in environment variables
- No sensitive data in prompts
- User data isolated in local database
- Configurable model access
- No automatic data sharing

## Future Architecture Considerations

### Potential Enhancements
1. **Plugin System**: Dynamic loading of extensions
2. **Multi-user Support**: User isolation and permissions  
3. **Cloud Sync**: Optional conversation backup
4. **Advanced Analytics**: Conversation insights
5. **Voice Interface**: Speech-to-text integration
6. **Export Formats**: Markdown, JSON, HTML exports
7. **Embedding Storage**: Persistent embedding cache
8. **Real-time Collaboration**: Shared conversations