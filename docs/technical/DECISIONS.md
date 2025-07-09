# Architectural Decisions

This document records key architectural and design decisions made during the development of Episodic.

## Topic Detection Architecture (2024-11)

### Decision: Sliding Window Detection
- **Choice**: Sliding window approach over single-message detection
- **Reason**: Better accuracy (83% vs 45% in testing)
- **Trade-off**: Slightly more complex implementation
- **Implementation**: See `episodic/topics/windows.py`

### Decision: Hybrid Detection System
- **Choice**: Combine multiple detection methods (embeddings, keywords, LLM)
- **Reason**: No single method works well for all cases
- **Benefits**: 
  - Embeddings catch semantic drift
  - Keywords catch explicit transitions
  - LLM provides nuanced analysis
- **Trade-off**: More complex, but more robust

### Decision: Configurable Thresholds
- **Choice**: Different thresholds for first topics vs later topics
- **Reason**: Conversations often start with exploration
- **Implementation**: First 2 topics use threshold/2, then full threshold

## Database Schema (2024-11)

### Decision: Separate Detection Scores Table
- **Choice**: Store detection scores in `topic_detection_scores` table
- **Reason**: Flexibility for experiments without polluting main tables
- **Benefits**:
  - Can store multiple detection attempts
  - Rich metadata for analysis
  - Easy to clear/reset for testing

### Decision: Nullable end_node_id for Topics
- **Choice**: Allow NULL end_node_id for ongoing topics
- **Reason**: Topics need to exist before they're complete
- **Benefits**:
  - Natural representation of ongoing conversations
  - Simplifies topic creation logic
  - Easy to query active topics

### Decision: Migration System
- **Choice**: Custom migration system over SQLAlchemy
- **Reason**: Lightweight, no heavy dependencies
- **Implementation**: Simple up/down migrations with history tracking

## Command Structure (2025-01)

### Decision: Unified Commands with Subactions
- **Choice**: `/topics [action]` over `/topics`, `/rename-topics`, etc.
- **Reason**: Reduces command sprawl
- **Benefits**:
  - Logical grouping of related functions
  - Easier discovery
  - Cleaner help text
- **Example**: `/topics rename` instead of `/rename-topics`

### Decision: Command Registry Pattern
- **Choice**: Central registry over scattered command definitions
- **Reason**: Better organization and discoverability
- **Benefits**:
  - Easy to add deprecation warnings
  - Organized help by categories
  - Single source of truth

### Decision: Maintain Backward Compatibility
- **Choice**: Keep old commands working with deprecation warnings
- **Reason**: Don't break existing user workflows
- **Timeline**: Remove in v0.5.0

## Configuration Management (2025-01)

### Decision: Centralized Defaults
- **Choice**: All defaults in `config_defaults.py`
- **Reason**: No more magic numbers throughout code
- **Benefits**:
  - Easy to find and modify defaults
  - Self-documenting
  - Consistent naming

### Decision: Database-Stored Configuration
- **Choice**: Store config in SQLite, not files
- **Reason**: Single source of truth with conversation data
- **Trade-off**: Can't edit config when app not running

## Testing Strategy (2025-01)

### Decision: Separate Unit and Integration Tests
- **Choice**: Clear directory structure by test type
- **Reason**: Different testing needs and run times
- **Benefits**:
  - Can run quick unit tests during development
  - Integration tests for full flows
  - Easy to run categories separately

### Decision: Reusable Test Fixtures
- **Choice**: Centralized test conversations and utilities
- **Reason**: Avoid duplication across tests
- **Implementation**: `tests/fixtures/` directory

## Code Organization (2025-01)

### Decision: Module-Based Topic Detection
- **Choice**: Split `topics.py` into organized module
- **Reason**: Single file was becoming too large
- **Benefits**:
  - Clear separation of concerns
  - Easier to find specific functionality
  - Better for team development

### Decision: Keep ML Experiments Separate
- **Choice**: `ml/` directory for experimental features
- **Reason**: Not ready for production use
- **Future**: Either implement fully or remove

## LLM Integration

### Decision: LiteLLM for Provider Abstraction
- **Choice**: Use LiteLLM over custom provider code
- **Reason**: Maintained library with wide provider support
- **Benefits**:
  - Automatic provider detection
  - Consistent interface
  - Built-in retry logic

### Decision: Centralized API Call Tracking
- **Choice**: `LLMManager` singleton for call statistics
- **Reason**: Need accurate cost tracking across threads
- **Implementation**: Thread-safe call logging

## Performance Decisions

### Decision: Thread-Local Database Connections
- **Choice**: Don't share connections across threads
- **Reason**: SQLite threading limitations
- **Trade-off**: More connections, but safer

### Decision: Lazy Context Loading
- **Choice**: Load conversation context only when needed
- **Reason**: Not all commands need full history
- **Benefit**: Faster command response times

### Decision: Streaming LLM Responses
- **Choice**: Stream tokens as they arrive
- **Reason**: Better user experience
- **Implementation**: Character-based streaming with buffer

## Future Decisions to Make

1. **Branching Strategy**: How to implement conversation branches?
2. **Multi-User**: Separate databases or single with user column?
3. **Plugin System**: What's the plugin API?
4. **Semantic Search**: Which embedding model and vector store?
5. **Real-time Sync**: WebSockets or Server-Sent Events?

## Rejected Alternatives

### Single-File Commands
- **Rejected**: Each command in separate file
- **Reason**: Too many small files
- **Chosen**: Group related commands

### ORM for Database
- **Rejected**: SQLAlchemy or similar
- **Reason**: Overkill for our needs
- **Chosen**: Direct SQL with careful practices

### YAML Configuration Files
- **Rejected**: File-based configuration
- **Reason**: Another file to manage
- **Chosen**: Database configuration

### Monolithic Test File
- **Rejected**: All tests in one file
- **Reason**: Hard to maintain and run selectively
- **Chosen**: Organized test structure