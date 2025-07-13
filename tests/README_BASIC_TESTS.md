# Basic Test Suite for Episodic

This directory contains comprehensive tests for the core Episodic functionality.

## Test Coverage

### Core Tests (`test_core.py`)
- Node creation and management
- ConversationDAG operations
- Node relationships and ancestry
- Node deletion and descendants

### Configuration Tests (`test_config.py`, `test_configuration_comprehensive.py`)
- Configuration file creation and persistence
- Value setting and getting with defaults
- Type handling (bool, int, float, list, dict)
- Nested configuration structures
- Model configurations for different contexts (chat, detection, compression, synthesis)
- Parameter validation
- Environment variable handling
- Configuration migration from old formats

### Conversation Flow Tests (`test_conversation_flow.py`)
- Basic conversation exchanges
- Multi-turn conversations
- Context building from history
- System prompt handling
- Node relationships in conversations
- Error handling
- Streaming responses
- Cost tracking
- Conversation state persistence

### Topic Detection Tests (`test_topic_detection_comprehensive.py`)
- Automatic topic detection during conversation
- Topic boundary detection and analysis
- Topic naming extraction
- Manual topic indexing
- Topic compression eligibility
- Configuration of detection parameters
- Hybrid detection methods (embeddings + keywords)
- Topic statistics
- Gradual drift detection

## Running Tests

### Run All Basic Tests
```bash
python tests/run_basic_tests.py
```

### Run Specific Test Module
```bash
python -m unittest tests.test_core -v
python -m unittest tests.test_conversation_flow -v
```

### Run Specific Test Class
```bash
python -m unittest tests.test_core.TestNode -v
python -m unittest tests.test_conversation_flow.TestConversationFlow -v
```

### Run Specific Test Method
```bash
python tests/run_basic_tests.py TestCore.test_node_creation
```

### Run with Different Verbosity
```bash
# Quiet mode
python tests/run_basic_tests.py -q

# Verbose mode
python tests/run_basic_tests.py -v
```

## Test Fixtures

The tests use fixtures from `tests/fixtures/`:
- `conversations.py` - Pre-defined test conversations with known topic boundaries
- `test_utils.py` - Utility functions for isolated configs and temporary databases

## Test Requirements

The tests use Python's built-in `unittest` framework with mocking for external dependencies:
- LLM calls are mocked to avoid API costs
- Database operations use temporary SQLite databases
- Configuration uses isolated test configs

## Adding New Tests

When adding new tests:
1. Follow the existing naming convention: `test_<feature>.py`
2. Use the test fixtures for consistent test data
3. Mock external dependencies (LLM calls, file I/O)
4. Clean up resources in `tearDown()` methods
5. Add the test module to `run_basic_tests.py`

## Common Test Patterns

### Mocking LLM Responses
```python
@patch('episodic.llm.query_llm')
def test_something(self, mock_llm):
    mock_llm.return_value = mock_llm_response("Test response")
    # Your test code
```

### Using Isolated Config
```python
def setUp(self):
    self.config_context = isolated_config()
    self.config = self.config_context.__enter__()
    
def tearDown(self):
    self.config_context.__exit__(None, None, None)
```

### Using Temporary Database
```python
def setUp(self):
    self.db_context = temp_database()
    self.db_path = self.db_context.__enter__()
    
def tearDown(self):
    self.db_context.__exit__(None, None, None)
```

## Troubleshooting

If tests fail:
1. Check that all dependencies are installed: `pip install -e .`
2. Ensure you're in the project root when running tests
3. Check for any hardcoded paths that might be system-specific
4. Look for any required environment variables
5. Verify that temporary directories have write permissions