# Episodic Test Suite

Comprehensive test suite for the Episodic CLI application, ensuring stability during development and modifications.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── README.md                   # This file
├── run_tests.py               # Test runner with colored output
├── test_cli.py                # CLI command and interface tests
├── test_config.py             # Configuration management tests
├── test_llm_integration.py    # LLM integration and provider tests
├── test_prompt_manager.py     # Prompt management system tests
└── test_caching.py            # Prompt caching functionality tests
```

## Running Tests

### Quick Start

Run all tests:
```bash
cd tests
python run_tests.py
```

Run only quick unit tests (recommended during development):
```bash
python run_tests.py quick
```

### Test Coverage

Generate coverage report:
```bash
pip install coverage
python run_tests.py coverage
```

### Individual Test Files

Run specific test modules:
```bash
python -m unittest test_cli
python -m unittest test_config
python -m unittest test_prompt_manager
python -m unittest test_llm_integration
python -m unittest test_caching
```

Run specific test classes:
```bash
python -m unittest test_cli.TestCLIHelpers
python -m unittest test_config.TestConfig
```

Run specific test methods:
```bash
python -m unittest test_cli.TestCLIHelpers.test_parse_flag_value
```

### Using pytest (alternative)

If you have pytest installed:
```bash
pip install pytest
pytest tests/
pytest tests/test_cli.py -v
```

## Test Categories

### 1. Unit Tests

**CLI Tests (`test_cli.py`)**
- Command parsing and flag handling
- Helper function validation
- Command execution (mocked)
- Session management
- Configuration integration

**Configuration Tests (`test_config.py`)**
- Config file creation and persistence
- Value setting/getting with different types
- Error handling for malformed files
- Unicode and large data handling

**Prompt Manager Tests (`test_prompt_manager.py`)**
- Prompt loading from files
- YAML frontmatter parsing
- Active prompt selection
- Integration with config system

**LLM Integration Tests (`test_llm_integration.py`)**
- Model string formatting
- Provider-specific handling
- Query execution (mocked)
- Cost calculation

**Caching Tests (`test_caching.py`)**
- Prompt caching implementation
- Cache control application
- Cost savings calculation
- Provider-specific caching behavior

### 2. Integration Tests

The existing integration tests in the main `episodic/` directory cover:
- Database operations (`test_db.py`)
- Core data structures (`test_core.py`) 
- Visualization (`test_integration.py`)
- WebSocket functionality (`test_websocket*.py`)

### 3. Manual/Interactive Tests

Located in `episodic/` directory:
- `test_interactive_features.py` - Interactive visualization tests
- `test_websocket_browser.py` - Browser-based WebSocket tests

## Test Development Guidelines

### Writing New Tests

1. **Follow naming conventions**: `test_*.py` for files, `test_*` for methods
2. **Use descriptive names**: `test_config_handles_unicode_values`
3. **Include docstrings**: Explain what each test validates
4. **Mock external dependencies**: Don't make real API calls or file operations
5. **Clean up after tests**: Use setUp/tearDown to manage test state

### Test Structure Template

```python
import unittest
from unittest.mock import patch, MagicMock

class TestNewFeature(unittest.TestCase):
    """Test description of the feature being tested."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize test data, mock objects, etc.
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore state, clean up temp files, etc.
        pass
    
    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        # Act  
        # Assert
        pass
```

### Mocking Guidelines

- **Mock external APIs**: LLM providers, file system operations
- **Mock time-dependent operations**: Use fixed timestamps for consistency
- **Mock configuration**: Use temporary config for isolation
- **Preserve original behavior**: Store and restore original values

### Coverage Goals

- **Minimum 80% code coverage** for new features
- **100% coverage** for critical paths (CLI commands, data handling)
- **Focus on edge cases**: Error conditions, malformed input, boundary values

## Continuous Integration

### Pre-commit Checks

Before committing changes, run:
```bash
python run_tests.py quick  # Fast feedback
```

### Full Test Suite

Before major changes or releases:
```bash
python run_tests.py        # Complete test suite
python run_tests.py coverage  # With coverage report
```

### Test Performance

- **Quick tests** should complete in < 30 seconds
- **Full test suite** should complete in < 2 minutes
- **Individual test methods** should complete in < 1 second

## Debugging Test Failures

### Common Issues

1. **Import errors**: Check PYTHONPATH and module structure
2. **Mock setup**: Verify mock objects are configured correctly
3. **State pollution**: Ensure tests clean up after themselves
4. **Timing issues**: Use deterministic values instead of time-dependent ones

### Debugging Tips

```bash
# Run single test with verbose output
python -m unittest test_cli.TestCLIHelpers.test_parse_flag_value -v

# Run with debugging
python -m pdb -m unittest test_cli.TestCLIHelpers.test_parse_flag_value

# Print debug information
python -c "import test_cli; test_cli.TestCLIHelpers().debug_method()"
```

## Test Data Management

### Temporary Files

Tests use `tempfile` module for temporary directories and files:
```python
import tempfile
import shutil

def setUp(self):
    self.test_dir = tempfile.mkdtemp()

def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)
```

### Mock Data

Common mock objects are defined in test files and can be reused:
```python
def create_mock_llm_response(self, content="Test response", tokens=100):
    """Create a standardized mock LLM response."""
    mock_response = Mock()
    mock_response.choices[0].message.content = content
    mock_response.usage.prompt_tokens = tokens
    return mock_response
```

## Adding Tests for New Features

When adding a new feature to Episodic:

1. **Add unit tests** for the core functionality
2. **Add integration tests** if it interacts with multiple components  
3. **Add CLI tests** if it includes new commands or flags
4. **Update this README** if new test categories are needed
5. **Ensure tests pass** before submitting changes

Example workflow:
```bash
# 1. Write your feature
# 2. Write tests
# 3. Run tests
python run_tests.py quick

# 4. Check coverage
python run_tests.py coverage

# 5. Run full suite
python run_tests.py

# 6. Commit when all tests pass
```

This test suite ensures that your modifications to Episodic won't break existing functionality and provides confidence in the stability of the codebase.