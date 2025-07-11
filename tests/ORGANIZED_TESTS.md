# Episodic Test Organization and Running Guide

## Overview

The Episodic test suite is organized into different categories to support various testing needs:
- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test complete command execution paths
- **CLI tests**: Test the interactive command-line interface

## Test Directory Structure

```
tests/
├── run_all_tests.py         # Main test runner with category support
├── unit/                    # Unit tests (mocked dependencies)
│   ├── commands/            # Command function tests
│   └── topics/              # Topic detection tests
├── integration/             # Integration tests
│   └── cli/                 # CLI command tests
│       ├── test_all_commands.py    # Comprehensive CLI testing
│       └── test_cli_commands.py    # unittest-based CLI tests
└── fixtures/                # Shared test data and utilities
```

## Running Tests

### Using the Main Test Runner

The primary way to run tests is using `run_all_tests.py`:

```bash
# Run all tests
python tests/run_all_tests.py all

# Run specific test categories
python tests/run_all_tests.py unit        # Unit tests only
python tests/run_all_tests.py integration # Integration tests only
python tests/run_all_tests.py quick       # Quick, stable tests
python tests/run_all_tests.py topics      # Topic-related tests
python tests/run_all_tests.py commands    # Command tests
python tests/run_all_tests.py coverage    # With coverage report

# Additional options
python tests/run_all_tests.py all -v      # Verbose output
python tests/run_all_tests.py all --failfast  # Stop on first failure
```

### Running Comprehensive CLI Tests

For the most thorough testing of all CLI commands:

```bash
# Run comprehensive CLI test (recommended)
python tests/integration/cli/test_all_commands.py

# Analyze results
python tests/integration/cli/analyze_test_results.py
```

This tests ~60 CLI commands and generates a detailed report of any failures.

### Running Specific Test Files

```bash
# Using unittest directly
python -m unittest tests.unit.commands.test_unified_commands -v
python -m unittest tests.integration.cli.test_cli_commands -v

# Using pytest (if installed)
pytest tests/unit -v
pytest tests/integration -v
pytest -k "test_topics" -v  # Run tests matching pattern
```

## Test Categories Explained

### Unit Tests (`tests/unit/`)
- Mock external dependencies (database, LLM calls)
- Test individual functions and classes
- Fast execution
- Good for TDD and quick feedback

### Integration Tests (`tests/integration/`)
- Test complete workflows
- Use real database and configurations
- May use some mocks for external services
- Good for verifying feature behavior

### CLI Integration Tests (`tests/integration/cli/`)
- **test_all_commands.py**: Most comprehensive
  - Runs actual CLI with subprocess
  - Tests interactive commands with echo piping
  - Handles timeout issues properly
  - Best for finding real-world issues
  
- **test_cli_commands.py**: unittest framework integration
  - Attempts to work within test framework
  - Has timeout challenges with interactive CLI
  - Good for CI/CD integration

## Why CLI Testing Is Challenging

1. **Interactive Nature**: The Episodic CLI is designed for interactive use, expecting continuous input
2. **Database State**: Commands depend on conversation history and configuration
3. **Streaming Output**: Real-time response streaming complicates output capture
4. **External Services**: Some commands (RAG, web search) require external dependencies

## Recommended Testing Workflow

1. **During Development**: Run unit tests for the module you're working on
   ```bash
   python tests/run_all_tests.py unit
   ```

2. **Before Committing**: Run quick tests to catch obvious issues
   ```bash
   python tests/run_all_tests.py quick
   ```

3. **After Major Changes**: Run comprehensive CLI tests
   ```bash
   python tests/integration/cli/test_all_commands.py
   ```

4. **For Full Validation**: Run all tests with coverage
   ```bash
   python tests/run_all_tests.py coverage
   ```

## Test Fixtures and Utilities

The `tests/fixtures/` directory contains:
- `conversations.py`: Sample conversation data
- `test_utils.py`: Shared testing utilities
- Mock objects for database and LLM interactions

## Adding New Tests

1. **For new commands**: Add unit tests in `tests/unit/commands/`
2. **For new features**: Add integration tests in appropriate subdirectory
3. **For CLI behavior**: Update `test_all_commands.py` with new command tests

## Continuous Integration

The test suite is designed to work with CI systems:
- Exit code 0 on success, 1 on failure
- Detailed output for debugging
- Coverage reports for tracking test completeness

## Known Issues

1. **Interactive CLI Timeouts**: Some unittest-based CLI tests may timeout due to the interactive nature
2. **Database Isolation**: Tests may interfere if run in parallel
3. **External Dependencies**: RAG and web search tests require additional setup

## Best Practices

1. Use `test_all_commands.py` for comprehensive CLI validation
2. Write unit tests for new functions before implementation
3. Use mocks to avoid external dependencies in unit tests
4. Keep integration tests focused on specific workflows
5. Run full test suite before major commits