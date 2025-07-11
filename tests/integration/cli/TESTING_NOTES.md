# CLI Testing Notes

## Current Testing Approach

The Episodic project has different levels of CLI testing:

### 1. Unit Tests (tests/unit/commands/)
- Mock-based testing of individual command functions
- Fast execution
- Tests logic in isolation
- Examples: `test_unified_commands.py`, `test_topics_command.py`

### 2. Integration Tests (tests/integration/cli/)
- **test_all_commands.py**: Comprehensive command testing
  - Runs actual CLI with subprocess
  - Tests ~60 commands end-to-end
  - Generates detailed error reports
  - Best for finding real-world issues

- **test_cli_commands.py**: unittest-based integration tests
  - Attempts to integrate with test framework
  - Has timeout issues due to interactive CLI

### 3. Manual Testing Scripts
- `test_all_commands.py` remains the most reliable way to test all CLI commands
- Handles the interactive nature of the CLI properly

## Why Comprehensive CLI Testing Is Challenging

1. **Interactive Nature**: Episodic CLI expects interactive input, making automated testing complex
2. **Database State**: Commands depend on database state and configuration
3. **External Dependencies**: Some commands (RAG, web search) require external services
4. **Streaming Output**: Real-time streaming makes output capture difficult

## Recommended Testing Approach

For comprehensive CLI testing:
```bash
# From project root
python tests/integration/cli/test_all_commands.py

# Analyze results
python tests/integration/cli/analyze_test_results.py
```

For unit testing specific command logic:
```bash
python tests/run_all_tests.py unit
```

## Test Coverage Status

- **Unit tests**: Cover individual command functions with mocks
- **Integration tests**: `test_all_commands.py` provides comprehensive coverage
- **Missing**: Automated integration tests that work with unittest framework (due to interactive CLI challenges)