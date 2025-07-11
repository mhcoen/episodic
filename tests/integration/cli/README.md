# CLI Integration Tests

This directory contains comprehensive integration tests for the Episodic CLI that test actual command execution rather than mocked components.

## Test Scripts

### `test_all_commands.py`
- **Purpose**: Tests every CLI command by actually running them
- **Coverage**: ~60 commands tested end-to-end
- **Method**: Runs `python -m episodic` with each command and checks output
- **Usage**: `python tests/integration/cli/test_all_commands.py`

### `analyze_test_results.py`
- **Purpose**: Parses test output and categorizes failures
- **Input**: Reads `full_test_output.log` from test runs
- **Output**: Creates `error_summary.md` with categorized issues
- **Usage**: Run after `test_all_commands.py` to analyze results

## Difference from Unit Tests

Unlike the unit tests in `tests/unit/commands/`, these integration tests:
- Actually execute the full CLI application
- Test real command routing and execution
- Verify actual output and error messages
- Don't use mocks - they test the real implementation
- Can catch integration issues between components

## Running the Tests

```bash
# Run all CLI integration tests
cd /path/to/episodic
python tests/integration/cli/test_all_commands.py

# Analyze results
python tests/integration/cli/analyze_test_results.py
```

## Test Results

After running, check:
- `full_test_output.log` - Complete test output
- `error_summary.md` - Categorized list of failures
- Console output shows pass/fail summary

## Adding New Tests

To add a new command test, edit `test_all_commands.py` and add to the `commands_to_test` list:
```python
("/your-command", "Description of what it does"),
("/your-command arg1 arg2", "Description with arguments"),
```