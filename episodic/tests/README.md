# Episodic Tests

This directory contains tests for the Episodic project.

## Test Types

The tests are organized into several categories:

1. **Unit Tests**: Test individual components in isolation
   - `test_core.py`: Tests for the core data structures
   - `test_db.py`: Tests for database operations
   - `test_server.py`: Tests for server endpoints

2. **Integration Tests**: Test the interaction between components
   - `test_integration.py`: Tests for visualization generation and WebSocket communication

3. **Manual Tests**: Interactive tests that require human verification
   - `test_interactive_features.py`: Tests for the interactive features of the visualization
   - `test_websocket.py`: Tests for WebSocket real-time updates

## Running the Tests

### Running All Tests

To run all the automated tests, use the following command:

```bash
python -m unittest discover episodic "test_*.py"
```

### Running Specific Tests

To run a specific test file, use the following command:

```bash
python -m unittest episodic.test_core
python -m unittest episodic.test_db
python -m unittest episodic.test_server
python -m unittest episodic.test_integration
```

To run a specific test case or method, use the following command:

```bash
python -m unittest episodic.test_core.TestNode
python -m unittest episodic.test_core.TestNode.test_node_creation
```

### Running Manual Tests

The manual tests require human interaction to verify the results. To run these tests, use the following commands:

```bash
python -m episodic.test_interactive_features
python -m episodic.test_websocket
```

Follow the instructions printed to the console to complete the tests.

## Test Coverage

To generate a test coverage report, install the `coverage` package and run the tests with coverage:

```bash
pip install coverage
coverage run -m unittest discover episodic "test_*.py"
coverage report -m
```

This will show the percentage of code covered by the tests and highlight any lines that are not covered.

## Writing New Tests

When adding new features to Episodic, please also add tests for those features. Follow these guidelines:

1. **Unit Tests**: Add tests for new classes and methods in the appropriate test file.
2. **Integration Tests**: Add tests for interactions between components in `test_integration.py`.
3. **Manual Tests**: If the feature requires human verification, add a manual test script.

All tests should follow the unittest framework conventions and include clear docstrings explaining what they test.