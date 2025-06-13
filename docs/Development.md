# Development & Testing

This document provides information for developers who want to contribute to Episodic or modify it for their own purposes.

## Project Structure

```
episodic/
├── __init__.py
├── __main__.py
├── db.py
├── state.py
└── ...
```

## Testing

Episodic includes a comprehensive test suite to ensure code quality and reliability. The tests include:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test the interaction between components
- **Manual Tests**: Interactive tests that require human verification
- **Real-time Update Tests**: Test real-time updates via HTTP polling

### Running Tests

For general information on running tests and test coverage, see the [TESTING.md](./TESTING.md) file.

### Real-time Update Testing

To test the HTTP polling functionality for real-time updates, you can use the following test scripts:

```bash
# Run an interactive browser-based update test
python -m episodic.test_websocket_browser
```

Note: Despite the filename containing "websocket", this test now uses HTTP polling for updates.

The browser-based test provides an interactive experience:
- Opens the visualization in a browser
- Performs automated changes to the graph
- Allows you to verify that updates appear in real-time without page reloads
- Helps identify any browser-specific issues

## Contributing

Contributions to Episodic are welcome! Here are some guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests to ensure everything still works
5. Submit a pull request

## Future Development

Planned features for future development include:

- State summarization
- Enhanced visualization capabilities
- Additional LLM provider integrations
- Performance optimizations for large conversation graphs

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](../LICENSE) file for details.
