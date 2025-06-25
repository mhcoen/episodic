# Development & Testing

This document provides information for developers who want to contribute to Episodic or modify it for their own purposes.

## Project Structure

```
episodic/
├── __init__.py
├── __main__.py
├── cli.py
├── core.py
├── db.py
├── llm.py
├── visualization.py
└── ...
```

## Testing

The test suite is located in the `tests/` directory. See [tests/README.md](../tests/README.md) for details on running tests.

Quick start:
```bash
# Run all tests
python -m unittest discover

# Run tests with the test runner
cd tests
python run_tests.py
```

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
