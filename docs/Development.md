# Development & Testing

This document provides information for developers who want to contribute to Episodic or modify it for their own purposes.

## Project Structure

```
episodic/
├── __init__.py
├── __main__.py
├── cli.py                    # Main CLI loop and command dispatcher
├── core.py                   # Core data structures (Node, ConversationDAG)
├── db.py                     # Database operations
├── db_compression.py         # Compression storage system
├── llm.py                    # LLM integration via LiteLLM
├── llm_config.py            # LLM provider configuration
├── conversation.py          # Conversation management
├── topics.py                # Topic detection and management
├── compression.py           # Async compression system
├── config.py                # Configuration management
├── visualization.py         # Graph visualization
├── commands/                # Command implementations
│   ├── __init__.py
│   ├── navigation.py        # Navigation commands (/head, /show, etc.)
│   ├── settings.py          # Configuration commands (/set, /verify)
│   ├── topics.py            # Topic commands (/topics, /rename-topics)
│   ├── compression.py       # Compression commands
│   ├── prompts.py           # Prompt management
│   ├── summary.py           # Summary generation
│   └── ...
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
