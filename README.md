# Episodic: A Conversational DAG-Based Memory Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Episodic is a persistent, navigable memory system for conversational AI that stores conversations as a directed acyclic graph (DAG). Each node represents a message exchange, enabling rich conversation history, topic tracking, and context management.

## Key Features

### Core Functionality
- **Persistent Memory**: SQLite-based storage of all conversations
- **Talk-First Interface**: Natural conversation with `/` prefix for commands
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Ollama, Google, and more via LiteLLM
- **DAG Structure**: Navigate and branch conversations (currently linear, branching planned)

### Advanced Features
- **Automatic Topic Detection**: Intelligent conversation segmentation using:
  - Sliding window semantic analysis
  - Keyword and transition detection
  - LLM-based topic identification
- **Conversation Compression**: Automatic summarization of completed topics
- **Interactive Visualization**: Real-time graph view of conversation flow
- **Cost Tracking**: Token usage and API cost monitoring
- **Script Automation**: Save and replay conversation sequences
- **RAG (Retrieval Augmented Generation)**: Enhance responses with external knowledge
  - Index documents and PDFs into vector database
  - Automatic context retrieval for relevant queries
  - Document management with deduplication
- **Web Search Integration**: Access current information from the web
  - Configurable search providers (DuckDuckGo by default)
  - Automatic web search when local knowledge insufficient
  - Result caching and rate limiting for efficiency

### Recent Improvements (v0.4.0)
- **Unified Commands**: Cleaner CLI with grouped subcommands
- **Enhanced Testing**: Organized test infrastructure with fixtures
- **Modular Architecture**: Reorganized codebase for maintainability
- **Migration System**: Database schema versioning and updates

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/episodic.git
cd episodic
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Optional: Install RAG dependencies for knowledge base features
pip install chromadb sentence-transformers

# Optional: Install web search dependencies
pip install aiohttp beautifulsoup4

# Set up LLM provider (choose one)
export OPENAI_API_KEY=your_api_key_here
# OR
export ANTHROPIC_API_KEY=your_api_key_here
# OR use local Ollama (no API key needed)

# Start Episodic
python -m episodic

# Initialize on first run
> /init
> Hello! Let's have a conversation.
```

## Usage Examples

### Basic Conversation

```bash
# Natural conversation flow
> Tell me about the history of computing
[Assistant responds...]

> What role did Ada Lovelace play?
[Assistant responds...]

# Check current topics
> /topics
╭─ Recent Topics ──────────────────────────────────╮
│ 1. computing-history (12 messages) - ongoing    │
╰──────────────────────────────────────────────────╯
```

### Command Examples

```bash
# Topic management
> /topics list               # List all topics
> /topics rename            # Rename ongoing topics
> /topics stats             # Show topic statistics

# Settings management
> /settings                 # Show all settings
> /settings set debug true  # Enable debug mode
> /settings cost           # Show session costs

# Navigation
> /list                    # List recent nodes
> /head n3                # Jump to node n3
> /ancestry n5            # Show path to node n5

# Visualization
> /visualize              # Open interactive graph
```

### Advanced Features

```bash
# Model switching
> /model
[1] openai/gpt-3.5-turbo
[2] anthropic/claude-2
[3] ollama/llama2
Select model: 2

# Script automation
> /save my-conversation     # Save current session
> /script my-conversation   # Replay saved script

# Run scripts non-interactively
python -m episodic --execute scripts/my-conversation.txt
# Or short form:
python -m episodic -e scripts/my-conversation.txt

# Manual topic detection
> /topics index 5          # Analyze with window size 5

# RAG (Knowledge Base)
> /rag on                  # Enable RAG
> /index README.md         # Index a document
> /search neural networks  # Search knowledge base
> /docs list              # List indexed documents

# Web Search
> /websearch on            # Enable web search
> /websearch Python 3.12   # Search the web
> /ws latest AI news       # Short form
# Web results can enhance responses when local knowledge is insufficient!
```

## Project Structure

```
episodic/
├── __main__.py          # Entry point
├── cli.py               # CLI interface
├── conversation.py      # Conversation management
├── topics/              # Topic detection module
│   ├── detector.py      # Main detection logic
│   ├── boundaries.py    # Boundary analysis
│   └── windows.py       # Sliding window analysis
├── commands/            # CLI commands
│   ├── registry.py      # Command registry
│   ├── unified_*.py     # Unified command handlers
│   ├── rag.py          # RAG commands
│   └── web_search.py   # Web search commands
├── rag.py              # RAG system with ChromaDB
├── rag_utils.py        # RAG utilities and decorators
├── web_search.py       # Web search providers and caching
└── migrations/          # Database migrations
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and components
- **[CLAUDE.md](CLAUDE.md)** - AI assistant integration guide
- **[DECISIONS.md](DECISIONS.md)** - Architectural decisions log
- **[DEPRECATED.md](DEPRECATED.md)** - Deprecated features and timeline

## Development

```bash
# Run tests
python tests/run_all_tests.py all      # All tests
python tests/run_all_tests.py unit     # Unit tests only
python tests/run_all_tests.py quick    # Quick tests

# Clean up imports
python scripts/cleanup/remove_unused_imports.py --apply
```

## Requirements

- Python 3.8+
- SQLite (included with Python)
- LLM provider API key (or local Ollama)

## Contributing

Contributions are welcome! Please:
1. Check existing issues and PRs
2. Follow the existing code style
3. Add tests for new features
4. Update documentation as needed

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Built with [LiteLLM](https://github.com/BerriAI/litellm) for LLM provider abstraction
- Uses [Typer](https://typer.tiangolo.com/) for CLI interface
- Visualization powered by [Plotly](https://plotly.com/) and [NetworkX](https://networkx.org/)
