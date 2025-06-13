# Episodic: A Conversational DAG-Based Memory Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Episodic is a persistent, navigable memory system for interacting with generative language models. It treats conversation as a directed acyclic graph (DAG), where each node represents a discrete conversational step (query and response).

## Key Features

- Persistent storage of conversations using SQLite
- CLI tool and interactive shell for managing conversations
- Branching conversations with easy navigation
- Integration with various LLM providers (OpenAI, Anthropic, Ollama, etc.)
- Interactive visualization of conversation graphs
- Short, human-readable node IDs for easy reference

## Quick Start

```bash
# Installation
git clone https://github.com/yourusername/episodic.git
cd episodic
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set up your API key (for LLM integration)
export OPENAI_API_KEY=your_api_key_here

# Initialize and start using
episodic init
episodic-shell
```

## Example Usage

### Basic Conversation Flow

```bash
# Start the interactive shell
episodic-shell

# In the shell (notice no quotes needed):
episodic> add Hello, I'd like to explore quantum computing
Added node 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

episodic> query What is quantum superposition?
Added query node 02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da)
Added response node 03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb)

LLM Response:
Quantum superposition is a fundamental principle of quantum mechanics...

# Branch the conversation to explore a different topic
episodic> goto 01
Current node changed to: 01

episodic> query Tell me about quantum entanglement instead
Added query node 04 (UUID: 6da179fc-be31-5f4d-d2d3-b17e6da179fc)
Added response node 05 (UUID: 7eb28a0d-cf42-6e5e-e3e4-c28f7eb28a0d)

# Visualize the conversation graph
episodic> visualize --native
```

### Visualization Demo

```bash
# In a separate terminal, start the visualization server
episodic visualize

# Then continue using the CLI in another terminal
episodic-shell
```

## Documentation

For more detailed information, see:

- [Installation Guide](docs/Installation.md)
- [CLI Reference](docs/CLIReference.md)
- [LLM Providers](docs/LLMProviders.md)
- [Visualization Guide](docs/Visualization.md)
- [Advanced Usage](docs/AdvancedUsage.md)
- [Development & Testing](docs/Development.md)

## License

This project is licensed under the [MIT License](LICENSE).
