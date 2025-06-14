# Episodic: A Conversational DAG-Based Memory Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Episodic is a persistent, navigable memory system for interacting with generative language models. It treats conversation as a directed acyclic graph (DAG), where each node represents a discrete conversational step (query and response).

## Key Features

- Persistent storage of conversations using SQLite
- Simplified CLI with a talk-first interface and command access via "/" prefix
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
pip install typer  # Required for the new CLI

# Set up your API key (for LLM integration)
export OPENAI_API_KEY=your_api_key_here

# Start the application
python -m episodic

# In the talk mode:
> /init  # Initialize the database
```

## Example Usage

### Basic Conversation Flow

```bash
# Start the application
python -m episodic

# In the talk mode (notice the / prefix for commands):
> Hello, I'd like to explore quantum computing
🤖 openai/gpt-3.5-turbo:
I'd be happy to explore quantum computing with you! It's a fascinating field that combines physics, computer science, and mathematics. What specific aspects of quantum computing would you like to learn about? For example, we could discuss:

1. Basic principles of quantum mechanics relevant to computing
2. Quantum bits (qubits) and how they differ from classical bits
3. Quantum gates and circuits
4. Quantum algorithms like Shor's or Grover's
5. Current state of quantum hardware
6. Potential applications and impact
7. Challenges in quantum computing

Let me know what interests you most!

> What is quantum superposition?
🤖 openai/gpt-3.5-turbo:
Quantum superposition is a fundamental principle of quantum mechanics...

# Branch the conversation to explore a different topic
> /head 01
Current node changed to: 01

> Tell me about quantum entanglement instead
🤖 openai/gpt-3.5-turbo:
Quantum entanglement is a fascinating phenomenon in quantum physics...

# Visualize the conversation graph
> /visualize
```

### Visualization Demo

```bash
# In the talk mode, start the visualization server
> /visualize

# The visualization will open in your browser
# Continue the conversation in the same terminal
> Tell me more about quantum computing
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
