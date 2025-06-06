Episodic: A Conversational DAG-Based Memory Agent

Overview

Episodic is a prototype for a persistent, navigable memory system for interacting with generative language models. It treats conversation as a directed acyclic graph (DAG), where each node represents a discrete conversational step (query and response). Nodes are linked by parent-child relationships, enabling users to branch, backtrack, and resume conversations at any point.

Features
	•	Persistent storage of conversations using SQLite
	•	CLI tool for adding and inspecting conversation nodes
	•	Unique ID referencing and human-friendly aliases for navigation
	•	Ability to show ancestry (thread history) of any conversational node
	•	Designed to support eventual LLM integration

Goals

This system is designed to explore richer interfaces for language models, overcoming the limitations of linear chat histories. Higher-level objectives include:
	•	Supporting user-directed and model-assisted navigation of past conversational states
	•	Allowing semantic forking and merging of conversation threads
	•	Investigating mechanisms for episodic memory using external summarization tools
	•	Exploring git-like branching for experimentation with prompts, models, or memory length

Getting Started

# Clone the repo and navigate to the directory
cd ~/proj/episodic
python -m venv .venv
source .venv/bin/activate
pip install -e .

## Running Episodic Commands

There are two ways to run Episodic commands:

### Before Installation (or without installation)

Use the Python module syntax:

```bash
python -m episodic <command> [arguments]
```

### After Installation

After installing the package with pip (as shown above), you can use the direct command:

```bash
episodic <command> [arguments]
```

This works because pip automatically creates executable wrapper scripts for the entry points defined in setup.py.

## Basic Usage

Initialize the database:

```bash
python -m episodic init
```

Add a message:

```bash
python -m episodic add "Hello, world."
```

Show a message:

```bash
python -m episodic show HEAD
```

Project Structure

episodic/
├── __init__.py
├── __main__.py
├── db.py
├── state.py
└── ...

LLM Integration

Episodic now integrates with OpenAI's chat completion API, allowing you to:
	•	Query an LLM and store both the query and response in the conversation DAG
	•	Chat with an LLM using conversation history as context

To use these features, you need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Example usage:

```bash
# Send a one-off query to the LLM
python -m episodic query "What is the capital of France?"

# Continue a conversation with context from previous messages
python -m episodic chat "Tell me more about its history."

# Specify a different model
python -m episodic query --model gpt-4 "Explain quantum computing."

# Customize the system message
python -m episodic query --system "You are a helpful coding assistant." "How do I write a Python function?"
```

## Visualization

Episodic now includes a visualization tool to explore the conversation DAG:

```bash
# Generate and open an interactive visualization in your browser
python -m episodic visualize

# Save the visualization to a specific file
python -m episodic visualize --output conversation.html

# Generate the visualization without opening it in a browser
python -m episodic visualize --no-browser
```

The visualization allows you to:
- See the entire conversation structure as a directed graph
- Hover over nodes to see the full content
- Zoom in/out and pan around the graph
- Move nodes to explore different layouts
- See the hierarchical structure of conversations

## Interactive Shell

Episodic now includes an interactive shell for a more fluid user experience:

### After Installation

```bash
# Launch the interactive shell (after installing with pip)
episodic-shell
```

### Before Installation (or without installation)

```bash
# Launch the interactive shell using the Python module syntax
python -m episodic.cli
```

The interactive shell provides several advantages over the command-line interface:

- **Persistent State**: Maintains context between commands, including the current node
- **Command History**: Remembers commands between sessions
- **Auto-completion**: Suggests commands and arguments as you type (press Tab)
- **Syntax Highlighting**: Makes commands more readable
- **Help System**: Built-in documentation for all commands
- **No Quotation Marks Required**: Arguments are automatically parsed based on command flags (--)

Example usage in the shell:

The following examples show what you'll see after launching the interactive shell. The `episodic>` prompt is displayed by the shell itself, and you only need to type the commands that follow it:

```
episodic> init
Database initialized.

episodic> add Hello, world.
Added node 1234-5678-90ab-cdef

episodic> show
Node ID: 1234-5678-90ab-cdef
Parent: None
Message: Hello, world.

episodic> query What is the capital of France? --model gpt-4
Added query node 2345-6789-0abc-defg
Added response node 3456-7890-abcd-efgh

LLM Response:
The capital of France is Paris. It's one of the world's major global cities and...

episodic> help
Available commands:
  add         - Add a new node with content
  ancestry    - Show the ancestry of a node
  chat        - Chat with an LLM using conversation history
  exit        - Exit the shell
  help        - Show help for a command or list all commands
  init        - Initialize the database
  query       - Query an LLM and store the result
  quit        - Exit the shell
  show        - Show a specific node
  visualize   - Create an interactive visualization of the conversation DAG

Type 'help <command>' for more information on a specific command.
```

Next Steps
	•	Implement state summarization
	•	Add support for other LLM providers

License

MIT
