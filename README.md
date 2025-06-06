Episodic: A Conversational DAG-Based Memory Agent

Overview

Episodic is a prototype for a persistent, navigable memory system for interacting with generative language models. It treats conversation as a directed acyclic graph (DAG), where each node represents a discrete conversational step (query and response). Nodes are linked by parent-child relationships, enabling users to branch, backtrack, and resume conversations at any point.

Features
	•	Persistent storage of conversations using SQLite
	•	CLI tool for adding and inspecting conversation nodes
	•	Short, human-readable node IDs for easy reference
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

Change the current node:

```bash
python -m episodic goto 01
```

List recent nodes:

```bash
python -m episodic list
python -m episodic list --count 10
```

## Short Node IDs

Episodic now uses short, human-readable IDs for nodes in addition to the traditional UUIDs. These short IDs:

- Are 2-3 characters long (alphanumeric, base-36 encoding)
- Are sequential, making it easy to understand the order of creation
- Can be used anywhere a node ID is required (show, ancestry, parent references)
- Make it much easier to reference nodes in the command line

### Migrating Existing Databases

If you have an existing database created with a previous version of Episodic, you can migrate it to use short IDs by running the following Python code:

```python
from episodic.db import migrate_to_short_ids

# Migrate existing nodes to use short IDs
count = migrate_to_short_ids()
print(f"Migrated {count} nodes to use short IDs")
```

This will add short IDs to all existing nodes in your database.

Example:

```bash
# Adding a node shows both the short ID and UUID
$ python -m episodic add "Hello, world."
Added node 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

# You can reference nodes using the short ID
$ python -m episodic show 01
Node ID: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)
Parent: None
Message: Hello, world.

# Short IDs are also shown in ancestry
$ python -m episodic ancestry 01
01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9): Hello, world.
```

## Project Structure

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
Added query node 03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb)
Added response node 04 (UUID: 6da179fc-be31-5f4d-d2d3-b17e6da179fc)

# Continue a conversation with context from previous messages
python -m episodic chat "Tell me more about its history."
Added query node 05 (UUID: 7eb28a0d-cf42-6e5e-e3e4-c28f7eb28a0d)
Added response node 06 (UUID: 8fc39b1e-d053-7f6f-f4f5-d3908fc39b1e)

# Specify a different model
python -m episodic query --model gpt-4 "Explain quantum computing."
Added query node 07 (UUID: 90d4ac2f-e164-8g7g-g5g6-e4a190d4ac2f)
Added response node 08 (UUID: a1e5bd30-f275-9h8h-h6h7-f5b2a1e5bd30)

# Customize the system message
python -m episodic query --system "You are a helpful coding assistant." "How do I write a Python function?"
Added query node 09 (UUID: b2f6ce41-g386-ai9i-i7i8-g6c3b2f6ce41)
Added response node 0a (UUID: c3g7df52-h497-bj0j-j8j9-h7d4c3g7df52)
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
Database initialized with a default root node (ID: 01, UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9).

episodic> add Hello, world.
Added node 02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da)

episodic> show
Node ID: 02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da)
Parent: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)
Message: Hello, world.

episodic> goto 01
Current node changed to: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

episodic> query What is the capital of France? --model gpt-4
Added query node 03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb)
Added response node 04 (UUID: 6da179fc-be31-5f4d-d2d3-b17e6da179fc)

LLM Response:
The capital of France is Paris. It's one of the world's major global cities and...

episodic> list
Recent nodes (showing 5 of 5 requested):
04 (UUID: 6da179fc-be31-5f4d-d2d3-b17e6da179fc): The capital of France is Paris. It's one of the world...
03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb): What is the capital of France?
02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da): Hello, world.
01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9): 

episodic> list --count 3
Recent nodes (showing 3 of 3 requested):
04 (UUID: 6da179fc-be31-5f4d-d2d3-b17e6da179fc): The capital of France is Paris. It's one of the world...
03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb): What is the capital of France?
02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da): Hello, world.

episodic> help
Available commands:
  add         - Add a new node with content
  ancestry    - Show the ancestry of a node
  chat        - Chat with an LLM using conversation history
  exit        - Exit the shell
  goto        - Change the current node
  help        - Show help for a command or list all commands
  init        - Initialize the database
  list        - List recent nodes
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
