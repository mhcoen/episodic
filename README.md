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

Initialize the database:

python -m episodic init

Add a message:

python -m episodic add "Hello, world."

Show a message:

python -m episodic show HEAD

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

```bash
# Launch the interactive shell
episodic-shell
```

The interactive shell provides several advantages over the command-line interface:

- **Persistent State**: Maintains context between commands, including the current node
- **Command History**: Remembers commands between sessions
- **Auto-completion**: Suggests commands and arguments as you type (press Tab)
- **Syntax Highlighting**: Makes commands more readable
- **Help System**: Built-in documentation for all commands

Example usage in the shell:

```
episodic> init
Database initialized.

episodic> add "Hello, world."
Added node 1234-5678-90ab-cdef

episodic> show
Node ID: 1234-5678-90ab-cdef
Parent: None
Message: Hello, world.

episodic> query "What is the capital of France?" --model gpt-4
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

MIT (or TBD)
