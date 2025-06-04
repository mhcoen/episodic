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
episodic query "What is the capital of France?"

# Continue a conversation with context from previous messages
episodic chat "Tell me more about its history."

# Specify a different model
episodic query --model gpt-4 "Explain quantum computing."

# Customize the system message
episodic query --system "You are a helpful coding assistant." "How do I write a Python function?"
```

Next Steps
	•	Implement state summarization
	•	Build a basic TUI or web UI for visual graph traversal
	•	Add support for other LLM providers

License

MIT (or TBD)
