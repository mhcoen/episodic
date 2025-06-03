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

Next Steps
	•	Integrate with OpenAI’s chat completion API
	•	Automate message submission and response storage
	•	Implement state summarization
	•	Build a basic TUI or web UI for visual graph traversal

License

MIT (or TBD)
