# Advanced Usage

This document covers advanced features and usage patterns for Episodic.

## Interactive Shell

Episodic includes an interactive shell for a more fluid user experience:

```bash
# Launch the interactive shell (after installing with pip)
episodic-shell

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

### Example Usage in the Shell

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

## LLM Integration

Episodic integrates with various LLM providers through LiteLLM, allowing you to:
- Query an LLM and store both the query and response in the conversation DAG
- Chat with an LLM using conversation history as context
- Switch between different LLM providers (OpenAI, Anthropic, Ollama, LMStudio, etc.)

### Example LLM Usage

```bash
# Send a one-off query to the LLM
episodic query "What is the capital of France?"
Added query node 03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb)
Added response node 04 (UUID: 6da179fc-be31-5f4d-d2d3-b17e6da179fc)

# Continue a conversation with context from previous messages
episodic chat "Tell me more about its history."
Added query node 05 (UUID: 7eb28a0d-cf42-6e5e-e3e4-c28f7eb28a0d)
Added response node 06 (UUID: 8fc39b1e-d053-7f6f-f4f5-d3908fc39b1e)

# Specify a different model
episodic query --model gpt-4 "Explain quantum computing."
Added query node 07 (UUID: 90d4ac2f-e164-8g7g-g5g6-e4a190d4ac2f)
Added response node 08 (UUID: a1e5bd30-f275-9h8h-h6h7-f5b2a1e5bd30)

# Customize the system message
episodic query --system "You are a helpful coding assistant." "How do I write a Python function?"
Added query node 09 (UUID: b2f6ce41-g386-ai9i-i7i8-g6c3b2f6ce41)
Added response node 0a (UUID: c3g7df52-h497-bj0j-j8j9-h7d4c3g7df52)
```

For more details on LLM providers, see the [LLM Providers](./LLMProviders.md) documentation.

## Conversation Branching and Navigation

One of Episodic's key features is the ability to branch conversations and navigate between different branches:

```bash
# Start a conversation
episodic add "Hello, world."
Added node 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

# Add a child node
episodic add "This is branch A."
Added node 02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da)

# Go back to the root node
episodic goto 01
Current node changed to: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

# Create a different branch
episodic add "This is branch B."
Added node 03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb)

# View the ancestry of a node to see its thread history
episodic ancestry 03
03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb): This is branch B.
01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9): Hello, world.
```

This branching capability allows you to explore different conversation paths and easily switch between them.
