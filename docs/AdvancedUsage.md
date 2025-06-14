# Advanced Usage

This document covers advanced features and usage patterns for Episodic.

## Talk Mode Interface

Episodic now uses a simplified CLI structure where the talk loop is the main interface, and commands are accessed by prefixing them with a "/" character.

```bash
# Start the application
python -m episodic
```

The talk mode interface provides several advantages:

- **Persistent State**: Maintains context between commands, including the current node
- **Command History**: Remembers commands between sessions
- **Auto-suggestion**: Suggests commands and messages as you type
- **Syntax Highlighting**: Makes commands and responses more readable
- **Help System**: Built-in documentation for all commands
- **Seamless Conversation**: Chat with the LLM without any special commands
- **Easy Command Access**: Access commands with the "/" prefix

### Example Usage in Talk Mode

The following examples show what you'll see in the talk mode interface:

```
> /init
Database initialized with a default root node (ID: 01, UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9).

> /add Hello, world.
Added node 02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da)

> /show 02
Node ID: 02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da)
Parent: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)
Message: Hello, world.

> /head 01
Current node changed to: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

> What is the capital of France?
🤖 openai/gpt-3.5-turbo:
The capital of France is Paris. It's one of the world's major global cities and...

> /list
Recent nodes (showing 5 of 5 requested):
04 (UUID: 6da179fc-be31-5f4d-d2d3-b17e6da179fc): The capital of France is Paris. It's one of the world...
03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb): What is the capital of France?
02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da): Hello, world.
01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9): 

> /help
Available commands:
  /help                - Show this help message
  /exit, /quit         - Exit the application
  /init [--erase]      - Initialize the database (--erase to erase existing)
  /add <content>       - Add a new node with the given content
  /show <node_id>      - Show details of a specific node
  /head [node_id]      - Show current node or change to specified node
  /list [--count N]    - List recent nodes (default: 5)
  /ancestry <node_id>  - Trace the ancestry of a node
  /visualize           - Visualize the conversation DAG
  /model               - Show or change the current model
  /prompts             - Manage system prompts

Type a message without a leading / to chat with the LLM.
```

## LLM Integration

Episodic integrates with various LLM providers through LiteLLM, allowing you to:
- Query an LLM and store both the query and response in the conversation DAG
- Chat with an LLM using conversation history as context
- Switch between different LLM providers (OpenAI, Anthropic, Ollama, LMStudio, etc.)

### Example LLM Usage

```bash
# Simply type your message to chat with the LLM
> What is the capital of France?
🤖 openai/gpt-3.5-turbo:
The capital of France is Paris. It's one of the world's major global cities and...

# Continue a conversation with context from previous messages
> Tell me more about its history.
🤖 openai/gpt-3.5-turbo:
Paris has a rich history dating back to ancient times. It was originally founded...

# Specify a different model using the /model command
> /model gpt-4
Switched to model: gpt-4 (Provider: openai)

> Explain quantum computing.
🤖 openai/gpt-4:
Quantum computing is a type of computing that uses quantum-mechanical phenomena...

# Customize the system message using the /prompts command
> /prompts use coding_assistant
Now using prompt: coding_assistant - A helpful coding assistant

> How do I write a Python function?
🤖 openai/gpt-3.5-turbo:
In Python, you define a function using the `def` keyword followed by the function name...
```

For more details on LLM providers, see the [LLM Providers](./LLMProviders.md) documentation.

## Conversation Branching and Navigation

One of Episodic's key features is the ability to branch conversations and navigate between different branches:

```bash
# Start a conversation
> /add Hello, world.
Added node 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

# Add a child node
> /add This is branch A.
Added node 02 (UUID: 4b8f57da-9c1f-4d2b-b0b1-9f5c4b8f57da)

# Go back to the root node
> /head 01
Current node changed to: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

# Create a different branch
> /add This is branch B.
Added node 03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb)

# View the ancestry of a node to see its thread history
> /ancestry 03
03 (UUID: 5c9068eb-ad20-4e3c-c1c2-a06d5c9068eb): This is branch B.
01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9): Hello, world.
```

This branching capability allows you to explore different conversation paths and easily switch between them.
