# CLI Reference

This document provides a comprehensive reference for all Episodic CLI commands.

Episodic now uses a simplified CLI structure where the talk loop is the main interface, and commands are accessed by prefixing them with a "/" character.

## Starting the Application

```bash
python -m episodic
```

This starts the application in talk mode, where you can chat with the LLM and use commands.

## Basic Commands

All commands are prefixed with a "/" character in the talk mode.

### Navigation Commands

#### Initialize the Database

```bash
/init
/init --erase  # Erase existing database and reset state
```

Creates a new database or resets an existing one.

#### Add a Message

```bash
/add Your message here
/add --parent 01 Your message here  # Add as child of specific node
```

Adds a new node with the specified content as a child of the current node.

#### Show a Node

```bash
/show <node_id>
```

Displays the content and metadata of the specified node.

#### Print Current Node

```bash
/print
/print <node_id>  # Print specific node
```

Prints the content of the current node or specified node.

#### Change the Current Node

```bash
/head
/head <node_id>
```

Shows or changes the current node to the specified node.

#### List Recent Nodes

```bash
/list
/list --count 10  # Show 10 most recent nodes
```

Lists the most recent nodes in the database.

#### Show Ancestry

```bash
/ancestry <node_id>
```

Shows the ancestry (thread history) of the specified node.

### Configuration Commands

#### Set Configuration

```bash
/set                           # Show all settings
/set <parameter>              # Show specific setting
/set <parameter> <value>      # Set a value

# Examples:
/set debug on                 # Enable debug mode
/set cache off               # Disable caching
/set cost on                 # Show cost information
/set show_topics true        # Show topic evolution
/set compression_model ollama/llama3
```

#### Verify Configuration

```bash
/verify
```

Verifies that the current configuration is valid and all providers are working.

#### Cost Information

```bash
/cost
```

Shows the session cost information for LLM usage.

### Topic Management

#### View Topics

```bash
/topics                      # Show recent topics
/topics --all               # Show all topics
/topics --limit 20          # Show 20 topics
/topics --verbose           # Show detailed topic info
```

Shows conversation topics with their ranges and message counts.

#### Rename Topics

```bash
/rename-topics
```

Analyzes and renames all placeholder "ongoing-*" topics based on their content.

#### Compress Current Topic

```bash
/compress-current-topic
```

Compresses the current topic (if closed) into a summary.

### Compression Commands

#### Manual Compression

```bash
/compress                    # Compress from root to head
/compress --node <node_id>   # Compress from root to specific node
/compress --strategy simple  # Use simple compression
/compress --dry-run         # Preview without compressing
```

#### Compression Statistics

```bash
/compression-stats
```

Shows statistics about compressed conversations.

#### Compression Queue

```bash
/compression-queue
```

Shows pending background compression jobs.

### Other Commands

#### Summary

```bash
/summary              # Summarize last 5 messages
/summary 10          # Summarize last 10 messages
/summary all         # Summarize entire conversation
```

Generates a summary of the conversation.

#### Benchmark

```bash
/benchmark
```

Shows performance benchmarks for various operations.

#### Script Execution

```bash
/script <filename>    # Execute commands from a script file
/save <filename>      # Save current session commands to a script
```

Execute or save conversation scripts.

#### Help

```bash
/help
```

Shows available commands and their usage.

## LLM Integration

In the talk mode, you can simply type your message without any command prefix to chat with the LLM. The system will automatically use the conversation history as context.

```bash
> What is quantum computing?
🤖 openai/gpt-3.5-turbo:
Quantum computing is a type of computing that uses quantum-mechanical phenomena...
```

### Change the Model

```bash
/model
/model gpt-4
```

Shows the current model or changes to a different model.

## Visualization Commands

### Generate Visualization

```bash
/visualize
/visualize --output conversation.html
/visualize --no-browser
/visualize --port 5001
```

Generates and opens an interactive visualization of the conversation DAG.

## Talk Mode Interface

The talk mode is now the main interface for Episodic. It provides a fluid user experience with command history, auto-suggestion, and a simple way to interact with the LLM.

```bash
# Start the application
python -m episodic
```

In the talk mode:
- Type a message without any prefix to chat with the LLM
- Use the "/" prefix to access commands

```
> Hello, world!
🤖 openai/gpt-3.5-turbo:
Hello! How can I assist you today?

> /help
Available commands:
  /help                - Show this help message
  /exit, /quit         - Exit the application
  /init [--erase]      - Initialize the database (--erase to erase existing)
  ...
```

The talk mode uses the [Typer](https://typer.tiangolo.com/) package for command handling and [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/) for the interactive interface.

## Short Node IDs

Episodic uses short, human-readable IDs for nodes in addition to the traditional UUIDs. These short IDs:

- Are 2-3 characters long (alphanumeric, base-36 encoding)
- Are sequential, making it easy to understand the order of creation
- Can be used anywhere a node ID is required (/show, /ancestry, --parent references)
- Make it much easier to reference nodes in the command line

Example:

```bash
# Adding a node shows both the short ID and UUID
> /add Hello, world.
Added node 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

# You can reference nodes using the short ID
> /show 01
Node ID: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)
Parent: None
Message: Hello, world.

# Short IDs are also shown in ancestry
> /ancestry 01
01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9): Hello, world.
```
