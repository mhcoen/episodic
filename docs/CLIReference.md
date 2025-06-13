# CLI Reference

This document provides a comprehensive reference for all Episodic CLI commands.

## Basic Commands

### Initialize the Database

```bash
episodic init
```

Creates a new database or resets an existing one.

### Add a Message

```bash
episodic add "Your message here"
```

Adds a new node with the specified content as a child of the current node.

### Show a Node

```bash
episodic show <node_id>
episodic show HEAD  # Shows the current node
```

Displays the content and metadata of the specified node.

### Change the Current Node

```bash
episodic goto <node_id>
```

Changes the current node to the specified node.

### List Recent Nodes

```bash
episodic list
episodic list --count 10  # Show 10 most recent nodes
```

Lists the most recent nodes in the database.

### Show Ancestry

```bash
episodic ancestry <node_id>
```

Shows the ancestry (thread history) of the specified node.

## LLM Integration Commands

### Query an LLM

```bash
episodic query "Your question here"
episodic query --model gpt-4 "Your question here"
episodic query --system "Custom system message" "Your question here"
```

Sends a one-off query to the LLM and stores both the query and response in the conversation DAG.

### Chat with an LLM

```bash
episodic chat "Your message here"
episodic chat --model claude-3-opus "Your message here"
```

Continues a conversation with context from previous messages.

## Visualization Commands

### Generate Visualization

```bash
episodic visualize
episodic visualize --output conversation.html
episodic visualize --no-browser
episodic visualize --port 5001
```

Generates and opens an interactive visualization of the conversation DAG.

### Native Window Visualization

```bash
episodic visualize --native
episodic visualize --native --width 1200 --height 900
```

Opens the visualization in a native window instead of a web browser.

## Interactive Shell

The interactive shell provides a more fluid user experience with command history, auto-completion, and syntax highlighting.

```bash
# After installation
episodic-shell

# Before installation (or without installation)
python -m episodic.cli
```

In the shell, you can use all the commands without quotation marks:

```
episodic> add Hello, world.
episodic> query What is the capital of France?
episodic> goto 01
```

Use `help` to see all available commands:

```
episodic> help
```

Or get help for a specific command:

```
episodic> help query
```

## Short Node IDs

Episodic uses short, human-readable IDs for nodes in addition to the traditional UUIDs. These short IDs:

- Are 2-3 characters long (alphanumeric, base-36 encoding)
- Are sequential, making it easy to understand the order of creation
- Can be used anywhere a node ID is required (show, ancestry, parent references)
- Make it much easier to reference nodes in the command line

Example:

```bash
# Adding a node shows both the short ID and UUID
$ episodic add "Hello, world."
Added node 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)

# You can reference nodes using the short ID
$ episodic show 01
Node ID: 01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9)
Parent: None
Message: Hello, world.

# Short IDs are also shown in ancestry
$ episodic ancestry 01
01 (UUID: 3a7e46c9-8b0e-4c1a-9f0a-8e5b3a7e46c9): Hello, world.
```