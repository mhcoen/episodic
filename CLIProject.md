
# CLI Structure with Typer

This document describes the CLI structure of Episodic, which uses Typer to provide a clean, intuitive interface for interacting with the system.

## Current Structure

The CLI has a single main component:

1. A command-line interface in `episodic/__main__.py` that uses the implementation in `episodic/cli.py`

## Implementation

The CLI is implemented using Typer:

### 1. Typer Installation

```bash
pip install typer
```

### 2. CLI Structure

The CLI is implemented in `episodic/cli.py` with the following structure:

```python
import typer
import shlex
from typing import Optional, List
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML

from episodic.db import (
    insert_node, get_node, get_ancestry, initialize_db, 
    resolve_node_ref, get_head, set_head, database_exists,
    get_recent_nodes
)
from episodic.llm import query_llm, query_with_context
from episodic.llm_config import get_current_provider, get_default_model
from episodic.prompt_manager import PromptManager
from episodic.config import config

# Create a Typer app for command handling
app = typer.Typer(add_completion=False)

# Global variables to store state
current_node_id = None
default_model = "gpt-3.5-turbo"
default_system = "You are a helpful assistant."
default_context_depth = 5
session_costs = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_tokens": 0,
    "total_cost_usd": 0.0
}

# Command handlers
@app.command()
def init(erase: bool = typer.Option(False, "--erase", "-e", help="Erase existing database")):
    """Initialize the database."""
    global current_node_id

    if database_exists() and not erase:
        erase_confirm = typer.confirm("The database already exists. Do you want to erase it?")
        if not erase_confirm:
            typer.echo("Database initialization cancelled.")
            return

    result = initialize_db(erase=True if database_exists() else False)
    if result:
        root_node_id, root_short_id = result
        current_node_id = root_node_id
        # Reset the active prompt to default
        config.set("active_prompt", "default")
        # Update the default system message
        try:
            manager = PromptManager()
            global default_system
            default_system = manager.get_active_prompt_content(config.get)
        except Exception:
            # Fallback to default if there's an error
            default_system = "You are a helpful assistant."
        typer.echo(f"Database initialized with a default root node (ID: {root_short_id}, UUID: {root_node_id}).")
        typer.echo("Prompt has been restored to default.")
    else:
        typer.echo("Database initialized.")

@app.command()
def add(content: str, parent: Optional[str] = typer.Option(None, "--parent", "-p", help="Parent node ID")):
    """Add a new node with the given content."""
    global current_node_id

    parent_id = resolve_node_ref(parent) if parent else current_node_id or get_head()
    node_id, short_id = insert_node(content, parent_id)
    set_head(node_id)
    current_node_id = node_id
    typer.echo(f"Added node {short_id} (UUID: {node_id})")

# Add more command handlers for other commands (show, head, ancestry, etc.)

# Main talk loop
def talk_loop():
    """Main talk loop that handles both conversation and commands."""
    global current_node_id, default_model, default_system, default_context_depth, session_costs

    # Initialize the database if it doesn't exist
    if not database_exists():
        typer.echo("No database found. Initializing...")
        init()

    # Get the current head node
    current_node_id = get_head()
    if not current_node_id:
        typer.echo("No conversation history found. Creating initial node...")
        init()
        current_node_id = get_head()

    # Create a prompt session for the talk mode
    history_file = config.get("history_file", "~/.episodic_history")
    session = PromptSession(
        message=HTML("<ansigreen>> </ansigreen>"),
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
    )

    # Print welcome message
    typer.echo("Welcome to Episodic. You are now in talk mode.")
    typer.echo("Type a message to chat with the LLM, or use / to access commands.")
    typer.echo("Examples: '/help', '/init', '/add Hello', '/exit'")

    # Get the current model and provider
    provider = get_current_provider()
    model = get_default_model()
    typer.echo(f"Current model: {model} (Provider: {provider})")

    # Main loop
    while True:
        try:
            # Get user input
            user_input = session.prompt()

            # Skip empty input
            if not user_input.strip():
                continue

            # Check if it's a command (starts with /)
            if user_input.startswith('/'):
                # Remove the / prefix and parse the command
                command_text = user_input[1:].strip()
                if not command_text:
                    typer.echo("Empty command. Type '/help' for available commands.")
                    continue

                # Handle exit command directly
                if command_text.lower() in ["exit", "quit"]:
                    typer.echo("Goodbye!")
                    break

                # Parse the command and arguments
                try:
                    args = shlex.split(command_text)
                    command = args[0].lower()
                    command_args = args[1:]

                    # Use Typer's CLI to handle the command
                    try:
                        # Convert command_args to the format expected by Typer
                        from typer.core import TyperCommand
                        for cmd in app.registered_commands:
                            if cmd.name == command:
                                cmd.callback(*command_args)
                                break
                        else:
                            typer.echo(f"Unknown command: {command}")
                            typer.echo("Type '/help' for available commands.")
                    except Exception as e:
                        typer.echo(f"Error executing command: {str(e)}")
                except Exception as e:
                    typer.echo(f"Error parsing command: {str(e)}")
            else:
                # It's a conversation message, handle it like the talk command
                # Get the ancestry of the head node to use as context
                head_id = current_node_id or get_head()
                ancestry = get_ancestry(head_id)

                # Limit the context to the specified depth
                context_ancestry = ancestry[-default_context_depth:] if default_context_depth > 0 else ancestry

                # Convert the ancestry to the format expected by the LLM
                context_messages = []
                for i, node in enumerate(context_ancestry):
                    # Skip the first node if it's a system message or has no parent
                    if i == 0 and node['parent_id'] is None:
                        continue

                    # Use the stored role if available, otherwise fall back to alternating roles
                    role = node.get('role')
                    if role is None:
                        # Fallback to alternating roles if role is not stored
                        role = "user" if i % 2 == 0 else "assistant"
                    context_messages.append({"role": role, "content": node['content']})

                # Store the user query as a node with "user" role
                query_node_id, query_short_id = insert_node(user_input, head_id, role="user")

                # Query the LLM with context
                try:
                    response, cost_info = query_with_context(
                        prompt=user_input,
                        context_messages=context_messages,
                        model=default_model,
                        system_message=default_system
                    )

                    # Store the LLM response as a node with the query as its parent and "assistant" role
                    response_node_id, response_short_id = insert_node(response, query_node_id, role="assistant")

                    # Update the current node and head
                    current_node_id = response_node_id
                    set_head(response_node_id)

                    # Display the response with colored formatting
                    typer.echo("")  # Empty line before response
                    provider = get_current_provider()

                    # Display model info with cost information on the same line if enabled
                    if config.get("show_cost", False):
                        typer.echo(f"\033[36m🤖 {provider}/{default_model}: ({cost_info['input_tokens']}_in + {cost_info['output_tokens']}_out = {cost_info['total_tokens']}_tokens ${cost_info['cost_usd']:.6f} USD)\033[0m")
                    else:
                        typer.echo(f"\033[36m🤖 {provider}/{default_model}:\033[0m")

                    typer.echo(response)

                    # Update session cost totals
                    session_costs["total_input_tokens"] += cost_info["input_tokens"]
                    session_costs["total_output_tokens"] += cost_info["output_tokens"]
                    session_costs["total_tokens"] += cost_info["total_tokens"]
                    session_costs["total_cost_usd"] += cost_info["cost_usd"]
                except Exception as e:
                    typer.echo(f"Error: {str(e)}")

        except KeyboardInterrupt:
            # Handle Ctrl+C
            typer.echo("^C")
            continue
        except EOFError:
            # Handle Ctrl+D
            typer.echo("^D")
            typer.echo("Goodbye!")
            break
        except Exception as e:
            typer.echo(f"Error: {str(e)}")

def main():
    """Main entry point for the Episodic CLI."""
    # Always set debug to False when starting the CLI
    config.set("debug", False)

    # Start the talk loop
    talk_loop()

if __name__ == "__main__":
    main()
```

### 3. Update `episodic/__main__.py`

Update the `main()` function in `episodic/__main__.py` to use the new CLI:

```python
def main():
    # Import and use the CLI
    from episodic.cli import main as cli_main
    cli_main()

if __name__ == "__main__":
    main()
```

## Key Features of the New CLI

1. **Single Loop**: The talk loop is now the main loop, eliminating the CLI within a CLI structure.

2. **Command Prefix**: Commands are accessed by prefixing them with a "/" character.

3. **Typer Integration**: Commands are defined using Typer decorators, making the code more maintainable and providing automatic help text.

4. **Preserved Functionality**: All existing functionality is preserved, just reorganized into a more user-friendly structure.

## Benefits

1. **Simplified User Experience**: Users interact with a single interface for both conversation and commands.

2. **Intuitive Command Access**: The "/" prefix is a common convention for accessing commands in chat interfaces.

3. **Maintainable Code**: Typer provides a clean, declarative way to define commands with proper argument handling.

4. **Extensible**: Adding new commands is as simple as adding new decorated functions.

This approach gives you the best of both worlds: a conversational interface that's primarily focused on talking with the LLM, with easy access to commands when needed.
