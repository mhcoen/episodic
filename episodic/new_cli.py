import typer
import shlex
import os
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
from episodic.llm_config import get_current_provider, get_default_model, get_available_providers, LOCAL_PROVIDERS
from episodic.prompt_manager import PromptManager
from episodic.config import config
from litellm import cost_per_token

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

@app.command()
def show(node_id: str):
    """Show details of a specific node."""
    try:
        resolved_id = resolve_node_ref(node_id)
        node = get_node(resolved_id)
        if node:
            typer.echo(f"Node ID: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.echo(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                typer.echo(f"Parent: None")

            # Display role information
            role = node.get('role')
            if role == "assistant":
                typer.echo(f"Role: Model (assistant)")
            elif role == "user":
                typer.echo(f"Role: User")
            elif role == "system":
                typer.echo(f"Role: System")
            else:
                typer.echo(f"Role: Unknown")

            typer.echo(f"Message: {node['content']}")
        else:
            typer.echo("Node not found.")
    except Exception as e:
        typer.echo(f"Error: {str(e)}")

@app.command()
def print_node(node_id: Optional[str] = None):
    """Print node info (defaults to current node)."""
    try:
        # Determine which node to print
        if not node_id:
            # If no node ID is provided, use the current head node
            node_id = current_node_id or get_head()
            if not node_id:
                typer.echo("No current node. Specify a node ID or use 'add' to create a node.")
                return
        else:
            # Resolve the provided node ID
            node_id = resolve_node_ref(node_id)

        # Get and display the node
        node = get_node(node_id)
        if node:
            typer.echo(f"Node ID: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.echo(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                typer.echo(f"Parent: None")

            # Display role information
            role = node.get('role')
            if role == "assistant":
                typer.echo(f"Role: Model (assistant)")
            elif role == "user":
                typer.echo(f"Role: User")
            elif role == "system":
                typer.echo(f"Role: System")
            else:
                typer.echo(f"Role: Unknown")

            typer.echo(f"Message: {node['content']}")
        else:
            typer.echo("Node not found.")
    except Exception as e:
        typer.echo(f"Error: {str(e)}")

@app.command()
def head(node_id: Optional[str] = None):
    """Show current node or change to specified node."""
    global current_node_id

    if not node_id:
        # If no node ID is provided, display the current node's info
        head_id = current_node_id or get_head()
        if not head_id:
            typer.echo("No current node. Specify a node ID or use 'add' to create a node.")
            return

        # Get and display the node
        node = get_node(head_id)
        if node:
            # Use a slightly different format for head to indicate it's the current node
            typer.echo(f"Current node: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.echo(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                typer.echo(f"Parent: None")
            typer.echo(f"Message: {node['content']}")
        else:
            typer.echo("Node not found.")
    else:
        # Resolve the node ID
        try:
            resolved_id = resolve_node_ref(node_id)

            # Verify that the node exists
            node = get_node(resolved_id)
            if not node:
                typer.echo(f"Error: Node not found: {node_id}")
                return

            # Update the current node
            set_head(resolved_id)
            current_node_id = resolved_id

            # Display confirmation
            typer.echo(f"Current node changed to: {node['short_id']} (UUID: {node['id']})")
        except Exception as e:
            typer.echo(f"Error: {str(e)}")

@app.command()
def list(count: int = typer.Option(5, "--count", "-c", help="Number of recent nodes to list")):
    """List recent nodes."""
    try:
        # Get recent nodes
        nodes = get_recent_nodes(count)

        if not nodes:
            typer.echo("No nodes found in the database.")
            return

        typer.echo(f"Recent nodes (showing {len(nodes)} of {count} requested):")
        for node in nodes:
            # Truncate content for display
            content = node['content']
            if len(content) > 50:
                content = content[:47] + "..."

            # Display node information
            typer.echo(f"{node['short_id']} (UUID: {node['id']}): {content}")
    except Exception as e:
        typer.echo(f"Error retrieving recent nodes: {str(e)}")

@app.command()
def handle_model(name: Optional[str] = None):
    """Show or change the current model."""
    global default_model

    from episodic.llm_config import get_provider_models, set_default_model

    if not name:
        # Show current model and available models
        provider = get_current_provider()
        current_model = get_default_model()
        typer.echo(f"Current model: {current_model} (Provider: {provider})")

        # Show available models from all providers
        try:
            # Get all available providers
            providers = get_available_providers()

            # Process all providers and collect model information
            for display_provider, provider_details in providers.items():
                provider_models = provider_details.get("models", [])
                if not provider_models:
                    continue

                # Collect models with pricing information
                priced_models = []  # Models with pricing > 0
                free_models = []    # Models with pricing = 0 or local models
                no_price_models = []  # Models without pricing information

                for model in provider_models:
                    model_name = model.get('name') if isinstance(model, dict) else model
                    model_with_provider = f"{display_provider}/{model_name}"

                    # Skip cost calculation for local providers
                    if display_provider in LOCAL_PROVIDERS:
                        free_models.append((model_name, "(Free - Local model)"))
                        continue

                    # Calculate cost information for non-local providers
                    try:
                        input_cost, output_cost = cost_per_token(
                            model=model_with_provider,
                            prompt_tokens=1000,
                            completion_tokens=1000
                        )
                        # Multiply by 1000 to get cost per 1K tokens
                        input_cost *= 1000
                        output_cost *= 1000
                        total_cost = input_cost + output_cost

                        # Check if it's a free model (both costs are 0)
                        if input_cost == 0 and output_cost == 0:
                            free_models.append((model_name, "(Free)"))
                        else:
                            cost_info = f"(${input_cost:7.4f}/1K input, ${output_cost:7.4f}/1K output tokens)"
#                            cost_info = f"(${input_cost:6.4f}/1K input, ${output_cost:6.4f}/1K output tokens)"
#                            cost_info = f"(${input_cost:>8.4f}/1K input, ${output_cost:>8.4f}/1K output tokens)"
                            priced_models.append((model_name, cost_info, total_cost))
                    except Exception:
                        # If cost calculation fails, add to no_price_models
                        no_price_models.append((model_name, "(Pricing not available)"))

                # Sort priced models by total cost
                priced_models.sort(key=lambda x: x[2])

                # Display models for this provider
                typer.echo(f"\nAvailable models from {display_provider}:")

                # First display priced models (sorted by price)
                for model_name, cost_info, _ in priced_models:
                    typer.echo(f"  - {model_name:<20}\t{cost_info}")

                # Then display free models
                for model_name, cost_info in free_models:
                    typer.echo(f"  - {model_name:<20}\t{cost_info}")

                # Finally display models without pricing
                for model_name, cost_info in no_price_models:
                    typer.echo(f"  - {model_name:<20}\t{cost_info}")

        except Exception as e:
            typer.echo(f"Error retrieving models: {str(e)}")
    else:
        # Change the model
        try:
            set_default_model(name)
            default_model = name
            provider = get_current_provider()
            typer.echo(f"Switched to model: {name} (Provider: {provider})")
        except ValueError as e:
            typer.echo(f"Error: {str(e)}")

@app.command()
def ancestry(node_id: str):
    """Trace the ancestry of a node."""
    try:
        resolved_id = resolve_node_ref(node_id)
        ancestry = get_ancestry(resolved_id)

        if not ancestry:
            typer.echo(f"No ancestry found for node: {node_id}")
            return

        for ancestor in ancestry:
            typer.echo(f"{ancestor['short_id']} (UUID: {ancestor['id']}): {ancestor['content']}")
    except Exception as e:
        typer.echo(f"Error: {str(e)}")

@app.command()
def visualize(output: Optional[str] = None, no_browser: bool = False, port: int = 5000):
    """Visualize the conversation DAG."""
    try:
        from episodic.visualization import visualize_dag
        import webbrowser
        import os
        import time

        output_path = visualize_dag(output)

        if output_path and not no_browser:
            from episodic.server import start_server, stop_server
            server_url = start_server(server_port=port)
            typer.echo(f"Starting visualization server at {server_url}")
            typer.echo("Press Ctrl+C when done to stop the server.")
            webbrowser.open(server_url)

            try:
                # Keep the server running until the user presses Ctrl+C
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                typer.echo("\nStopping server...")
                stop_server()
                typer.echo("Server stopped.")
        elif output_path:
            typer.echo(f"Visualization saved to: {output_path}")
            typer.echo(f"Opening visualization in browser: {output_path}")
            webbrowser.open(f"file://{os.path.abspath(output_path)}")
    except Exception as e:
        typer.echo(f"Error generating visualization: {str(e)}")

@app.command()
def set(param: Optional[str] = None, value: Optional[str] = None):
    """Configure various parameters."""
    global default_context_depth

    # If no parameter is provided, show all parameters and their current values
    if not param:
        typer.echo("Current parameters:")
        typer.echo(f"  cost: {config.get('show_cost', False)}")
        typer.echo(f"  depth: {default_context_depth}")
        typer.echo(f"  debug: {config.get('debug', False)}")
        return

    # Handle the 'cost' parameter
    if param.lower() == "cost":
        if not value:
            # Toggle the current value if no value is provided
            current = config.get("show_cost", False)
            config.set("show_cost", not current)
            typer.echo(f"Cost display: {'ON' if not current else 'OFF'}")
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("show_cost", val)
            typer.echo(f"Cost display: {'ON' if val else 'OFF'}")

    # Handle the 'depth' parameter
    elif param.lower() == "depth":
        if not value:
            typer.echo(f"Current context depth: {default_context_depth}")
        else:
            try:
                depth = int(value)
                if depth < 0:
                    typer.echo("Context depth must be a non-negative integer")
                    return
                default_context_depth = depth
                typer.echo(f"Context depth set to {depth}")
            except ValueError:
                typer.echo("Invalid value for depth. Please provide a non-negative integer")

    # Handle the 'debug' parameter
    elif param.lower() == "debug":
        if not value:
            # Toggle the current value if no value is provided
            current = config.get("debug", False)
            config.set("debug", not current)
            typer.echo(f"Debug mode: {'ON' if not current else 'OFF'}")
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("debug", val)
            typer.echo(f"Debug mode: {'ON' if val else 'OFF'}")

    # Handle unknown parameter
    else:
        typer.echo(f"Unknown parameter: {param}")
        typer.echo("Available parameters: cost, depth, debug")
        typer.echo("Use 'set' without arguments to see all parameters and their current values")

@app.command()
def verify():
    """Verify the current model with a test prompt."""
    try:
        # Get the current model and provider
        provider = get_current_provider()
        model_name = get_default_model()

        typer.echo(f"Verifying model: {model_name} (Provider: {provider})")
        typer.echo("Sending test prompts to verify model identity and behavior...")

        # Test prompt 1: Ask for model identification
        test_prompt1 = "What model are you? Please identify yourself and your capabilities."

        # Test prompt 2: Ask for something that might vary between models (using names)
        test_prompt2 = "My name is Michael. Can you address me by my name in your response?"

        # Send the first test prompt
        typer.echo("\nTest 1: Model self-identification")
        response1, cost_info1 = query_llm(
            prompt=test_prompt1,
            model=model_name,
            system_message="You are a helpful assistant. Please be truthful about your identity."
        )

        # Display the response
        typer.echo(f"\n🤖 {provider}/{model_name}:")
        typer.echo(response1)

        # Send the second test prompt
        typer.echo("\nTest 2: Name usage behavior")
        response2, cost_info2 = query_llm(
            prompt=test_prompt2,
            model=model_name,
            system_message="You are a helpful assistant."
        )

        # Display the response
        typer.echo(f"\n🤖 {provider}/{model_name}:")
        typer.echo(response2)

        # Provide a summary
        typer.echo("\nVerification complete. Different models may have different:")
        typer.echo("- Self-identification information")
        typer.echo("- Policies about using personal names")
        typer.echo("- Response styles and capabilities")
        typer.echo("- Knowledge cutoff dates")
        typer.echo("\nCompare these responses with other models to verify the model has changed.")

    except Exception as e:
        typer.echo(f"Error verifying model: {str(e)}")

@app.command()
def prompts(action: Optional[str] = None, name: Optional[str] = None):
    """Manage system prompts."""
    # Create a prompt manager instance
    manager = PromptManager()

    if not action or action == "list":
        # List all available prompts
        prompts = manager.list()
        if not prompts:
            typer.echo("No prompts found.")
            return

        typer.echo("Available prompts:")
        for prompt_name in prompts:
            metadata = manager.get_metadata(prompt_name)
            description = metadata.get('description', '') if metadata else ''
            typer.echo(f"  - {prompt_name}: {description}")

    elif action == "use":
        if not name:
            typer.echo("Error: Please specify a prompt name to use.")
            return

        # Set the active prompt
        if name not in manager.list():
            typer.echo(f"Prompt '{name}' not found.")
            return

        # Store the active prompt name in config
        config.set("active_prompt", name)

        # Update the default system message
        global default_system
        default_system = manager.get_active_prompt_content(config.get)

        # Display confirmation and description if available
        metadata = manager.get_metadata(name)
        description = f" - {metadata.get('description')}" if metadata and 'description' in metadata else ''
        typer.echo(f"Now using prompt: {name}{description}")

    elif action == "show":
        # Determine which prompt to show
        prompt_name = name if name else config.get("active_prompt", "default")

        # Get the prompt content
        prompt = manager.get(prompt_name)
        if not prompt:
            typer.echo(f"Prompt '{prompt_name}' not found.")
            return

        # Display the prompt information
        metadata = manager.get_metadata(prompt_name)
        description = f" - {metadata.get('description')}" if metadata and 'description' in metadata else ''

        typer.echo(f"--- Prompt: {prompt_name}{description} ---")
        typer.echo(prompt)

    elif action == "reload":
        # Reload prompts from disk
        manager.reload()
        typer.echo(f"Reloaded {len(manager.list())} prompts.")

    else:
        typer.echo("Unknown action. Available actions: list, use, show, reload")
        typer.echo("Usage:")
        typer.echo("  /prompts                - List all available prompts")
        typer.echo("  /prompts use <name>     - Set the active prompt")
        typer.echo("  /prompts show [name]    - Show the content of a prompt")
        typer.echo("  /prompts reload         - Reload prompts from disk")

@app.command()
def help():
    """Show available commands."""
    typer.echo("Available commands:")
    typer.echo("  /help                - Show this help message")
    typer.echo("  /exit, /quit         - Exit the application")
    typer.echo("  /init [--erase]      - Initialize the database (--erase to erase existing)")
    typer.echo("  /add <content>       - Add a new node with the given content")
    typer.echo("  /show <node_id>      - Show details of a specific node")
    typer.echo("  /print [node_id]     - Print node info (defaults to current node)")
    typer.echo("  /head [node_id]      - Show current node or change to specified node")
    typer.echo("  /list [--count N]    - List recent nodes (default: 5)")
    typer.echo("  /ancestry <node_id>  - Trace the ancestry of a node")
    typer.echo("  /visualize           - Visualize the conversation DAG")
    typer.echo("  /model               - Show or change the current model")
    typer.echo("  /verify              - Verify the current model with a test prompt")
    typer.echo("  /set [param] [value] - Configure parameters (cost, depth, debug)")
    typer.echo("  /prompts             - Manage system prompts")

    typer.echo("\nType a message without a leading / to chat with the LLM.")

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
    history_file = os.path.expanduser(config.get("history_file", "~/.episodic_history"))
    # Ensure the directory exists
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
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
    from episodic.llm_config import get_current_provider, get_default_model, set_default_model, ensure_provider_matches_model

    # Ensure the provider matches the model
    ensure_provider_matches_model()

    # Get the current model
    current_model_name = get_default_model()

    # Set the default model to ensure proper initialization
    try:
        set_default_model(current_model_name)
        # Update the default model in the global variable
        global default_model
        default_model = current_model_name
    except ValueError:
        # If there's an error setting the model, just continue
        pass

    # Get the current provider and model after initialization
    provider = get_current_provider()
    current_model_name = get_default_model()
    typer.echo(f"Current model: {current_model_name} (Provider: {provider})")

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
                    # Display total cost if any tokens were used
                    if session_costs["total_tokens"] > 0:
                        typer.echo("\nSession Summary:")
                        typer.echo(f"Total input tokens: {session_costs['total_input_tokens']}")
                        typer.echo(f"Total output tokens: {session_costs['total_output_tokens']}")
                        typer.echo(f"Total tokens: {session_costs['total_tokens']}")
                        typer.echo(f"Total cost: ${session_costs['total_cost_usd']:.6f} USD")
                    typer.echo("Goodbye!")
                    break

                # Parse the command and arguments
                try:
                    args = shlex.split(command_text)
                    command = args[0].lower()
                    command_args = args[1:]

                    # Handle commands directly
                    if command == "help":
                        help()
                    elif command == "init":
                        # Check for --erase flag
                        erase = False
                        if "--erase" in command_args or "-e" in command_args:
                            erase = True
                        init(erase=erase)
                    elif command == "add":
                        if not command_args:
                            typer.echo("Error: Missing content for add command")
                            continue
                        # Check for --parent flag
                        parent = None
                        if "--parent" in command_args:
                            parent_idx = command_args.index("--parent")
                            if parent_idx + 1 < len(command_args):
                                parent = command_args[parent_idx + 1]
                                # Remove parent and its value from args
                                command_args.pop(parent_idx)  # Remove --parent
                                command_args.pop(parent_idx)  # Remove parent value
                        elif "-p" in command_args:
                            parent_idx = command_args.index("-p")
                            if parent_idx + 1 < len(command_args):
                                parent = command_args[parent_idx + 1]
                                # Remove parent and its value from args
                                command_args.pop(parent_idx)  # Remove -p
                                command_args.pop(parent_idx)  # Remove parent value
                        # Join remaining args as content
                        content = " ".join(command_args)
                        add(content=content, parent=parent)
                    elif command == "show":
                        if not command_args:
                            typer.echo("Error: Missing node_id for show command")
                            continue
                        show(node_id=command_args[0])
                    elif command == "head":
                        node_id = command_args[0] if command_args else None
                        head(node_id=node_id)
                    elif command == "print":
                        node_id = command_args[0] if command_args else None
                        print_node(node_id=node_id)
                    elif command == "list":
                        # Check for --count flag
                        count = 5  # Default
                        if "--count" in command_args:
                            count_idx = command_args.index("--count")
                            if count_idx + 1 < len(command_args):
                                try:
                                    count = int(command_args[count_idx + 1])
                                except ValueError:
                                    typer.echo(f"Error: Invalid count value: {command_args[count_idx + 1]}")
                                    continue
                        elif "-c" in command_args:
                            count_idx = command_args.index("-c")
                            if count_idx + 1 < len(command_args):
                                try:
                                    count = int(command_args[count_idx + 1])
                                except ValueError:
                                    typer.echo(f"Error: Invalid count value: {command_args[count_idx + 1]}")
                                    continue
                        list(count=count)
                    elif command == "model":
                        name = command_args[0] if command_args else None
                        handle_model(name=name)
                    elif command == "prompts":
                        action = command_args[0] if len(command_args) > 0 else None
                        name = command_args[1] if len(command_args) > 1 else None
                        prompts(action=action, name=name)
                    elif command == "ancestry":
                        if not command_args:
                            typer.echo("Error: Missing node_id for ancestry command")
                            continue
                        ancestry(node_id=command_args[0])
                    elif command == "visualize":
                        # Parse options
                        output = None
                        no_browser = False
                        port = 5000

                        if "--output" in command_args:
                            output_idx = command_args.index("--output")
                            if output_idx + 1 < len(command_args):
                                output = command_args[output_idx + 1]

                        if "--no-browser" in command_args:
                            no_browser = True

                        if "--port" in command_args:
                            port_idx = command_args.index("--port")
                            if port_idx + 1 < len(command_args):
                                try:
                                    port = int(command_args[port_idx + 1])
                                except ValueError:
                                    typer.echo(f"Error: Invalid port value: {command_args[port_idx + 1]}")
                                    continue

                        visualize(output=output, no_browser=no_browser, port=port)
                    elif command == "verify":
                        verify()
                    elif command == "set":
                        param = command_args[0] if len(command_args) > 0 else None
                        value = command_args[1] if len(command_args) > 1 else None
                        set(param=param, value=value)
                    else:
                        typer.echo(f"Unknown command: {command}")
                        typer.echo("Type '/help' for available commands.")
                except Exception as e:
                    typer.echo(f"Error executing command: {str(e)}")
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
            # Display total cost if any tokens were used
            if session_costs["total_tokens"] > 0:
                typer.echo("\nSession Summary:")
                typer.echo(f"Total input tokens: {session_costs['total_input_tokens']}")
                typer.echo(f"Total output tokens: {session_costs['total_output_tokens']}")
                typer.echo(f"Total tokens: {session_costs['total_tokens']}")
                typer.echo(f"Total cost: ${session_costs['total_cost_usd']:.6f} USD")
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
