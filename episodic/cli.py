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
model_list = None  # List to store available models for number-based selection
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
    """Show current model or change to a new one."""
    global default_model, model_list

    # Get the current model first
    if name is None:
        current_model = default_model
        typer.echo(f"Current model: {current_model} (Provider: {get_current_provider()})")

        # Get provider models using our own configuration
        from episodic.llm_config import get_available_providers, get_provider_models
        try:
            providers = get_available_providers()
            current_idx = 1

            for provider_name, provider_config in providers.items():
                models = get_provider_models(provider_name)
                if models:
                    typer.echo(f"\nAvailable models from {provider_name}:")

                    for model in models:
                        if isinstance(model, dict):
                            model_name = model.get("name", "unknown")
                        else:
                            model_name = model

                        # Try to get pricing information using cost_per_token
                        # Use raw model name for pricing lookup (LiteLLM pricing doesn't use provider prefixes)
                        try:
                            import warnings
                            import sys
                            import io
                            from contextlib import redirect_stdout, redirect_stderr
                            
                            # Suppress both warnings and stdout/stderr output from LiteLLM during pricing lookup
                            with warnings.catch_warnings(), \
                                 redirect_stdout(io.StringIO()), \
                                 redirect_stderr(io.StringIO()):
                                warnings.simplefilter("ignore")
                                # Calculate cost for 1000 tokens (both input and output separately)
                                input_cost_raw = cost_per_token(model=model_name, prompt_tokens=1000, completion_tokens=0)
                                output_cost_raw = cost_per_token(model=model_name, prompt_tokens=0, completion_tokens=1000)

                            # Handle tuple results (sum if tuple, use directly if scalar)
                            input_cost = sum(input_cost_raw) if isinstance(input_cost_raw, tuple) else input_cost_raw
                            output_cost = sum(output_cost_raw) if isinstance(output_cost_raw, tuple) else output_cost_raw

                            if input_cost or output_cost:
                                pricing = f"${input_cost:.6f}/1K input, ${output_cost:.6f}/1K output"
                            else:
                                # For local providers, show "Local model" instead of "Pricing not available"
                                if provider_name in ["ollama", "lmstudio"]:
                                    pricing = "Local model"
                                else:
                                    pricing = "Pricing not available"
                        except Exception:
                            # For local providers, show "Local model" instead of "Pricing not available"
                            if provider_name in ["ollama", "lmstudio"]:
                                pricing = "Local model"
                            else:
                                pricing = "Pricing not available"

                        typer.echo(f"  {current_idx:2d}. {model_name:20s}\t({pricing})")
                        current_idx += 1

        except Exception as e:
            typer.echo(f"Error getting model list: {str(e)}")
        return

    # Handle model changes - check if input is a number first
    from episodic.llm_config import get_provider_models, set_default_model, get_available_providers
    
    # Check if the input is a number (model index)
    try:
        model_index = int(name)
        # Build the same model list to map index to model name
        providers = get_available_providers()
        current_idx = 1
        selected_model = None
        selected_provider = None
        
        for provider_name, provider_config in providers.items():
            models = get_provider_models(provider_name)
            if models:
                for model in models:
                    if isinstance(model, dict):
                        model_name = model.get("name", "unknown")
                    else:
                        model_name = model
                    
                    if current_idx == model_index:
                        selected_model = model_name
                        selected_provider = provider_name
                        break
                    current_idx += 1
                if selected_model:
                    break
        
        if selected_model:
            # Use the selected model name
            name = selected_model
        else:
            typer.echo(f"Error: Invalid model number '{model_index}'. Use '/model' to see available models.")
            return
            
    except ValueError:
        # Not a number, treat as model name
        pass
    
    # Now handle model change with the resolved model name
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
        typer.echo(f"  cache: {config.get('use_context_cache', True)}")
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

    # Handle the 'cache' parameter for context caching
    elif param.lower() == "cache":
        if not value:
            # Toggle the current value if no value is provided
            current = config.get("use_context_cache", True)
            config.set("use_context_cache", not current)
            typer.echo(f"Context caching: {'ON' if not current else 'OFF'}")
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("use_context_cache", val)
            typer.echo(f"Context caching: {'ON' if val else 'OFF'}")

    # Handle unknown parameter
    else:
        typer.echo(f"Unknown parameter: {param}")
        typer.echo("Available parameters: cost, depth, debug, cache")
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
    typer.echo("  /set [param] [value] - Configure parameters (cost, depth, debug, cache)")
    typer.echo("  /prompts             - Manage system prompts")

    typer.echo("\nType a message without a leading / to chat with the LLM.")

# Main talk loop
def display_session_summary():
    """Display session summary with token usage and costs if any LLM interactions occurred."""
    if session_costs["total_tokens"] > 0:
        typer.echo("Session Summary:")
        typer.echo(f"Total input tokens: {session_costs['total_input_tokens']}")
        typer.echo(f"Total output tokens: {session_costs['total_output_tokens']}")
        typer.echo(f"Total tokens: {session_costs['total_tokens']}")
        typer.echo(f"Total cost: ${session_costs['total_cost_usd']:.6f} USD")
    else:
        typer.echo(f"Total cost: ${0: .1f} USD")


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
                    typer.echo("\nExiting!")
                    display_session_summary()
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
                    elif command == "set":
                        param = command_args[0] if len(command_args) > 0 else None
                        value = command_args[1] if len(command_args) > 1 else None
                        set(param=param, value=value)
                    elif command == "verify":
                        verify()
                    else:
                        typer.echo(f"Unknown command: {command}")
                        typer.echo("Type '/help' for available commands.")
                except Exception as e:
                    typer.echo(f"Error executing command: {str(e)}")
            else:
                # This is a chat message, not a command
                # Add the user message to the database
                user_node_id, user_short_id = insert_node(user_input, current_node_id, role="user")

                # Query the LLM with context
                try:
                    # Get the context depth
                    depth = default_context_depth

                    # Query with context
                    response, cost_info = query_with_context(
                        user_node_id, 
                        model=default_model,
                        system_message=default_system,
                        context_depth=depth
                    )

                    # Update session costs
                    if cost_info:
                        session_costs["total_input_tokens"] += cost_info.get("input_tokens", 0)
                        session_costs["total_output_tokens"] += cost_info.get("output_tokens", 0)
                        session_costs["total_tokens"] += cost_info.get("total_tokens", 0)
                        session_costs["total_cost_usd"] += cost_info.get("cost_usd", 0.0)

                    # Display cost information if enabled
                    if config.get("show_cost", False) and cost_info:
                        typer.echo(f"\n💰 Tokens: {cost_info.get('total_tokens', 0)} | Cost: ${cost_info.get('cost_usd', 0.0):.6f} USD")

                    # Display the response
                    typer.echo(f"\n🤖 {response}")

                    # Add the assistant's response to the database
                    assistant_node_id, assistant_short_id = insert_node(response, user_node_id, role="assistant")

                    # Update the current node to the assistant's response
                    current_node_id = assistant_node_id
                    set_head(assistant_node_id)

                except Exception as e:
                    typer.echo(f"Error querying LLM: {str(e)}")
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            typer.echo("\nUse '/exit' or '/quit' to exit the application.")
        except EOFError:
            # Handle Ctrl+D gracefully (EOF)
            typer.echo("\nExiting!")
            display_session_summary()
            typer.echo("\nGoodbye!")
            break
        except Exception as e:
            typer.echo(f"Error: {str(e)}")

def set_default_model(model_name):
    """Set the default model for the CLI."""
    from episodic.llm_config import set_default_model as set_model
    set_model(model_name)

def main():
    """Main entry point for the CLI."""
    # Check if any command-line arguments were provided
    import sys
    if len(sys.argv) > 1:
        # If arguments were provided, run the Typer app
        app()
    else:
        # If no arguments were provided, start the talk loop
        talk_loop()

if __name__ == "__main__":
    main()
