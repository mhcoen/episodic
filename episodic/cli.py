import typer
import shlex
import shutil
import os
import warnings
import io
import logging
from typing import Optional, List, Dict, Any, Tuple

# Disable tokenizer parallelism to avoid warnings in CLI context
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from contextlib import redirect_stdout, redirect_stderr
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML

from episodic.db import (
    insert_node, get_node, get_ancestry, initialize_db, 
    resolve_node_ref, get_head, set_head, database_exists,
    get_recent_nodes, store_topic, get_recent_topics,
    store_compression, get_compression_stats, update_topic_end_node
)
from episodic.llm import query_llm, query_with_context
from episodic.llm_config import get_current_provider, get_default_model, get_available_providers
from episodic.prompt_manager import PromptManager
from episodic.config import config
from episodic.configuration import *
from episodic.configuration import get_llm_color, get_system_color, get_prompt_color, get_model_context_limit, get_text_color, get_heading_color
from litellm import cost_per_token
from episodic.compression import queue_topic_for_compression, start_auto_compression, compression_manager
from episodic.topics import (
    detect_topic_change_separately, extract_topic_ollama, should_create_first_topic,
    build_conversation_segment, is_node_in_topic_range, count_nodes_in_topic,
    _display_topic_evolution, TopicManager
)
from episodic.conversation import (
    ConversationManager, handle_chat_message, get_session_costs,
    wrapped_text_print, wrapped_llm_print
)
from episodic.benchmark import (
    benchmark_operation, benchmark_resource, display_benchmark_summary,
    reset_benchmarks
)

# Set up logging
logger = logging.getLogger(__name__)

# Create a Typer app for command handling
app = typer.Typer(add_completion=False)

# Global variables to store state
current_node_id = None
default_model = DEFAULT_MODEL
default_system = DEFAULT_SYSTEM_MESSAGE
default_context_depth = DEFAULT_CONTEXT_DEPTH
default_semdepth = 4  # Default semantic depth for drift calculation

# Create conversation manager instance
conversation_manager = ConversationManager()




def wrapped_text_print_with_indent(text: str, indent_length: int, **typer_kwargs):
    """Print text with automatic wrapping accounting for an existing prefix indent."""
    # Check if wrapping is enabled
    if not config.get("text_wrap", True):
        typer.secho(str(text), **typer_kwargs)
        return
    
    # Calculate available width accounting for the prefix
    import shutil
    import textwrap
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    margin = 4
    max_width = 100  # Maximum line length for readability
    available_width = min(max_width, max(40, terminal_width - margin))
    wrap_width = max(20, available_width - indent_length)  # Ensure minimum width
    
    # Process text to preserve formatting while wrapping long lines
    lines = str(text).split('\n')
    wrapped_lines = []
    
    for i, line in enumerate(lines):
        # Detect indentation (spaces or tabs at start of line)
        stripped = line.lstrip()
        line_indent = line[:len(line) - len(stripped)]
        
        if len(line) <= wrap_width:
            # Line is short enough, keep as-is
            wrapped_lines.append(line)
        else:
            # Line is too long, wrap it - NO INDENTATION for continuation lines
            if stripped:
                wrapped = textwrap.fill(
                    stripped, 
                    width=wrap_width,
                    initial_indent=line_indent,
                    subsequent_indent=line_indent  # No extra indentation
                )
                wrapped_lines.append(wrapped)
            else:
                # Empty line - preserve as-is
                wrapped_lines.append(line)
    
    # Join the processed lines back together
    wrapped_text = '\n'.join(wrapped_lines)
    
    # Print with the specified formatting
    typer.secho(wrapped_text, **typer_kwargs)











def _parse_flag_value(args: List[str], flag_names: List[str]) -> Optional[str]:
    """Parse a flag value from command arguments.
    
    Args:
        args: List of command arguments
        flag_names: List of flag names to look for (e.g., ["--count", "-c"])
        
    Returns:
        The flag value if found, None otherwise
    """
    for flag in flag_names:
        if flag in args:
            flag_idx = args.index(flag)
            if flag_idx + 1 < len(args):
                return args[flag_idx + 1]
    return None

def _remove_flag_and_value(args: List[str], flag_names: List[str]) -> List[str]:
    """Remove a flag and its value from command arguments.
    
    Args:
        args: List of command arguments
        flag_names: List of flag names to look for (e.g., ["--parent", "-p"])
        
    Returns:
        Modified argument list with flag and value removed
    """
    result = args.copy()
    for flag in flag_names:
        if flag in result:
            flag_idx = result.index(flag)
            if flag_idx + 1 < len(result):
                # Remove both flag and value
                result.pop(flag_idx)  # Remove flag
                result.pop(flag_idx)  # Remove value (now at same index)
            else:
                # Remove just the flag if no value
                result.pop(flag_idx)
    return result

def _has_flag(args: List[str], flag_names: List[str]) -> bool:
    """Check if any of the specified flags are present in arguments.
    
    Args:
        args: List of command arguments
        flag_names: List of flag names to check for
        
    Returns:
        True if any flag is found, False otherwise
    """
    return any(flag in args for flag in flag_names)

def format_role_display(role: Optional[str]) -> str:
    """Format role information for display.
    
    Args:
        role: The role string from the node data
        
    Returns:
        A human-readable role description
    """
    if role == "assistant":
        return "Model (assistant)"
    elif role == "user":
        return "User"
    elif role == "system":
        return "System"
    else:
        return "Unknown"

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
            default_system = DEFAULT_SYSTEM_MESSAGE
        typer.echo(f"Database initialized with a default root node (ID: {root_short_id}, UUID: {root_node_id}).")
        typer.echo("Prompt has been restored to default.")
    else:
        typer.echo("Database initialized.")

@app.command()
def add(content: str, parent: Optional[str] = typer.Option(None, "--parent", "-p", help="Parent node ID")):
    """Add a new node with the given content."""
    global current_node_id

    if not content or not content.strip():
        typer.echo("Error: Content cannot be empty")
        return

    try:
        parent_id = resolve_node_ref(parent) if parent else current_node_id or get_head()
        node_id, short_id = insert_node(content, parent_id)
        set_head(node_id)
        current_node_id = node_id
        typer.echo(f"Added node {short_id} (UUID: {node_id})")
    except ValueError as e:
        typer.echo(f"Invalid parent node ID: {str(e)}")
    except Exception as e:
        typer.echo(f"Error adding node: {str(e)}")

@app.command()
def show(node_id: str):
    """Show details of a specific node."""
    if not node_id or not node_id.strip():
        typer.echo("Error: node_id is required")
        return
        
    try:
        resolved_id = resolve_node_ref(node_id)
        node = get_node(resolved_id)
        if node:
            # Display node ID with color
            typer.secho("Node ID: ", nl=False, fg=get_text_color())
            typer.secho(f"{node['short_id']}", nl=False, fg=get_system_color())
            typer.secho(" (UUID: ", nl=False, fg=get_text_color())
            typer.secho(f"{node['id']}", nl=False, fg=get_system_color())
            typer.secho(")", fg=get_text_color())
            
            # Display parent with color
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.secho("Parent: ", nl=False, fg=get_text_color())
                typer.secho(f"{parent_short_id}", nl=False, fg=get_system_color())
                typer.secho(" (UUID: ", nl=False, fg=get_text_color())
                typer.secho(f"{node['parent_id']}", nl=False, fg=get_system_color())
                typer.secho(")", fg=get_text_color())
            else:
                typer.secho("Parent: ", nl=False, fg=get_text_color())
                typer.secho("None", fg=get_system_color())

            # Display role information with color
            role = node.get('role')
            typer.secho("Role: ", nl=False, fg=get_text_color())
            typer.secho(f"{format_role_display(role)}", fg=get_system_color())

            # Display provider and model information with color
            provider = node.get('provider')
            model = node.get('model')
            if provider or model:
                model_info = f"{provider}/{model}" if provider and model else provider or model
                typer.secho("Model: ", nl=False, fg=get_text_color())
                typer.secho(f"{model_info}", fg=get_system_color())

            # Display message with wrapping and appropriate color based on role
            typer.secho("Message: ", nl=False, fg=get_text_color())
            role = node.get('role', 'user')
            message_prefix_length = len("Message: ")
            if role == 'user':
                wrapped_text_print_with_indent(node['content'], message_prefix_length, fg=get_text_color())
            else:  # assistant/LLM role
                wrapped_text_print_with_indent(node['content'], message_prefix_length, fg=get_llm_color())
        else:
            typer.echo(f"Node not found: {node_id}")
    except ValueError as e:
        typer.echo(f"Invalid node ID: {str(e)}")
    except Exception as e:
        typer.echo(f"Error retrieving node: {str(e)}")

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
            # Display node ID with color
            typer.secho("Node ID: ", nl=False, fg=get_text_color())
            typer.secho(f"{node['short_id']}", nl=False, fg=get_system_color())
            typer.secho(" (UUID: ", nl=False, fg=get_text_color())
            typer.secho(f"{node['id']}", nl=False, fg=get_system_color())
            typer.secho(")", fg=get_text_color())
            
            # Display parent with color
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.secho("Parent: ", nl=False, fg=get_text_color())
                typer.secho(f"{parent_short_id}", nl=False, fg=get_system_color())
                typer.secho(" (UUID: ", nl=False, fg=get_text_color())
                typer.secho(f"{node['parent_id']}", nl=False, fg=get_system_color())
                typer.secho(")", fg=get_text_color())
            else:
                typer.secho("Parent: ", nl=False, fg=get_text_color())
                typer.secho("None", fg=get_system_color())

            # Display role information with color
            role = node.get('role')
            typer.secho("Role: ", nl=False, fg=get_text_color())
            typer.secho(f"{format_role_display(role)}", fg=get_system_color())

            # Display provider and model information with color
            provider = node.get('provider')
            model = node.get('model')
            if provider or model:
                model_info = f"{provider}/{model}" if provider and model else provider or model
                typer.secho("Model: ", nl=False, fg=get_text_color())
                typer.secho(f"{model_info}", fg=get_system_color())

            # Display message with wrapping and appropriate color based on role
            typer.secho("Message: ", nl=False, fg=get_text_color())
            role = node.get('role', 'user')
            message_prefix_length = len("Message: ")
            if role == 'user':
                wrapped_text_print_with_indent(node['content'], message_prefix_length, fg=get_text_color())
            else:  # assistant/LLM role
                wrapped_text_print_with_indent(node['content'], message_prefix_length, fg=get_llm_color())
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

            # Display provider and model information if available
            provider = node.get('provider')
            model = node.get('model')
            if provider or model:
                model_info = f"{provider}/{model}" if provider and model else provider or model
                typer.echo(f"Model: {model_info}")

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
        except ValueError as e:
            typer.echo(f"Invalid node ID: {str(e)}")
        except Exception as e:
            typer.echo(f"Error changing current node: {str(e)}")

@app.command()
def list(count: int = typer.Option(DEFAULT_LIST_COUNT, "--count", "-c", help="Number of recent nodes to list")):
    """List recent nodes."""
    if count <= 0:
        typer.echo("Error: Count must be a positive integer")
        return
        
    try:
        # Get recent nodes
        nodes = get_recent_nodes(count)

        if not nodes:
            typer.echo("No nodes found in the database.")
            return

        typer.echo(f"Recent nodes (showing {len(nodes)} of {count} requested):")
        for node in nodes:
            # Get content and limit to exactly 3 display lines
            content = node['content']
            role = node.get('role', 'user')  # Default to user if role not specified
            
            # Use shorter prefix for better readability
            prefix = f"{node['short_id']}: "
            prefix_length = len(prefix)
            
            # Calculate available width using same logic as wrapped_text_print_with_indent
            terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
            margin = 4
            max_width = 100
            available_width = min(max_width, max(40, terminal_width - margin))
            wrap_width = max(20, available_width - prefix_length)
            
            # Simulate text wrapping to find content that fits in exactly 3 lines
            words = content.split()
            lines = []
            current_line = ""
            
            for word in words:
                # Check if adding this word would exceed line width
                test_line = current_line + (" " + word if current_line else word)
                if len(test_line) <= wrap_width:
                    current_line = test_line
                else:
                    # Line is full, start a new line
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        # Single word is too long, break it
                        lines.append(word[:wrap_width])
                        current_line = word[wrap_width:]
                    
                    # Stop if we have 3 lines
                    if len(lines) >= 3:
                        break
            
            # Add the last line if we have room
            if current_line and len(lines) < 3:
                lines.append(current_line)
            
            # Join lines and add ... if content was truncated
            if len(lines) >= 3 and len(words) > len(' '.join(lines).split()):
                limited_content = '\n'.join(lines[:3]) + '...'
            else:
                limited_content = '\n'.join(lines)
            
            # Display prefix in system color
            typer.secho(f"{node['short_id']}", nl=False, fg=get_system_color())
            typer.secho(f": ", nl=False, fg=get_text_color())
            
            # Display content with wrapping that accounts for prefix length
            if role == 'user':
                wrapped_text_print_with_indent(limited_content, prefix_length, fg=get_text_color())
            else:  # assistant/LLM role
                wrapped_text_print_with_indent(limited_content, prefix_length, fg=get_llm_color())
    except Exception as e:
        typer.echo(f"Error retrieving recent nodes: {str(e)}")

@app.command()
def cost():
    """Display the current session cost."""
    global session_costs
    
    typer.echo("Session Cost Summary:")
    typer.secho(f"  Input tokens: ", nl=False, fg=get_text_color())
    typer.secho(f"{session_costs['total_input_tokens']}", fg=get_system_color())
    typer.secho(f"  Output tokens: ", nl=False, fg=get_text_color())
    typer.secho(f"{session_costs['total_output_tokens']}", fg=get_system_color())
    typer.secho(f"  Total tokens: ", nl=False, fg=get_text_color())
    typer.secho(f"{session_costs['total_tokens']}", fg=get_system_color())
    typer.secho(f"  Total cost: ", nl=False, fg=get_text_color())
    typer.secho(f"${session_costs['total_cost_usd']:.{COST_PRECISION}f} USD", fg=get_system_color())

@app.command()
def handle_model(name: Optional[str] = None):
    """Show current model or change to a new one."""
    global default_model

    # Get the current model first
    if name is None:
        current_model = default_model
        typer.secho(f"Current model: ", nl=False, fg=get_text_color())
        typer.secho(f"{current_model}", nl=False, fg=get_system_color())
        typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
        typer.secho(f"{get_current_provider()}", nl=False, fg=get_system_color())
        typer.secho(")", fg=get_text_color())

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
                        # This is somewhat lacking as the same model (e.g., llama) may be provided by different cloud services.
                        try:
                            # Suppress both warnings and stdout/stderr output from LiteLLM during pricing lookup
                            with warnings.catch_warnings(), \
                                 redirect_stdout(io.StringIO()), \
                                 redirect_stderr(io.StringIO()):
                                warnings.simplefilter("ignore")
                                # Calculate cost for 1000 tokens (both input and output separately)
                                input_cost_raw = cost_per_token(model=model_name, prompt_tokens=PRICING_TOKEN_COUNT, completion_tokens=0)
                                output_cost_raw = cost_per_token(model=model_name, prompt_tokens=0, completion_tokens=PRICING_TOKEN_COUNT)

                            # Handle tuple results (sum if tuple, use directly if scalar)
                            input_cost = sum(input_cost_raw) if isinstance(input_cost_raw, tuple) else input_cost_raw
                            output_cost = sum(output_cost_raw) if isinstance(output_cost_raw, tuple) else output_cost_raw

                            if input_cost or output_cost:
                                pricing = f"${input_cost:.6f}/1K input, ${output_cost:.6f}/1K output"
                            else:
                                # For local providers, show "Local model" instead of "Pricing not available"
                                if provider_name in LOCAL_PROVIDERS:
                                    pricing = "Local model"
                                else:
                                    pricing = "Pricing not available"
                        except Exception:
                            # For local providers, show "Local model" instead of "Pricing not available"
                            if provider_name in LOCAL_PROVIDERS:
                                pricing = "Local model"
                            else:
                                pricing = "Pricing not available"

                        typer.secho(f"  ", nl=False, fg=get_text_color())
                        typer.secho(f"{current_idx:2d}", nl=False, fg=get_system_color())
                        typer.secho(f". ", nl=False, fg=get_text_color())
                        typer.secho(f"{model_name:20s}", nl=False, fg=get_system_color())
                        typer.secho(f"\t({pricing})", fg=get_text_color())
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
    if not node_id or not node_id.strip():
        typer.echo("Error: node_id is required")
        return
        
    try:
        resolved_id = resolve_node_ref(node_id)
        ancestry = get_ancestry(resolved_id)

        if not ancestry:
            typer.echo(f"No ancestry found for node: {node_id}")
            return

        for ancestor in ancestry:
            # Display ancestor with colors and wrapping (shorter prefix for readability)
            typer.secho(f"{ancestor['short_id']}", nl=False, fg=get_system_color())
            typer.secho(": ", nl=False, fg=get_text_color())
            
            # Use shorter prefix for better readability
            prefix_length = len(f"{ancestor['short_id']}: ")
            
            # Display content with role-based coloring and wrapping
            role = ancestor.get('role', 'user')
            if role == 'user':
                wrapped_text_print_with_indent(ancestor['content'], prefix_length, fg=get_text_color())
            else:  # assistant/LLM role
                wrapped_text_print_with_indent(ancestor['content'], prefix_length, fg=get_llm_color())
    except ValueError as e:
        typer.echo(f"Invalid node ID: {str(e)}")
    except Exception as e:
        typer.echo(f"Error retrieving ancestry: {str(e)}")

@app.command()
def visualize(output: Optional[str] = None, no_browser: bool = False, port: int = DEFAULT_VISUALIZATION_PORT):
    """Visualize the conversation DAG."""
    # Validate port number
    if port <= 0 or port > 65535:
        typer.echo("Error: Port must be between 1 and 65535")
        return
        
    try:
        from episodic.visualization import visualize_dag
        import webbrowser
        import os
        import time
        import signal
        import sys

        if not no_browser:
            # Start server first, then generate interactive visualization
            from episodic.server import start_server, stop_server
            server_url = start_server(server_port=port)
            typer.echo(f"Starting visualization server at {server_url}")
            
            # Generate interactive visualization that connects to the server
            output_path = visualize_dag(output, interactive=True, server_url=server_url)
            
            typer.echo("Press Ctrl+C when done to stop the server.")
            
            # Give the server a moment to fully start before opening browser
            time.sleep(1)
            
            typer.echo(f"Opening browser to: {server_url}")
            webbrowser.open(server_url)
            
            # Set up signal handler for clean shutdown
            def signal_handler(signum, frame):
                typer.echo("\nStopping server...")
                stop_server()
                typer.echo("Server stopped.")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)

            try:
                # Keep the server running until the user presses Ctrl+C
                while True:
                    time.sleep(MAIN_LOOP_SLEEP_INTERVAL)
            except KeyboardInterrupt:
                typer.echo("\nStopping server...")
                stop_server()
                typer.echo("Server stopped.")
            except Exception as e:
                typer.echo(f"\nError in server loop: {e}")
                stop_server()
        else:
            # Generate static visualization without interactive features
            output_path = visualize_dag(output, interactive=False)
            if output_path:
                typer.echo(f"Visualization saved to: {output_path}")
                typer.echo(f"Opening visualization in browser: {output_path}")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
    except ImportError as e:
        typer.echo(f"Visualization dependencies not available: {str(e)}")
    except Exception as e:
        typer.echo(f"Error generating visualization: {str(e)}")

@app.command()
def set(param: Optional[str] = None, value: Optional[str] = None):
    """Configure various parameters."""
    global default_context_depth, default_semdepth

    # If no parameter is provided, show all parameters and their current values
    if not param:
        typer.secho("Current parameters:", fg=get_heading_color(), bold=True)
        
        # Core conversation settings
        typer.secho("\nConversation:", fg=get_heading_color())
        typer.secho("  depth: ", nl=False, fg=get_text_color())
        typer.secho(f"{default_context_depth}", fg=get_system_color())
        typer.secho("  cache: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('use_context_cache', True)}", fg=get_system_color())
        
        # Display settings
        typer.secho("\nDisplay:", fg=get_heading_color())
        typer.secho("  color: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('color_mode', DEFAULT_COLOR_MODE)}", fg=get_system_color())
        typer.secho("  wrap: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('text_wrap', True)}", fg=get_system_color())
        typer.secho("  cost: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_cost', False)}", fg=get_system_color())
        typer.secho("  topics: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_topics', False)}", fg=get_system_color())
        
        # Analysis features
        typer.secho("\nAnalysis:", fg=get_heading_color())
        typer.secho("  drift: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_drift', True)}", fg=get_system_color())
        typer.secho("  semdepth: ", nl=False, fg=get_text_color())
        typer.secho(f"{default_semdepth}", fg=get_system_color())
        
        # Topic detection & compression
        typer.secho("\nTopic Management:", fg=get_heading_color())
        typer.secho("  topic_detection_model: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('topic_detection_model', 'ollama/llama3')}", fg=get_system_color())
        typer.secho("  auto_compress_topics: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('auto_compress_topics', True)}", fg=get_system_color())
        typer.secho("  compression_model: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('compression_model', 'ollama/llama3')}", fg=get_system_color())
        typer.secho("  compression_min_nodes: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('compression_min_nodes', 10)}", fg=get_system_color())
        typer.secho("  show_compression_notifications: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('show_compression_notifications', True)}", fg=get_system_color())
        
        # Performance monitoring
        typer.secho("\nPerformance:", fg=get_heading_color())
        typer.secho("  benchmark: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('benchmark', False)}", fg=get_system_color())
        typer.secho("  benchmark_display: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('benchmark_display', False)}", fg=get_system_color())
        
        # Debug
        typer.secho("\nDebugging:", fg=get_heading_color())
        typer.secho("  debug: ", nl=False, fg=get_text_color())
        typer.secho(f"{config.get('debug', False)}", fg=get_system_color())
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

    # Handle the 'drift' parameter
    elif param.lower() == "drift":
        if not value:
            # Toggle the current value if no value is provided
            current = config.get("show_drift", True)
            config.set("show_drift", not current)
            typer.echo(f"Drift display: {'ON' if not current else 'OFF'}")
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("show_drift", val)
            typer.echo(f"Drift display: {'ON' if val else 'OFF'}")

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

    # Handle the 'semdepth' parameter
    elif param.lower() == "semdepth":
        if not value:
            typer.echo(f"Current semantic depth: {default_semdepth}")
        else:
            try:
                semdepth = int(value)
                if semdepth < 1:
                    typer.echo("Semantic depth must be a positive integer")
                    return
                default_semdepth = semdepth
                typer.echo(f"Semantic depth set to {semdepth}")
            except ValueError:
                typer.echo("Invalid value for semdepth. Please provide a positive integer")

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
            if current:
                from episodic.llm import disable_cache
                disable_cache()
            else:
                from episodic.llm import enable_cache
                enable_cache()
        else:
            # Set to the provided value
            val = value.lower() in ["on", "true", "yes", "1"]
            if val:
                from episodic.llm import enable_cache
                enable_cache()
            else:
                from episodic.llm import disable_cache
                disable_cache()

    # Handle the 'topics' parameter
    elif param.lower() == "topics":
        if not value:
            current = config.get("show_topics", False)
            typer.echo(f"Current topics display: {current}")
        else:
            val = value.lower() in ["true", "1", "yes", "on"]
            config.set("show_topics", val)
            typer.echo(f"Topics display set to {val}")

    # Handle the 'color' parameter
    elif param.lower() == "color":
        if not value:
            current = config.get("color_mode", DEFAULT_COLOR_MODE)
            typer.echo(f"Current color mode: {current}")
            typer.echo("Available modes: dark, light")
        else:
            if value.lower() in ["dark", "light"]:
                config.set("color_mode", value.lower())
                typer.echo(f"Color mode set to {value.lower()}")
                typer.echo("Note: Restart the session to see color changes in the prompt")
            else:
                typer.echo("Invalid color mode. Available options: dark, light")

    # Handle the 'wrap' parameter
    elif param.lower() == "wrap":
        if not value:
            current = config.get("text_wrap", True)
            typer.echo(f"Current text wrapping: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("text_wrap", val)
            typer.echo(f"Text wrapping: {'ON' if val else 'OFF'}")

    # Handle the 'auto_compress_topics' parameter
    elif param.lower() == "auto_compress_topics":
        if not value:
            current = config.get("auto_compress_topics", True)
            typer.echo(f"Current auto compression: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("auto_compress_topics", val)
            typer.echo(f"Auto compression: {'ON' if val else 'OFF'}")
            # Restart compression manager if needed
            if val:
                from episodic.compression import start_auto_compression
                start_auto_compression()
            else:
                from episodic.compression import stop_auto_compression
                stop_auto_compression()

    # Handle the 'show_compression_notifications' parameter
    elif param.lower() == "show_compression_notifications":
        if not value:
            current = config.get("show_compression_notifications", True)
            typer.echo(f"Current compression notifications: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("show_compression_notifications", val)
            typer.echo(f"Compression notifications: {'ON' if val else 'OFF'}")

    # Handle the 'compression_min_nodes' parameter
    elif param.lower() == "compression_min_nodes":
        if not value:
            typer.echo(f"Current compression minimum nodes: {config.get('compression_min_nodes', 10)}")
        else:
            try:
                min_nodes = int(value)
                if min_nodes < 3:
                    typer.echo("Compression minimum nodes must be at least 3")
                    return
                config.set("compression_min_nodes", min_nodes)
                typer.echo(f"Compression minimum nodes set to {min_nodes}")
            except ValueError:
                typer.echo("Invalid value for compression_min_nodes. Please provide a positive integer >= 3")

    # Handle the 'compression_model' parameter
    elif param.lower() == "compression_model":
        if not value:
            typer.echo(f"Current compression model: {config.get('compression_model', 'ollama/llama3')}")
        else:
            config.set("compression_model", value)
            typer.echo(f"Compression model set to {value}")

    # Handle the 'topic_detection_model' parameter
    elif param.lower() == "topic_detection_model":
        if not value:
            typer.echo(f"Current topic detection model: {config.get('topic_detection_model', 'ollama/llama3')}")
        else:
            config.set("topic_detection_model", value)
            typer.echo(f"Topic detection model set to {value}")

    # Handle the 'benchmark' parameter
    elif param.lower() == "benchmark":
        if not value:
            current = config.get("benchmark", False)
            typer.echo(f"Current benchmark: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("benchmark", val)
            typer.echo(f"Benchmark: {'ON' if val else 'OFF'}")
            if not val:
                # Reset benchmarks when disabled
                reset_benchmarks()

    # Handle the 'benchmark_display' parameter
    elif param.lower() == "benchmark_display":
        if not value:
            current = config.get("benchmark_display", False)
            typer.echo(f"Current benchmark display: {'ON' if current else 'OFF'}")
        else:
            val = value.lower() in ["on", "true", "yes", "1"]
            config.set("benchmark_display", val)
            typer.echo(f"Benchmark display: {'ON' if val else 'OFF'}")

    # Handle unknown parameter
    else:
        typer.echo(f"Unknown parameter: {param}")
        typer.secho("Available parameters:", fg=get_text_color())
        typer.secho("  Conversation: ", nl=False, fg=get_heading_color())
        typer.secho("depth, cache", fg=get_system_color())
        typer.secho("  Display: ", nl=False, fg=get_heading_color())
        typer.secho("color, wrap, cost, topics", fg=get_system_color())
        typer.secho("  Analysis: ", nl=False, fg=get_heading_color())
        typer.secho("drift, semdepth", fg=get_system_color())
        typer.secho("  Topic Management: ", nl=False, fg=get_heading_color())
        typer.secho("topic_detection_model, auto_compress_topics, compression_model, compression_min_nodes, show_compression_notifications", fg=get_system_color())
        typer.secho("  Performance: ", nl=False, fg=get_heading_color())
        typer.secho("benchmark, benchmark_display", fg=get_system_color())
        typer.secho("  Debugging: ", nl=False, fg=get_heading_color())
        typer.secho("debug", fg=get_system_color())
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
def topics(
    count: int = typer.Option(10, "--count", "-c", help="Number of recent topics to show"),
    all_topics: bool = typer.Option(False, "--all", help="Show all topics instead of just recent ones")
):
    """Show recent conversation topics."""
    try:
        if all_topics:
            topic_list = get_recent_topics(limit=1000)  # Get all topics
            typer.echo("All conversation topics:")
        else:
            topic_list = get_recent_topics(limit=count)
            typer.echo(f"Recent topics (last {count}):")
        
        if not topic_list:
            typer.echo("No topics found. Topics are created when the LLM detects topic changes.")
            return
            
        typer.echo()
        
        # Calculate number width for proper alignment
        max_number = len(topic_list)
        number_width = len(str(max_number))
        
        for i, topic in enumerate(topic_list, 1):
            confidence = topic['confidence'] or 'unknown'
            node_count = count_nodes_in_topic(topic['start_node_id'], topic['end_node_id'])
            
            # Show topic info with proper alignment
            typer.secho(f"{i:>{number_width}}. ", nl=False, fg=get_system_color())
            typer.secho(f"{topic['name']:<25}", nl=False, fg=get_llm_color())
            typer.secho(f" (", nl=False, fg=get_text_color())
            typer.secho(f"{confidence}", nl=False, fg=get_system_color())
            typer.secho(f" confidence, ", nl=False, fg=get_text_color())
            typer.secho(f"{node_count}", nl=False, fg=get_system_color())
            typer.secho(f" nodes)", fg=get_text_color())
            
            # Indent the range and created lines to align with topic name
            indent = " " * (number_width + 2)  # Account for number + ". "
            typer.secho(f"{indent}Range: ", nl=False, fg=get_text_color())
            typer.secho(f"{topic['start_short_id']}", nl=False, fg=get_system_color())
            typer.secho(f" → ", nl=False, fg=get_text_color())
            typer.secho(f"{topic['end_short_id']}", fg=get_system_color())
            
            typer.secho(f"{indent}Created: ", nl=False, fg=get_text_color())
            typer.secho(f"{topic['created_at']}", fg=get_system_color())
            
            if i < len(topic_list):  # Don't add extra line after last item
                typer.echo()
                
    except Exception as e:
        typer.echo(f"Error retrieving topics: {e}")


@app.command()
def script(filename: str):
    """Run scripted conversation from a text file."""
    try:
        # Check if file exists
        if not os.path.exists(filename):
            typer.echo(f"Error: File '{filename}' not found.")
            return
            
        # Read the script file
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse queries (ignore empty lines and comments starting with #)
        queries = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)
        
        if not queries:
            typer.echo(f"No queries found in '{filename}'. Add one query per line, use # for comments.")
            return
            
        typer.echo(f"Running script '{filename}' with {len(queries)} queries...")
        typer.echo("=" * 50)
        
        # Process each query
        for i, query in enumerate(queries, 1):
            typer.echo(f"\n[{i}/{len(queries)}] > {query}")
            
            # Check if it's a command (starts with /) or a chat message
            # Strip leading whitespace to allow commands like "  /help"
            stripped_query = query.lstrip()
            if stripped_query.startswith('/'):
                # Process as a command (remove the leading /)
                should_exit = _handle_command(stripped_query[1:])
                if should_exit:
                    typer.echo("Script terminated by exit command.")
                    break
            else:
                # Process as a chat message
                _handle_chat_message(query)
            
        typer.echo("\n" + "=" * 50)
        typer.echo(f"Script completed! Processed {len(queries)} queries.")
        typer.echo("Use '/topics' to see extracted topics.")
        
    except Exception as e:
        typer.echo(f"Error running script: {e}")


@app.command()
def compress(
    branch_id: Optional[str] = typer.Argument(None, help="Node ID to compress branch from (defaults to current node)"),
    strategy: str = typer.Option("simple", "--strategy", "-s", help="Compression strategy: simple, key-moments"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be compressed without making changes")
):
    """Compress conversation history by summarizing branches."""
    global current_node_id
    
    try:
        # Determine which node to compress from
        if not branch_id:
            branch_id = current_node_id or get_head()
            if not branch_id:
                typer.echo("Error: No current node. Specify a node ID.")
                return
        else:
            branch_id = resolve_node_ref(branch_id)
        
        # Get the branch to compress
        branch_nodes = get_ancestry(branch_id)
        if not branch_nodes or len(branch_nodes) < 3:
            typer.echo("Error: Branch too short to compress (need at least 3 nodes).")
            return
        
        typer.echo(f"Compressing branch from root to {branch_nodes[-1]['short_id']}...")
        typer.echo(f"Branch contains {len(branch_nodes)} nodes")
        
        # Start timing compression
        import time
        compression_start_time = time.time()
        
        if strategy == "simple":
            # Simple strategy: Summarize the entire branch
            summary = _compress_branch_simple(branch_nodes, dry_run)
        elif strategy == "key-moments":
            # Key moments strategy: Extract important exchanges
            summary = _compress_branch_key_moments(branch_nodes, dry_run)
        else:
            typer.echo(f"Error: Unknown compression strategy '{strategy}'")
            return
        
        if dry_run:
            typer.echo("\n--- DRY RUN: No changes made ---")
            typer.echo("Summary that would be created:")
            wrapped_text_print(summary, fg=get_system_color())
        else:
            # Create a new compressed node
            content = f"[COMPRESSED BRANCH SUMMARY]\n\n{summary}"
            # Get the parent of the first node in the branch (usually root)
            parent_id = branch_nodes[0]['parent_id'] if branch_nodes else None
            compressed_id, compressed_short_id = insert_node(content, parent_id, role="system")
            
            # Update the last node to point to the compressed node
            # This is a simplified approach - in a full implementation we might
            # restructure the DAG more intelligently
            typer.echo(f"\nCreated compressed node: {compressed_short_id}")
            typer.echo(f"Original branch preserved. Use compressed node as parent for future conversations on this topic.")
            
            # Calculate compression metrics
            import time
            compression_end_time = time.time()
            compression_duration = compression_end_time - compression_start_time if 'compression_start_time' in locals() else 0
            
            # Token estimation (rough approximation: 1 token ≈ 0.75 words)
            original_words = sum(len(node['content'].split()) for node in branch_nodes)
            compressed_words = len(summary.split())
            original_tokens_est = int(original_words / 0.75)
            compressed_tokens_est = int(compressed_words / 0.75)
            
            ratio = (1 - compressed_words / original_words) * 100
            token_reduction = original_tokens_est - compressed_tokens_est
            
            typer.echo(f"\n📊 Compression Metrics:")
            typer.echo(f"  Compression ratio: {ratio:.1f}% reduction")
            typer.echo(f"  Words: {original_words:,} → {compressed_words:,}")
            typer.echo(f"  Tokens (est): {original_tokens_est:,} → {compressed_tokens_est:,} (saved ~{token_reduction:,} tokens)")
            typer.echo(f"  Compression time: {compression_duration:.2f}s")
            
            # Store compression metadata in database
            try:
                store_compression(
                    compressed_node_id=compressed_id,
                    original_branch_head=branch_nodes[-1]['id'],
                    original_node_count=len(branch_nodes),
                    original_words=original_words,
                    compressed_words=compressed_words,
                    compression_ratio=ratio,
                    strategy=strategy,
                    duration_seconds=compression_duration
                )
                typer.echo(f"\n💾 Compression metadata stored in database")
            except Exception as e:
                typer.echo(f"\n⚠️  Warning: Could not store compression metadata: {e}")
            
    except Exception as e:
        typer.echo(f"Error compressing branch: {e}")


def _compress_branch_simple(nodes: List[Dict], dry_run: bool = False) -> str:
    """Simple compression: summarize the entire conversation."""
    # Build conversation text
    conversation_parts = []
    for node in nodes:
        role = node.get('role', 'user')
        if role == 'system' and '[COMPRESSED' in node.get('content', ''):
            # Skip existing compressions
            continue
        role_name = "User" if role == "user" else "Assistant"
        conversation_parts.append(f"{role_name}: {node['content']}")
    
    full_conversation = "\n\n".join(conversation_parts)
    
    # Create summarization prompt
    prompt = f"""Please create a concise summary of the following conversation. 
Focus on key topics discussed, important decisions made, and any conclusions reached.
Preserve essential context that would be needed to resume this conversation later.

Conversation:
{full_conversation}

Summary:"""
    
    if dry_run:
        return "[Would generate summary using LLM]\n\nExample summary:\nThis conversation covered [topics]. Key points included [main ideas]. The discussion concluded with [outcomes]."
    
    # Use LLM to generate summary
    try:
        summary, _ = query_llm(
            prompt=prompt,
            model=get_default_model(),
            system_message="You are a helpful assistant that creates concise, informative summaries.",
            max_tokens=500
        )
        return summary
    except Exception as e:
        return f"[Error generating summary: {e}]"


def _compress_branch_key_moments(nodes: List[Dict], dry_run: bool = False) -> str:
    """Key moments compression: extract important exchanges with intelligent detection."""
    # Track topic changes and key moments
    key_moments = []
    topics_seen = set()
    previous_topics = None
    
    for i, node in enumerate(nodes):
        content = node.get('content', '').lower()
        role = node.get('role', 'user')
        
        # Detect topic changes (using simple keyword detection)
        current_topics = set()
        topic_keywords = {
            'python': ['python', 'decorator', 'function', 'class', 'import'],
            'ml': ['machine learning', 'deep learning', 'neural', 'ai', 'model'],
            'api': ['api', 'rest', 'graphql', 'endpoint', 'http'],
            'architecture': ['microservice', 'architecture', 'design', 'pattern'],
            'database': ['database', 'sql', 'query', 'table', 'schema'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in content for kw in keywords):
                current_topics.add(topic)
        
        # Determine if this is a key moment
        is_key = False
        reason = ""
        
        # Always include first and last nodes
        if i == 0:
            is_key = True
            reason = "conversation start"
        elif i == len(nodes) - 1:
            is_key = True
            reason = "conversation end"
        # Topic changes
        elif current_topics and current_topics != previous_topics:
            is_key = True
            reason = f"topic change to {', '.join(current_topics)}"
            topics_seen.update(current_topics)
        # Questions (especially from users)
        elif '?' in content and role == 'user':
            is_key = True
            reason = "user question"
        # Important markers
        elif any(marker in content for marker in [
            'important', 'key point', 'summary', 'conclusion', 
            'let me explain', 'the main', 'in summary', 'to summarize',
            'best practice', 'recommendation', 'solution'
        ]):
            is_key = True
            reason = "important content"
        # Code examples (useful to preserve)
        elif '```' in node.get('content', '') or 'def ' in node.get('content', ''):
            is_key = True
            reason = "code example"
        
        if is_key:
            key_moments.append({
                'node': node,
                'index': i,
                'reason': reason
            })
        
        previous_topics = current_topics
    
    if dry_run:
        summary = f"[Would extract {len(key_moments)} key moments from {len(nodes)} total nodes]\n\n"
        summary += "Key moments identified:\n"
        for km in key_moments[:10]:  # Show first 10
            summary += f"  - Node {km['node']['short_id']}: {km['reason']}\n"
        if len(key_moments) > 10:
            summary += f"  ... and {len(key_moments) - 10} more\n"
        summary += f"\nTopics covered: {', '.join(sorted(topics_seen)) if topics_seen else 'various'}"
        return summary
    
    # Build intelligent summary from key moments
    if not key_moments:
        return "No key moments identified in conversation."
    
    # Group by topics/sections
    summary_parts = [f"Conversation Summary ({len(key_moments)} key moments from {len(nodes)} total exchanges):\n"]
    
    # Add topic overview if detected
    if topics_seen:
        summary_parts.append(f"Topics discussed: {', '.join(sorted(topics_seen))}\n")
    
    # Add key moments with context
    for i, km in enumerate(key_moments):
        node = km['node']
        role = node.get('role', 'user')
        role_name = "User" if role == "user" else "Assistant"
        
        # Include reason for key moment
        summary_parts.append(f"[{km['reason'].title()}] {role_name}:")
        
        # Include more content for code examples, less for regular text
        if km['reason'] == "code example":
            summary_parts.append(node['content'][:500] + ("..." if len(node['content']) > 500 else ""))
        else:
            summary_parts.append(node['content'][:250] + ("..." if len(node['content']) > 250 else ""))
        
        summary_parts.append("")  # Empty line between moments
    
    return "\n".join(summary_parts)


@app.command()
def compression_stats():
    """Show compression statistics."""
    try:
        # Get manual compression stats from database
        stats = get_compression_stats()
        
        typer.echo("📊 Compression Statistics:")
        typer.echo(f"\nTotal compressions: {stats['total_compressions']}")
        
        if stats['total_compressions'] > 0:
            typer.echo(f"Total words saved: {stats['total_words_saved']:,}")
            typer.echo(f"Average compression ratio: {stats['average_compression_ratio']:.1f}%")
            
            if stats['strategies_used']:
                typer.echo("\nStrategies used:")
                for strategy, data in stats['strategies_used'].items():
                    typer.echo(f"  - {strategy}: {data['count']} times (avg {data['avg_ratio']:.1f}% reduction)")
        
        # Get async compression stats if available
        try:
            from episodic.compression import compression_manager
            if compression_manager:
                async_stats = compression_manager.get_stats()
                if async_stats['total_compressed'] > 0 or async_stats['failed_compressions'] > 0:
                    typer.echo("\nAuto-compression Statistics:")
                    typer.echo(f"  - Completed: {async_stats['total_compressed']}")
                    typer.echo(f"  - Failed: {async_stats['failed_compressions']}")
                    typer.echo(f"  - Words saved: {async_stats['total_words_saved']:,}")
                    typer.echo(f"  - Queue size: {async_stats['queue_size']}")
        except:
            pass  # Async compression not available or not started
            
        if stats['total_compressions'] == 0:
            typer.echo("\nNo compressions performed yet.")
            typer.echo("Use '/compress' to compress a conversation branch.")
            
    except Exception as e:
        typer.echo(f"Error retrieving compression stats: {e}")

@app.command()
def compress_current_topic():
    """Manually compress the current topic."""
    from episodic.compression import queue_topic_for_compression
    
    # Get the most recent topic
    recent_topics = get_recent_topics(limit=1)
    if not recent_topics:
        typer.echo("No topics found to compress.")
        return
    
    current_topic = recent_topics[0]
    typer.echo(f"Compressing current topic: '{current_topic['name']}'")
    typer.echo(f"  Start node: {current_topic['start_node_id']}")
    typer.echo(f"  End node: {current_topic['end_node_id']}")
    
    # Queue it for compression with high priority
    queue_topic_for_compression(
        current_topic['start_node_id'],
        current_topic['end_node_id'],
        current_topic['name'],
        priority=1  # High priority
    )
    
    typer.echo("Topic queued for compression.")


def compression_queue():
    """Show pending compression jobs in the queue."""
    try:
        queue_info = compression_manager.get_queue_info()
        stats = compression_manager.get_stats()
        
        typer.echo("\n📊 Compression Queue Status")
        typer.echo(f"Pending jobs: {len(queue_info)}")
        typer.echo(f"Total compressed: {stats['total_compressed']}")
        typer.echo(f"Failed compressions: {stats['failed_compressions']}")
        typer.echo(f"Total words saved: {stats['total_words_saved']:,}")
        
        if queue_info:
            typer.echo("\nPending jobs:")
            for job in queue_info:
                typer.echo(f"  - Topic: '{job['topic']}' (priority: {job['priority']}, attempts: {job['attempts']})")
        else:
            typer.echo("\nNo pending compression jobs.")
            
    except Exception as e:
        typer.echo(f"Error retrieving queue info: {e}")


def benchmark():
    """Display benchmark statistics for the current session."""
    display_benchmark_summary()


def help():
    """Show available commands."""
    cmd_color = get_system_color()
    desc_color = get_text_color()
    
    typer.secho("\n═══ BASIC COMMANDS ═══", fg=get_heading_color(), bold=True)
    typer.secho("  /help", fg=cmd_color, bold=True, nl=False)
    typer.secho("                - Show this help message", fg=desc_color)
    typer.secho("  /last [N]", fg=cmd_color, bold=True, nl=False)
    typer.secho("            - List recent nodes (default: 5)", fg=desc_color)
    typer.secho("  /model", fg=cmd_color, bold=True, nl=False)
    typer.secho("               - Show or change the current model", fg=desc_color)
    typer.secho("  /head [node_id]", fg=cmd_color, bold=True, nl=False)
    typer.secho("      - Show current node or change to specified node", fg=desc_color)
    
    typer.secho("\n═══ CONVERSATION MANAGEMENT ═══", fg=get_heading_color(), bold=True)
    typer.secho("  /add <content>", fg=cmd_color, bold=True, nl=False)
    typer.secho("       - Add a new node with the given content", fg=desc_color)
    typer.secho("  /show <node_id>", fg=cmd_color, bold=True, nl=False)
    typer.secho("      - Show details of a specific node", fg=desc_color)
    typer.secho("  /print [node_id]", fg=cmd_color, bold=True, nl=False)
    typer.secho("     - Print node info (defaults to current node)", fg=desc_color)
    typer.secho("  /ancestry <node_id>", fg=cmd_color, bold=True, nl=False)
    typer.secho("  - Trace the ancestry of a node", fg=desc_color)
    typer.secho("  /visualize", fg=cmd_color, bold=True, nl=False)
    typer.secho("           - Visualize the conversation DAG", fg=desc_color)
    
    typer.secho("\n═══ TOPICS & ORGANIZATION ═══", fg=get_heading_color(), bold=True)
    typer.secho("  /topics [N] [--all]", fg=cmd_color, bold=True, nl=False)
    typer.secho("  - Show recent conversation topics (default: 10)", fg=desc_color)
    typer.secho("  /compress [node_id]", fg=cmd_color, bold=True, nl=False)
    typer.secho("  - Compress conversation branch into summary", fg=desc_color)
    typer.secho("  /compress-current-topic", fg=cmd_color, bold=True, nl=False)
    typer.secho(" - Manually compress the current topic", fg=desc_color)
    typer.secho("  /compression-queue", fg=cmd_color, bold=True, nl=False)
    typer.secho("   - Show pending auto-compression jobs", fg=desc_color)
    typer.secho("  /compression-stats", fg=cmd_color, bold=True, nl=False)
    typer.secho("   - Show compression statistics", fg=desc_color)
    
    typer.secho("\n═══ CONFIGURATION & SETTINGS ═══", fg=get_heading_color(), bold=True)
    typer.secho("  /set [param] [value]", fg=cmd_color, bold=True, nl=False)
    typer.secho(" - Configure parameters (see /set for full list)", fg=desc_color)
    typer.secho("  /prompts", fg=cmd_color, bold=True, nl=False)
    typer.secho("             - Manage system prompts", fg=desc_color)
    typer.secho("  /verify", fg=cmd_color, bold=True, nl=False)
    typer.secho("              - Verify the current model with a test prompt", fg=desc_color)
    
    typer.secho("\n═══ ADVANCED & DIAGNOSTIC ═══", fg=get_heading_color(), bold=True)
    typer.secho("  /init [--erase]", fg=cmd_color, bold=True, nl=False)
    typer.secho("      - Initialize the database (--erase to erase existing)", fg=desc_color)
    typer.secho("  /script <filename>", fg=cmd_color, bold=True, nl=False)
    typer.secho("   - Run scripted conversation from text file", fg=desc_color)
    typer.secho("  /benchmark", fg=cmd_color, bold=True, nl=False)
    typer.secho("           - Show performance benchmark statistics", fg=desc_color)
    
    typer.secho("\n═══ EXIT ═══", fg=get_heading_color(), bold=True)
    typer.secho("  /exit", fg=cmd_color, bold=True, nl=False)
    typer.secho(", ", fg=desc_color, nl=False)
    typer.secho("/quit", fg=cmd_color, bold=True, nl=False)
    typer.secho("         - Exit the application", fg=desc_color)

    typer.secho("\nType a message without a leading ", fg=desc_color, nl=False)
    typer.secho("/", fg=cmd_color, bold=True, nl=False)
    typer.secho(" to chat with the LLM.", fg=desc_color)

# Main talk loop
def display_session_summary() -> None:
    """Display session summary with token usage and costs if any LLM interactions occurred."""
    session_costs = get_session_costs()
    if session_costs["total_tokens"] > 0:
        typer.echo("Session Summary:")
        typer.secho(f"Total input tokens: ", nl=False, fg=get_text_color())
        typer.secho(f"{session_costs['total_input_tokens']}", fg=get_system_color())
        typer.secho(f"Total output tokens: ", nl=False, fg=get_text_color())
        typer.secho(f"{session_costs['total_output_tokens']}", fg=get_system_color())
        typer.secho(f"Total tokens: ", nl=False, fg=get_text_color())
        typer.secho(f"{session_costs['total_tokens']}", fg=get_system_color())
        typer.secho(f"Total cost: ", nl=False, fg=get_text_color())
        typer.secho(f"${session_costs['total_cost_usd']:.{COST_PRECISION}f} USD", fg=get_system_color())
    else:
        typer.secho(f"Total cost: ", nl=False, fg=get_text_color())
        typer.secho(f"${0: .{ZERO_COST_PRECISION}f} USD", fg=get_system_color())
    
    # Display benchmark summary if enabled
    if config.get("benchmark", False):
        display_benchmark_summary()


def _initialize_talk_session() -> None:
    """Initialize the database and set up the talk session.
    
    Sets up the database if it doesn't exist and initializes the current node.
    """
    global current_node_id, default_model

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
    
    # Initialize conversation manager
    conversation_manager.set_current_node_id(current_node_id)
    
    # Start background compression manager if enabled
    if config.get('auto_compress_topics', True):
        start_auto_compression()

def _create_prompt_session() -> PromptSession:
    """Create and configure the prompt session for talk mode.
    
    Returns:
        A configured PromptSession with history and auto-suggestion
    """
    history_file = os.path.expanduser(config.get("history_file", DEFAULT_HISTORY_FILE))
    # Ensure the directory exists
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    return PromptSession(
        message=HTML(get_prompt_color()),
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
    )

def _initialize_prompt() -> None:
    """Initialize the default system prompt from the prompt manager.
    
    Loads the active prompt from configuration and updates the global default_system.
    """
    global default_system
    try:
        manager = PromptManager()
        default_system = manager.get_active_prompt_content(config.get)
    except Exception:
        # Fallback to default if there's an error
        default_system = DEFAULT_SYSTEM_MESSAGE

def _initialize_model() -> None:
    """Initialize and display the current model information.
    
    Ensures provider and model are properly matched and displays current selection.
    """
    global default_model
    from episodic.llm_config import get_current_provider, get_default_model, set_default_model, ensure_provider_matches_model

    # Ensure the provider matches the model
    ensure_provider_matches_model()

    # Get the current model
    current_model_name = get_default_model()

    # Set the default model to ensure proper initialization
    try:
        set_default_model(current_model_name)
        # Update the default model in the global variable
        default_model = current_model_name
    except ValueError:
        # If there's an error setting the model, just continue
        pass

    # Get the current provider and model after initialization
    provider = get_current_provider()
    current_model_name = get_default_model()
    typer.secho("Current model: ", nl=False, fg=get_text_color())
    typer.secho(f"{current_model_name}", nl=False, fg=get_system_color())
    typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
    typer.secho(f"{provider}", nl=False, fg=get_system_color())
    typer.secho(")", fg=get_text_color())

def _print_welcome_message() -> None:
    """Print the welcome message for talk mode."""
    typer.echo("Welcome to Episodic. You are now in talk mode.")
    typer.echo("Type a message to chat with the LLM, or use / to access commands.")
    typer.echo("For available commands, type ", nl=False)
    typer.secho("/help", fg=get_system_color())

def _handle_command(command_text: str) -> bool:
    """Handle a command input. 
    
    Args:
        command_text: The command text (without the leading /)
        
    Returns:
        True if the application should exit, False otherwise
    """
    # Handle exit command directly
    if command_text.lower() in EXIT_COMMANDS:
        typer.echo("\nExiting!")
        display_session_summary()
        typer.echo("Goodbye!")
        return True

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
            erase = _has_flag(command_args, ["--erase", "-e"])
            init(erase=erase)
        elif command == "add":
            if not command_args:
                typer.echo("Error: Missing content for add command")
                return False
            # Check for --parent flag
            parent = _parse_flag_value(command_args, ["--parent", "-p"])
            # Remove parent flag and value from args
            command_args = _remove_flag_and_value(command_args, ["--parent", "-p"])
            # Join remaining args as content
            content = " ".join(command_args)
            add(content=content, parent=parent)
        elif command == "show":
            if not command_args:
                typer.echo("Error: Missing node_id for show command")
                return False
            show(node_id=command_args[0])
        elif command == "head":
            node_id = command_args[0] if command_args else None
            head(node_id=node_id)
        elif command == "print":
            node_id = command_args[0] if command_args else None
            print_node(node_id=node_id)
        elif command == "last" or command == "list":  # Support both for backward compatibility
            # Check for --count flag first
            count_str = _parse_flag_value(command_args, ["--count", "-c"])
            if count_str:
                try:
                    count = int(count_str)
                except ValueError:
                    typer.echo(f"Error: Invalid count value: {count_str}")
                    return False
            elif command_args and command_args[0].isdigit():
                # Accept positional argument like "/last 10"
                try:
                    count = int(command_args[0])
                except ValueError:
                    typer.echo(f"Error: Invalid count value: {command_args[0]}")
                    return False
            else:
                count = DEFAULT_LIST_COUNT
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
                return False
            ancestry(node_id=command_args[0])
        elif command == "visualize":
            # Parse options
            output = _parse_flag_value(command_args, ["--output"])
            no_browser = _has_flag(command_args, ["--no-browser"])
            
            port_str = _parse_flag_value(command_args, ["--port"])
            if port_str:
                try:
                    port = int(port_str)
                except ValueError:
                    typer.echo(f"Error: Invalid port value: {port_str}")
                    return False
            else:
                port = DEFAULT_VISUALIZATION_PORT

            visualize(output=output, no_browser=no_browser, port=port)
        elif command == "set":
            param = command_args[0] if len(command_args) > 0 else None
            value = command_args[1] if len(command_args) > 1 else None
            set(param=param, value=value)
        elif command == "verify":
            verify()
        elif command == "topics":
            # Parse --all flag
            all_topics = "--all" in command_args
            
            # Check for --count flag or positional argument
            count_str = _parse_flag_value(command_args, ["--count", "-c"])
            if count_str:
                try:
                    count = int(count_str)
                except ValueError:
                    typer.echo(f"Error: Invalid count value: {count_str}")
                    return False
            elif command_args and command_args[0].isdigit():
                # Accept positional argument like "/topics 10"
                try:
                    count = int(command_args[0])
                except ValueError:
                    typer.echo(f"Error: Invalid count value: {command_args[0]}")
                    return False
            else:
                count = 10  # Default
            
            topics(count=count, all_topics=all_topics)
        elif command == "script":
            if len(command_args) == 0:
                typer.echo("Usage: /script <filename>")
                return False
            filename = command_args[0]
            script(filename=filename)
        elif command == "compress":
            # Parse arguments
            branch_id = command_args[0] if command_args else None
            
            # Parse flags
            strategy = _parse_flag_value(command_args, ["--strategy", "-s"]) or "simple"
            dry_run = "--dry-run" in command_args
            
            compress(branch_id=branch_id, strategy=strategy, dry_run=dry_run)
        elif command == "compression-queue":
            compression_queue()
        elif command == "compression-stats":
            compression_stats()
        elif command == "compress-current-topic":
            compress_current_topic()
        elif command == "cost":
            cost()
        elif command == "benchmark":
            benchmark()
        else:
            typer.echo(f"Unknown command: {command}")
            typer.echo("Type '/help' for available commands.")
    except Exception as e:
        typer.echo(f"Error executing command: {str(e)}")
    
    return False  # Don't exit

def _handle_chat_message(user_input: str) -> None:
    """Handle a chat message (non-command input).
    
    Args:
        user_input: The user's chat message
        
    Processes the message through the LLM and updates the conversation DAG.
    """
    global current_node_id
    
    try:
        # Use the conversation manager to handle the message
        assistant_node_id, display_response = handle_chat_message(
            user_input,
            model=default_model,
            system_message=default_system,
            context_depth=default_context_depth
        )
        
        # Update the current node
        current_node_id = assistant_node_id
        
    except Exception as e:
        typer.echo(f"Error querying LLM: {str(e)}")

def talk_loop() -> None:
    """Main talk loop that handles both conversation and commands.
    
    Initializes the session and runs the main interaction loop, handling both
    chat messages and commands until the user exits.
    """
    global current_node_id, default_model, default_system, default_context_depth, session_costs

    _initialize_talk_session()
    session = _create_prompt_session()
    _print_welcome_message()
    _initialize_prompt()
    _initialize_model()

    # Main loop
    while True:
        try:
            # Get user input
            user_input = session.prompt()

            # Skip empty input
            if not user_input.strip():
                continue

            # Check if it's a command (starts with /)
            # Strip leading whitespace to allow commands like "  /help"
            stripped_input = user_input.lstrip()
            if stripped_input.startswith('/'):
                # Remove the / prefix and parse the command
                command_text = stripped_input[1:].strip()
                if not command_text:
                    typer.echo("Empty command. Type '/help' for available commands.")
                    continue

                # Handle command and check if we should exit
                should_exit = _handle_command(command_text)
                if should_exit:
                    break
            else:
                # This is a chat message, not a command
                # Use the original input to preserve any intentional leading spaces in messages
                _handle_chat_message(user_input)
                
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

def main() -> None:
    """Main entry point for the CLI.
    
    Determines whether to run individual commands or start the interactive talk loop
    based on command-line arguments.
    """
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
