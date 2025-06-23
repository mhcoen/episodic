import typer
import shlex
import os
import warnings
import io
from typing import Optional, List, Dict, Any

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
    get_recent_nodes, store_topic, get_recent_topics
)
from episodic.llm import query_llm, query_with_context
from episodic.llm_config import get_current_provider, get_default_model, get_available_providers
from episodic.prompt_manager import PromptManager
from episodic.config import config
from episodic.configuration import *
from episodic.ml import ConversationalDrift
from litellm import cost_per_token

# Create a Typer app for command handling
app = typer.Typer(add_completion=False)

# Global variables to store state
current_node_id = None
default_model = DEFAULT_MODEL
default_system = DEFAULT_SYSTEM_MESSAGE
default_context_depth = DEFAULT_CONTEXT_DEPTH
default_semdepth = 4  # Default semantic depth for drift calculation
session_costs = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_tokens": 0,
    "total_cost_usd": 0.0
}

# Global drift calculator for real-time drift detection
drift_calculator = None

def _get_drift_calculator():
    """Get or create the global drift calculator instance."""
    global drift_calculator
    
    # Check if drift detection is disabled in config
    if not config.get("show_drift", True):
        return None
        
    if drift_calculator is None:
        try:
            drift_calculator = ConversationalDrift()
        except Exception as e:
            # If drift calculator fails to initialize (e.g., missing dependencies),
            # disable drift detection for this session
            if config.get("debug", False):
                typer.echo(f"⚠️  Drift detection disabled: {e}")
            drift_calculator = False  # Mark as disabled
    return drift_calculator if drift_calculator is not False else None

def _build_semdepth_context(nodes: List[Dict[str, Any]], semdepth: int, text_field: str = "content") -> str:
    """
    Build combined context from the last N nodes for semantic analysis.
    
    Args:
        nodes: List of conversation nodes in chronological order
        semdepth: Number of most recent nodes to combine
        text_field: Field containing text content
        
    Returns:
        Combined text from the last semdepth nodes
    """
    if not nodes or semdepth < 1:
        return ""
    
    # Get the last semdepth nodes
    context_nodes = nodes[-semdepth:] if len(nodes) >= semdepth else nodes
    
    # Combine their content
    combined_text = []
    for node in context_nodes:
        content = node.get(text_field, "").strip()
        if content:
            combined_text.append(content)
    
    return "\n".join(combined_text)

def _display_semantic_drift(current_user_node_id: str) -> None:
    """
    Calculate and display semantic drift between consecutive user messages.
    
    Only compares user inputs to detect when the user changes topics,
    ignoring assistant responses which just follow the user's lead.
    
    Args:
        current_user_node_id: ID of the current user message node
    """
    calc = _get_drift_calculator()
    if not calc:
        return  # Drift detection disabled
    
    try:
        # Get conversation history from root to current node
        conversation_chain = get_ancestry(current_user_node_id)
        
        # Filter to user messages only
        user_messages = [node for node in conversation_chain 
                        if node.get("role") == "user" and node.get("content", "").strip()]
        
        # Need at least 2 user messages for comparison
        if len(user_messages) < 2:
            if config.get("debug", False):
                typer.echo(f"   (Need 2 user messages for drift, have {len(user_messages)})")
            return
        
        # Compare current user message to previous user message
        current_user = user_messages[-1]
        previous_user = user_messages[-2]
        
        # Calculate semantic drift between consecutive user inputs
        drift_score = calc.calculate_drift(previous_user, current_user, text_field="content")
        
        # Format drift display based on score level
        if drift_score >= 0.8:
            drift_emoji = "🔄"
            drift_desc = "High topic shift"
        elif drift_score >= 0.6:
            drift_emoji = "📈"
            drift_desc = "Moderate drift"
        elif drift_score >= 0.3:
            drift_emoji = "➡️"
            drift_desc = "Low drift"
        else:
            drift_emoji = "🎯"
            drift_desc = "Minimal drift"
        
        # Display drift information
        prev_short_id = previous_user.get("short_id", "??")
        typer.echo(f"\n{drift_emoji} Semantic drift: {drift_score:.3f} ({drift_desc}) from user message {prev_short_id}")
        
        # Show additional context if debug mode is enabled
        if config.get("debug", False):
            prev_content = previous_user.get("content", "")[:80]
            curr_content = current_user.get("content", "")[:80]
            typer.echo(f"   Previous: {prev_content}{'...' if len(previous_user.get('content', '')) > 80 else ''}")
            typer.echo(f"   Current:  {curr_content}{'...' if len(current_user.get('content', '')) > 80 else ''}")
            
            # Show embedding cache efficiency
            cache_size = calc.get_cache_size()
            typer.echo(f"   Embedding cache: {cache_size} entries")
        
    except Exception as e:
        # If drift calculation fails, silently continue (don't disrupt conversation flow)
        if config.get("debug", False):
            typer.echo(f"⚠️  Drift calculation error: {e}")

def extract_topic_ollama(conversation_segment: str) -> Optional[str]:
    """
    Extract topic name from conversation segment using Ollama.
    
    Args:
        conversation_segment: Text containing recent conversation exchanges
        
    Returns:
        Topic name as lowercase string with hyphens, or None if extraction fails
    """
    try:
        prompt = f"""Extract the main topic from this conversation in 1-3 words. Use lowercase with hyphens.

Examples:
- Conversation about movies and directors → "movies"
- Discussion of quantum physics concepts → "quantum-physics" 
- Debugging code and performance → "programming"
- Talking about semantic drift → "semantic-drift"

Conversation: {conversation_segment}

Topic:"""

        # Use ollama for silent topic extraction
        response, _ = query_llm(prompt, model="ollama/llama3")
        
        if response:
            # Clean and normalize the response
            topic = response.strip().lower()
            # Remove quotes if present
            topic = topic.strip('"\'')
            # Replace spaces with hyphens
            topic = topic.replace(' ', '-')
            # Remove any extra characters, keep only letters, numbers, hyphens
            import re
            topic = re.sub(r'[^a-z0-9-]', '', topic)
            
            return topic if topic else None
            
    except Exception as e:
        if config.get("debug", False):
            typer.echo(f"⚠️  Topic extraction error: {e}")
        return None

def build_conversation_segment(nodes: List[Dict[str, Any]], max_length: int = 500) -> str:
    """
    Build a conversation segment for topic extraction.
    
    Args:
        nodes: List of conversation nodes
        max_length: Maximum character length for the segment
        
    Returns:
        Formatted conversation segment
    """
    segment_parts = []
    current_length = 0
    
    for node in reversed(nodes):  # Start from most recent
        content = node.get("content", "").strip()
        role = node.get("role", "unknown")
        
        if content:
            part = f"{role}: {content}"
            if current_length + len(part) > max_length:
                break
            segment_parts.insert(0, part)  # Insert at beginning to maintain order
            current_length += len(part)
    
    return "\n".join(segment_parts)

def detect_and_extract_topic_from_response(response: str, current_node_id: str) -> Optional[str]:
    """
    Detect topic changes from LLM response and extract topic if change detected.
    
    Args:
        response: The LLM's response text
        current_node_id: ID of the current conversation node
        
    Returns:
        Confidence level if topic change detected, None otherwise
    """
    try:
        # Check if response indicates topic change
        first_line = response.strip().split('\n')[0].lower()
        if not first_line.startswith('change-'):
            return None
            
        # Extract confidence level
        confidence = first_line.split('-')[1] if '-' in first_line else 'unknown'
        
        # Get recent conversation for topic extraction
        conversation_chain = get_ancestry(current_node_id)
        
        # Build conversation segment for topic extraction (last 5-10 exchanges)
        recent_nodes = conversation_chain[-10:] if len(conversation_chain) > 10 else conversation_chain
        conversation_segment = build_conversation_segment(recent_nodes)
        
        if conversation_segment:
            # Extract topic name
            topic_name = extract_topic_ollama(conversation_segment)
            
            if topic_name:
                # Find the start of this topic (for now, use the previous topic end or conversation start)
                # This is simplified - we could make it more sophisticated later
                previous_topics = get_recent_topics(limit=1)
                if previous_topics:
                    start_node_id = previous_topics[0]['end_node_id'] 
                else:
                    # First topic - start from beginning of current conversation segment
                    start_node_id = recent_nodes[0]['id'] if recent_nodes else current_node_id
                
                # Store the topic
                store_topic(topic_name, start_node_id, current_node_id, confidence)
                
                if config.get("debug", False):
                    typer.echo(f"   📝 Extracted topic: '{topic_name}' with confidence: {confidence}")
                
                return confidence
                
    except Exception as e:
        if config.get("debug", False):
            typer.echo(f"⚠️  Topic extraction error: {e}")
    
    return None

# Helper functions
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
            typer.echo(f"Node ID: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.echo(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                typer.echo(f"Parent: None")

            # Display role information
            role = node.get('role')
            typer.echo(f"Role: {format_role_display(role)}")

            # Display provider and model information if available
            provider = node.get('provider')
            model = node.get('model')
            if provider or model:
                model_info = f"{provider}/{model}" if provider and model else provider or model
                typer.echo(f"Model: {model_info}")

            typer.echo(f"Message: {node['content']}")
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
            typer.echo(f"Node ID: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.echo(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                typer.echo(f"Parent: None")

            # Display role information
            role = node.get('role')
            typer.echo(f"Role: {format_role_display(role)}")

            # Display provider and model information if available
            provider = node.get('provider')
            model = node.get('model')
            if provider or model:
                model_info = f"{provider}/{model}" if provider and model else provider or model
                typer.echo(f"Model: {model_info}")

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
            # Truncate content for display
            content = node['content']
            if len(content) > MAX_CONTENT_DISPLAY_LENGTH:
                content = content[:MAX_CONTENT_DISPLAY_LENGTH-3] + "..."

            # Display node information
            typer.echo(f"{node['short_id']} (UUID: {node['id']}): {content}")
    except Exception as e:
        typer.echo(f"Error retrieving recent nodes: {str(e)}")

@app.command()
def handle_model(name: Optional[str] = None):
    """Show current model or change to a new one."""
    global default_model

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
            typer.echo(f"{ancestor['short_id']} (UUID: {ancestor['id']}): {ancestor['content']}")
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
        typer.echo("Current parameters:")
        typer.echo(f"  cost: {config.get('show_cost', False)}")
        typer.echo(f"  drift: {config.get('show_drift', True)}")
        typer.echo(f"  depth: {default_context_depth}")
        typer.echo(f"  semdepth: {default_semdepth}")
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

    # Handle unknown parameter
    else:
        typer.echo(f"Unknown parameter: {param}")
        typer.echo("Available parameters: cost, drift, depth, semdepth, debug, cache")
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
def topics(all_topics: bool = typer.Option(False, "--all", help="Show all topics instead of just recent ones")):
    """Show recent conversation topics."""
    try:
        if all_topics:
            topic_list = get_recent_topics(limit=1000)  # Get all topics
            typer.echo("All conversation topics:")
        else:
            topic_list = get_recent_topics(limit=20)
            typer.echo("Recent topics (last 20):")
        
        if not topic_list:
            typer.echo("No topics found. Topics are created when the LLM detects topic changes.")
            return
            
        typer.echo()
        for i, topic in enumerate(topic_list):
            # Calculate topic age
            confidence = topic['confidence'] or 'unknown'
            confidence_emoji = {"high": "🔄", "medium": "📈", "low": "➡️"}.get(confidence, "📝")
            
            # Show topic info
            typer.echo(f"{confidence_emoji} {topic['name']:<20} ({confidence} confidence)")
            typer.echo(f"   Range: {topic['start_short_id']} → {topic['end_short_id']}")
            typer.echo(f"   Created: {topic['created_at']}")
            
            if i < len(topic_list) - 1:  # Don't add extra line after last item
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
            
            # Process the query using the same handler as manual input
            _handle_chat_message(query)
            
        typer.echo("\n" + "=" * 50)
        typer.echo(f"Script completed! Processed {len(queries)} queries.")
        typer.echo("Use '/topics' to see extracted topics.")
        
    except Exception as e:
        typer.echo(f"Error running script: {e}")


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
    typer.echo("  /set [param] [value] - Configure parameters (cost, drift, depth, semdepth, debug, cache)")
    typer.echo("  /prompts             - Manage system prompts")
    typer.echo("  /topics [--all]      - Show recent conversation topics")
    typer.echo("  /script <filename>   - Run scripted conversation from text file")

    typer.echo("\nType a message without a leading / to chat with the LLM.")

# Main talk loop
def display_session_summary() -> None:
    """Display session summary with token usage and costs if any LLM interactions occurred."""
    if session_costs["total_tokens"] > 0:
        typer.echo("Session Summary:")
        typer.echo(f"Total input tokens: {session_costs['total_input_tokens']}")
        typer.echo(f"Total output tokens: {session_costs['total_output_tokens']}")
        typer.echo(f"Total tokens: {session_costs['total_tokens']}")
        typer.echo(f"Total cost: ${session_costs['total_cost_usd']:.{COST_PRECISION}f} USD")
    else:
        typer.echo(f"Total cost: ${0: .{ZERO_COST_PRECISION}f} USD")


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

def _create_prompt_session() -> PromptSession:
    """Create and configure the prompt session for talk mode.
    
    Returns:
        A configured PromptSession with history and auto-suggestion
    """
    history_file = os.path.expanduser(config.get("history_file", DEFAULT_HISTORY_FILE))
    # Ensure the directory exists
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    return PromptSession(
        message=HTML(PROMPT_COLOR),
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
    typer.echo(f"Current model: {current_model_name} (Provider: {provider})")

def _print_welcome_message() -> None:
    """Print the welcome message for talk mode."""
    typer.echo("Welcome to Episodic. You are now in talk mode.")
    typer.echo("Type a message to chat with the LLM, or use / to access commands.")
    typer.echo("Examples: '/help', '/init', '/add Hello', '/exit'")

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
        elif command == "list":
            # Check for --count flag
            count_str = _parse_flag_value(command_args, ["--count", "-c"])
            if count_str:
                try:
                    count = int(count_str)
                except ValueError:
                    typer.echo(f"Error: Invalid count value: {count_str}")
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
            topics(all_topics=all_topics)
        elif command == "script":
            if len(command_args) == 0:
                typer.echo("Usage: /script <filename>")
                return False
            filename = command_args[0]
            script(filename=filename)
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
    global current_node_id, default_model, default_system, default_context_depth, session_costs
    
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

        # Calculate and display semantic drift if enabled
        if config.get("show_drift", True):
            _display_semantic_drift(user_node_id)

        # Detect topic changes and extract topics from LLM response
        topic_confidence = detect_and_extract_topic_from_response(response, user_node_id)
        display_response = response
        
        if topic_confidence:
            confidence_emoji = {"high": "🔄", "medium": "📈", "low": "➡️"}.get(topic_confidence, "📝")
            typer.echo(f"\n{confidence_emoji} Topic change detected ({topic_confidence} confidence)")
            
            # Remove the change indicator line from the displayed response
            lines = response.split('\n')
            if lines and lines[0].lower().startswith('change-'):
                display_response = '\n'.join(lines[1:]).strip()

        # Display cost information if enabled
        if config.get("show_cost", False) and cost_info:
            typer.echo(f"\n💰 Tokens: {cost_info.get('total_tokens', 0)} | Cost: ${cost_info.get('cost_usd', 0.0):.{COST_PRECISION}f} USD")

        # Display the response
        typer.echo(f"\n🤖 {display_response}")

        # Add the assistant's response to the database with provider and model information
        provider = get_current_provider()
        assistant_node_id, assistant_short_id = insert_node(
            response, 
            user_node_id, 
            role="assistant",
            provider=provider,
            model=default_model
        )

        # Update the current node to the assistant's response
        current_node_id = assistant_node_id
        set_head(assistant_node_id)

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
            if user_input.startswith('/'):
                # Remove the / prefix and parse the command
                command_text = user_input[1:].strip()
                if not command_text:
                    typer.echo("Empty command. Type '/help' for available commands.")
                    continue

                # Handle command and check if we should exit
                should_exit = _handle_command(command_text)
                if should_exit:
                    break
            else:
                # This is a chat message, not a command
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
