import typer
import shlex
import os
import warnings
import io
import textwrap
import shutil
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
from episodic.configuration import get_llm_color, get_system_color, get_prompt_color, get_model_context_limit
from episodic.ml import ConversationalDrift
from litellm import cost_per_token
from episodic.compression import queue_topic_for_compression, start_auto_compression, compression_manager
from episodic.topics import (
    detect_topic_change_separately, extract_topic_ollama, should_create_first_topic,
    build_conversation_segment, is_node_in_topic_range, count_nodes_in_topic,
    _display_topic_evolution, TopicManager
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
session_costs = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_tokens": 0,
    "total_cost_usd": 0.0
}

# Global drift calculator for real-time drift detection
drift_calculator = None

def _get_wrap_width():
    """Get the appropriate text wrapping width for the terminal."""
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    margin = 4
    max_width = 100  # Maximum line length for readability
    return min(max_width, max(40, terminal_width - margin))

def wrapped_text_print(text: str, **typer_kwargs):
    """Print text with automatic wrapping while preserving formatting."""
    # Check if wrapping is enabled
    if not config.get("text_wrap", True):
        typer.secho(str(text), **typer_kwargs)
        return
    
    wrap_width = _get_wrap_width()
    
    # Process text to preserve formatting while wrapping long lines
    lines = str(text).split('\n')
    wrapped_lines = []
    
    for line in lines:
        if len(line) <= wrap_width:
            # Line is short enough, keep as-is
            wrapped_lines.append(line)
        else:
            # Line is too long, wrap it while preserving indentation
            # Detect indentation (spaces or tabs at start of line)
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            
            # Wrap the content while preserving indentation
            if stripped:  # Only wrap if there's actual content
                wrapped = textwrap.fill(
                    stripped, 
                    width=wrap_width,
                    initial_indent=indent,
                    subsequent_indent=indent + "  "  # Add slight extra indent for continuation
                )
                wrapped_lines.append(wrapped)
            else:
                # Empty or whitespace-only line, keep as-is
                wrapped_lines.append(line)
    
    # Join the processed lines back together
    wrapped_text = '\n'.join(wrapped_lines)
    
    # Print with the specified formatting
    typer.secho(wrapped_text, **typer_kwargs)

def wrapped_llm_print(text: str, **typer_kwargs):
    """Print LLM text with automatic wrapping while preserving formatting."""
    wrapped_text_print(text, **typer_kwargs)

def wrapped_text_print_with_indent(text: str, indent_length: int, **typer_kwargs):
    """Print text with automatic wrapping accounting for an existing prefix indent."""
    # Check if wrapping is enabled
    if not config.get("text_wrap", True):
        typer.secho(str(text), **typer_kwargs)
        return
    
    # Calculate available width accounting for the prefix
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
        typer.secho(f"\n{drift_emoji} Semantic drift: {drift_score:.3f} ({drift_desc}) from user message {prev_short_id}", fg=get_system_color())
        
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






def detect_and_extract_topic_from_response(response: str, user_node_id: str, assistant_node_id: str) -> Optional[str]:
    """
    Detect topic changes from LLM response and extract topic if change detected.
    
    Args:
        response: The LLM's response text
        user_node_id: ID of the user's question node that triggered this response
        assistant_node_id: ID of the assistant's response node
        
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
        
        # Get full conversation history
        conversation_chain = get_ancestry(assistant_node_id)
        
        if conversation_chain:
            # Find previous topics to determine boundaries
            previous_topics = get_recent_topics(limit=1)
            
            if previous_topics:
                # We have a previous topic - update its end boundary and create new topic
                prev_topic = previous_topics[0]
                
                # Find the parent of the topic-changing user node (last node of previous topic)
                user_node = get_node(user_node_id)
                if user_node and user_node.get('parent_id'):
                    # Update the previous topic to end at the node before the topic change
                    update_topic_end_node(prev_topic['name'], prev_topic['start_node_id'], user_node.get('parent_id'))
                    
                    # Queue previous topic for compression if auto-compression is enabled
                    if config.get('auto_compress_topics', True):
                        queue_topic_for_compression(
                            prev_topic['start_node_id'],
                            user_node.get('parent_id'),
                            prev_topic['name']
                        )
                
                # Extract topic name for the new topic from just the new exchange
                new_topic_nodes = []
                for i, node in enumerate(conversation_chain):
                    if node.get('id') == user_node_id:
                        # Include nodes from topic change onwards
                        new_topic_nodes = conversation_chain[i:]
                        break
                
                if new_topic_nodes:
                    # Extract topic from the new topic nodes (limit to first few for clarity)
                    new_topic_segment = build_conversation_segment(new_topic_nodes[:4], max_length=800)
                    topic_name = extract_topic_ollama(new_topic_segment)
                    
                    if topic_name:
                        # Store the new topic starting from the topic-changing user message
                        store_topic(topic_name, user_node_id, assistant_node_id, confidence)
            else:
                # This is the first topic change - need to retroactively create the first topic
                if len(conversation_chain) > 2:  # Need at least a few nodes
                    # Find all nodes before the topic change
                    prev_topic_nodes = []
                    for i, node in enumerate(conversation_chain):
                        if node.get('id') == user_node_id:
                            prev_topic_nodes = conversation_chain[:i]
                            break
                    
                    if prev_topic_nodes:
                        # Extract topic from all nodes of the previous conversation
                        prev_segment = build_conversation_segment(prev_topic_nodes, max_length=2000)
                        prev_topic_name = extract_topic_ollama(prev_segment)
                        
                        if prev_topic_name:
                            # Find first user node as start
                            first_user_node = None
                            last_node_before_change = None
                            
                            for node in prev_topic_nodes:
                                if node.get('role') == 'user' and first_user_node is None:
                                    first_user_node = node
                                last_node_before_change = node
                            
                            if first_user_node and last_node_before_change:
                                # Store the previous topic with all its nodes
                                store_topic(prev_topic_name, first_user_node['id'], last_node_before_change['id'], 'retroactive')
                
                # Now extract and store the new topic
                new_topic_nodes = []
                for i, node in enumerate(conversation_chain):
                    if node.get('id') == user_node_id:
                        new_topic_nodes = conversation_chain[i:]
                        break
                
                if new_topic_nodes:
                    # Extract topic from new nodes
                    new_topic_segment = build_conversation_segment(new_topic_nodes[:4], max_length=800)
                    topic_name = extract_topic_ollama(new_topic_segment)
                    
                    if topic_name:
                        # Store the new topic
                        store_topic(topic_name, user_node_id, assistant_node_id, confidence)
                
                if config.get("debug", False):
                    typer.echo(f"   📝 Extracted topic: '{topic_name}' with confidence: {confidence}")
                
            return confidence
                
    except Exception as e:
        if config.get("debug", False):
            typer.echo(f"⚠️  Topic extraction error: {e}")
    
    return None



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
            typer.secho("Node ID: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{node['short_id']}", nl=False, fg=get_system_color())
            typer.secho(" (UUID: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{node['id']}", nl=False, fg=get_system_color())
            typer.secho(")", fg=typer.colors.WHITE)
            
            # Display parent with color
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.secho("Parent: ", nl=False, fg=typer.colors.WHITE)
                typer.secho(f"{parent_short_id}", nl=False, fg=get_system_color())
                typer.secho(" (UUID: ", nl=False, fg=typer.colors.WHITE)
                typer.secho(f"{node['parent_id']}", nl=False, fg=get_system_color())
                typer.secho(")", fg=typer.colors.WHITE)
            else:
                typer.secho("Parent: ", nl=False, fg=typer.colors.WHITE)
                typer.secho("None", fg=get_system_color())

            # Display role information with color
            role = node.get('role')
            typer.secho("Role: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{format_role_display(role)}", fg=get_system_color())

            # Display provider and model information with color
            provider = node.get('provider')
            model = node.get('model')
            if provider or model:
                model_info = f"{provider}/{model}" if provider and model else provider or model
                typer.secho("Model: ", nl=False, fg=typer.colors.WHITE)
                typer.secho(f"{model_info}", fg=get_system_color())

            # Display message with wrapping and appropriate color based on role
            typer.secho("Message: ", nl=False, fg=typer.colors.WHITE)
            role = node.get('role', 'user')
            message_prefix_length = len("Message: ")
            if role == 'user':
                wrapped_text_print_with_indent(node['content'], message_prefix_length, fg=typer.colors.WHITE)
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
            typer.secho("Node ID: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{node['short_id']}", nl=False, fg=get_system_color())
            typer.secho(" (UUID: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{node['id']}", nl=False, fg=get_system_color())
            typer.secho(")", fg=typer.colors.WHITE)
            
            # Display parent with color
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                typer.secho("Parent: ", nl=False, fg=typer.colors.WHITE)
                typer.secho(f"{parent_short_id}", nl=False, fg=get_system_color())
                typer.secho(" (UUID: ", nl=False, fg=typer.colors.WHITE)
                typer.secho(f"{node['parent_id']}", nl=False, fg=get_system_color())
                typer.secho(")", fg=typer.colors.WHITE)
            else:
                typer.secho("Parent: ", nl=False, fg=typer.colors.WHITE)
                typer.secho("None", fg=get_system_color())

            # Display role information with color
            role = node.get('role')
            typer.secho("Role: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{format_role_display(role)}", fg=get_system_color())

            # Display provider and model information with color
            provider = node.get('provider')
            model = node.get('model')
            if provider or model:
                model_info = f"{provider}/{model}" if provider and model else provider or model
                typer.secho("Model: ", nl=False, fg=typer.colors.WHITE)
                typer.secho(f"{model_info}", fg=get_system_color())

            # Display message with wrapping and appropriate color based on role
            typer.secho("Message: ", nl=False, fg=typer.colors.WHITE)
            role = node.get('role', 'user')
            message_prefix_length = len("Message: ")
            if role == 'user':
                wrapped_text_print_with_indent(node['content'], message_prefix_length, fg=typer.colors.WHITE)
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
            typer.secho(f": ", nl=False, fg=typer.colors.WHITE)
            
            # Display content with wrapping that accounts for prefix length
            if role == 'user':
                wrapped_text_print_with_indent(limited_content, prefix_length, fg=typer.colors.WHITE)
            else:  # assistant/LLM role
                wrapped_text_print_with_indent(limited_content, prefix_length, fg=get_llm_color())
    except Exception as e:
        typer.echo(f"Error retrieving recent nodes: {str(e)}")

@app.command()
def cost():
    """Display the current session cost."""
    global session_costs
    
    typer.echo("Session Cost Summary:")
    typer.secho(f"  Input tokens: ", nl=False, fg=typer.colors.WHITE)
    typer.secho(f"{session_costs['total_input_tokens']}", fg=get_system_color())
    typer.secho(f"  Output tokens: ", nl=False, fg=typer.colors.WHITE)
    typer.secho(f"{session_costs['total_output_tokens']}", fg=get_system_color())
    typer.secho(f"  Total tokens: ", nl=False, fg=typer.colors.WHITE)
    typer.secho(f"{session_costs['total_tokens']}", fg=get_system_color())
    typer.secho(f"  Total cost: ", nl=False, fg=typer.colors.WHITE)
    typer.secho(f"${session_costs['total_cost_usd']:.{COST_PRECISION}f} USD", fg=get_system_color())

@app.command()
def handle_model(name: Optional[str] = None):
    """Show current model or change to a new one."""
    global default_model

    # Get the current model first
    if name is None:
        current_model = default_model
        typer.secho(f"Current model: ", nl=False, fg=typer.colors.WHITE)
        typer.secho(f"{current_model}", nl=False, fg=get_system_color())
        typer.secho(f" (Provider: ", nl=False, fg=typer.colors.WHITE)
        typer.secho(f"{get_current_provider()}", nl=False, fg=get_system_color())
        typer.secho(")", fg=typer.colors.WHITE)

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

                        typer.secho(f"  ", nl=False, fg=typer.colors.WHITE)
                        typer.secho(f"{current_idx:2d}", nl=False, fg=get_system_color())
                        typer.secho(f". ", nl=False, fg=typer.colors.WHITE)
                        typer.secho(f"{model_name:20s}", nl=False, fg=get_system_color())
                        typer.secho(f"\t({pricing})", fg=typer.colors.WHITE)
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
            typer.secho(": ", nl=False, fg=typer.colors.WHITE)
            
            # Use shorter prefix for better readability
            prefix_length = len(f"{ancestor['short_id']}: ")
            
            # Display content with role-based coloring and wrapping
            role = ancestor.get('role', 'user')
            if role == 'user':
                wrapped_text_print_with_indent(ancestor['content'], prefix_length, fg=typer.colors.WHITE)
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
        typer.echo("Current parameters:")
        typer.echo(f"  cost: {config.get('show_cost', False)}")
        typer.echo(f"  drift: {config.get('show_drift', True)}")
        typer.echo(f"  depth: {default_context_depth}")
        typer.echo(f"  semdepth: {default_semdepth}")
        typer.echo(f"  debug: {config.get('debug', False)}")
        typer.echo(f"  cache: {config.get('use_context_cache', True)}")
        typer.echo(f"  topics: {config.get('show_topics', False)}")
        typer.echo(f"  color: {config.get('color_mode', DEFAULT_COLOR_MODE)}")
        typer.echo(f"  wrap: {config.get('text_wrap', True)}")
        typer.echo(f"  auto_compress_topics: {config.get('auto_compress_topics', True)}")
        typer.echo(f"  show_compression_notifications: {config.get('show_compression_notifications', True)}")
        typer.echo(f"  compression_min_nodes: {config.get('compression_min_nodes', 10)}")
        typer.echo(f"  compression_model: {config.get('compression_model', 'ollama/llama3')}")
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

    # Handle unknown parameter
    else:
        typer.echo(f"Unknown parameter: {param}")
        typer.echo("Available parameters: cost, drift, depth, semdepth, debug, cache, topics, color, wrap, auto_compress_topics, show_compression_notifications, compression_min_nodes, compression_model")
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
            typer.secho(f" (", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{confidence}", nl=False, fg=get_system_color())
            typer.secho(f" confidence, ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{node_count}", nl=False, fg=get_system_color())
            typer.secho(f" nodes)", fg=typer.colors.WHITE)
            
            # Indent the range and created lines to align with topic name
            indent = " " * (number_width + 2)  # Account for number + ". "
            typer.secho(f"{indent}Range: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{topic['start_short_id']}", nl=False, fg=get_system_color())
            typer.secho(f" → ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{topic['end_short_id']}", fg=get_system_color())
            
            typer.secho(f"{indent}Created: ", nl=False, fg=typer.colors.WHITE)
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
            if query.startswith('/'):
                # Process as a command (remove the leading /)
                should_exit = _handle_command(query[1:])
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
    typer.echo("  /last [N]            - List recent nodes (default: 5)")
    typer.echo("  /ancestry <node_id>  - Trace the ancestry of a node")
    typer.echo("  /visualize           - Visualize the conversation DAG")
    typer.echo("  /model               - Show or change the current model")
    typer.echo("  /verify              - Verify the current model with a test prompt")
    wrapped_text_print("  /set [param] [value] - Configure parameters (cost, drift, depth, semdepth, debug, cache, topics, color, wrap, auto_compress_topics, show_compression_notifications, compression_min_nodes, compression_model)")
    typer.echo("  /prompts             - Manage system prompts")
    typer.echo("  /topics [N] [--all]  - Show recent conversation topics (default: 10)")
    typer.echo("  /script <filename>   - Run scripted conversation from text file")
    typer.echo("  /compress [node_id]  - Compress conversation branch into summary")
    typer.echo("  /compression-queue   - Show pending auto-compression jobs")
    typer.echo("  /compression-stats   - Show compression statistics")
    typer.echo("  /compress-current-topic - Manually compress the current topic")

    typer.echo("\nType a message without a leading / to chat with the LLM.")

# Main talk loop
def display_session_summary() -> None:
    """Display session summary with token usage and costs if any LLM interactions occurred."""
    if session_costs["total_tokens"] > 0:
        typer.echo("Session Summary:")
        typer.secho(f"Total input tokens: ", nl=False, fg=typer.colors.WHITE)
        typer.secho(f"{session_costs['total_input_tokens']}", fg=get_system_color())
        typer.secho(f"Total output tokens: ", nl=False, fg=typer.colors.WHITE)
        typer.secho(f"{session_costs['total_output_tokens']}", fg=get_system_color())
        typer.secho(f"Total tokens: ", nl=False, fg=typer.colors.WHITE)
        typer.secho(f"{session_costs['total_tokens']}", fg=get_system_color())
        typer.secho(f"Total cost: ", nl=False, fg=typer.colors.WHITE)
        typer.secho(f"${session_costs['total_cost_usd']:.{COST_PRECISION}f} USD", fg=get_system_color())
    else:
        typer.secho(f"Total cost: ", nl=False, fg=typer.colors.WHITE)
        typer.secho(f"${0: .{ZERO_COST_PRECISION}f} USD", fg=get_system_color())


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
    typer.secho(f"Current model: ", nl=False, fg=typer.colors.WHITE)
    typer.secho(f"{current_model_name}", nl=False, fg=get_system_color())
    typer.secho(f" (Provider: ", nl=False, fg=typer.colors.WHITE)
    typer.secho(f"{provider}", nl=False, fg=get_system_color())
    typer.secho(")", fg=typer.colors.WHITE)

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
    
    # Detect topic change BEFORE querying the main LLM
    topic_changed = False
    new_topic_name = None
    
    # Get recent messages for context
    recent_nodes = get_recent_nodes(limit=10)  # Get last 10 nodes for context
    if recent_nodes and len(recent_nodes) >= 2:  # Need at least some history
        topic_changed, new_topic_name = detect_topic_change_separately(recent_nodes, user_input)
        
        if config.get("debug", False) and topic_changed:
            typer.echo(f"\n🔍 DEBUG: Topic change detected before LLM query")
            typer.echo(f"   New topic: {new_topic_name}")

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

        # Use the response directly (no need to check for topic change indicators)
        display_response = response
        
        # Collect all status messages to display in one block
        status_messages = []

        # Add cost information if enabled
        if config.get("show_cost", False) and cost_info:
            # Calculate context usage
            current_tokens = cost_info.get('input_tokens', 0)  # Use input tokens for context calculation
            context_limit = get_model_context_limit(default_model)
            context_percentage = (current_tokens / context_limit) * 100
            
            # Format context percentage with appropriate precision
            if context_percentage < 1.0:
                context_display = f"{context_percentage:.1f}%"
            else:
                context_display = f"{int(context_percentage)}%"
            
            status_messages.append(f"Tokens: {cost_info.get('total_tokens', 0)} | Cost: ${cost_info.get('cost_usd', 0.0):.{COST_PRECISION}f} USD | Context: {context_display} full")

        # Display the response block with proper spacing
        if status_messages:
            # Show blank line, then status messages, then LLM response
            typer.echo("")
            for msg in status_messages:
                typer.secho(msg, fg=get_system_color())
            wrapped_llm_print(f"🤖 {display_response}", fg=get_llm_color())
        else:
            # No status messages, just show blank line then LLM response
            typer.echo("")  # Blank line
            wrapped_llm_print(f"🤖 {display_response}", fg=get_llm_color())

        # Add the assistant's response to the database with provider and model information
        provider = get_current_provider()
        assistant_node_id, assistant_short_id = insert_node(
            display_response, 
            user_node_id, 
            role="assistant",
            provider=provider,
            model=default_model
        )

        # Update the current node to the assistant's response
        current_node_id = assistant_node_id
        set_head(assistant_node_id)
        
        # Process topic changes based on our earlier detection
        if topic_changed and new_topic_name:
            # End the previous topic if one exists
            recent_topics = get_recent_topics(limit=1)
            if recent_topics:
                previous_topic = recent_topics[0]
                # Update the previous topic to end at the last assistant message before this user message
                # Find the parent of the user node (should be the last assistant message)
                parent_node = get_node(user_node_id).get('parent_id')
                if parent_node:
                    update_topic_end_node(previous_topic['name'], previous_topic['start_node_id'], parent_node)
                    # Queue the old topic for compression
                    queue_topic_for_compression(previous_topic['start_node_id'], parent_node, previous_topic['name'])
                    if config.get("debug", False):
                        typer.echo(f"   📦 Queued topic '{previous_topic['name']}' for compression")
            
            # Create the new topic starting from this user message
            store_topic(new_topic_name, user_node_id, assistant_node_id, 'detected')
            typer.echo("")
            typer.secho(f"🔄 Topic changed to: {new_topic_name}", fg=get_system_color())
        else:
            # No topic change - extend the current topic if one exists
            recent_topics = get_recent_topics(limit=1)
            if recent_topics:
                current_topic = recent_topics[0]
                # If no topic change was detected, we're continuing the current topic
                # Update it to include the new assistant response
                update_topic_end_node(current_topic['name'], current_topic['start_node_id'], assistant_node_id)
            else:
                # No topics exist yet - check if we should create the first topic
                if should_create_first_topic(user_node_id):
                    # Get conversation for topic extraction
                    conversation_chain = get_ancestry(assistant_node_id)
                    
                    # Build segment from the entire conversation so far
                    segment = build_conversation_segment(conversation_chain, max_length=2000)
                    
                    if config.get("debug", False):
                        typer.echo(f"\n🔍 DEBUG: Extracting first topic from conversation:")
                        typer.echo(f"   Conversation preview: {segment[:200]}...")
                        typer.echo(f"   Total length: {len(segment)} chars")
                        typer.echo(f"   Number of nodes: {len(conversation_chain)}")
                    
                    topic_name = extract_topic_ollama(segment)
                    
                    if topic_name:
                        # Find the first user node to use as the start of the topic
                        first_user_node = None
                        for node in reversed(conversation_chain):
                            if node.get('role') == 'user':
                                first_user_node = node
                        
                        if first_user_node:
                            # Store the topic spanning from first user message to current assistant message
                            store_topic(topic_name, first_user_node['id'], assistant_node_id, 'initial')
                            typer.echo("")
                            typer.secho(f"📌 Created first topic: {topic_name}", fg=get_system_color())
        
        # Show topic evolution if enabled (after topic detection)
        if config.get("show_topics", False):
            _display_topic_evolution(assistant_node_id)

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
