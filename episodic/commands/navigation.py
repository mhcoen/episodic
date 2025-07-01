"""
Navigation commands for the Episodic CLI.

Handles node creation, traversal, and display operations.
"""

import typer
from typing import Optional, Dict
from episodic.db import (
    initialize_db as init_db, insert_node as add_node, get_node,
    get_recent_nodes, get_ancestry
)
from episodic.config import config
from episodic.configuration import (
    DEFAULT_LIST_COUNT, get_text_color, get_system_color, 
    get_heading_color, get_llm_color
)
from episodic.conversation import conversation_manager


def wrapped_text_print_with_indent(text: str, indent_length: int, **typer_kwargs):
    """Print text with specific indentation, handling wrapped text."""
    from episodic.conversation import wrapped_text_print
    
    if config.get("text_wrap", True):
        # Split text into lines first
        lines = text.split('\n')
        output_lines = []
        
        for line in lines:
            if not line.strip():
                output_lines.append('')
                continue
                
            # For the first line, include the indent in the line itself
            # For wrapped continuations, they'll be handled by wrapped_text_print
            indented_line = ' ' * indent_length + line
            output_lines.append(indented_line)
        
        # Join and print with wrapped_text_print
        wrapped_text_print('\n'.join(output_lines), **typer_kwargs)
    else:
        # No wrapping, just indent each line
        lines = text.split('\n')
        indented_text = '\n'.join(' ' * indent_length + line for line in lines)
        typer.secho(indented_text, **typer_kwargs)


def format_role_display(role: Optional[str]) -> str:
    """Format role for display with proper styling."""
    if role == "assistant":
        model_name = config.get("model", "gpt-3.5-turbo")
        model_str = model_name
        
        # Special handling for Claude models
        if "claude" in model_name.lower():
            # Extract the meaningful part of the model name
            parts = model_name.split('-')
            if len(parts) >= 3:
                # e.g., "claude-3-opus-20240229" -> "Claude 3 Opus"
                model_str = f"Claude {parts[1].title()} {parts[2].title()}"
        
        return f"🤖 {model_str}"
    else:
        return "👤 You"


def _display_node_details(node: Dict) -> None:
    """Display detailed information about a node."""
    
    # Header
    typer.secho(f"\n{'='*60}", fg=get_heading_color())
    typer.secho(f"Node Details: {node['short_id']}", fg=get_heading_color(), bold=True)
    typer.secho(f"{'='*60}", fg=get_heading_color())
    
    # Metadata
    typer.secho("\nMetadata:", fg=get_heading_color(), bold=True)
    typer.echo(f"  ID: {node['id']}")
    typer.echo(f"  Short ID: {node['short_id']}")
    typer.echo(f"  Role: {node.get('role', 'user')}")
    if node.get('model'):
        typer.echo(f"  Model: {node['model']}")
    typer.echo(f"  Parent: {node.get('parent_id', 'None')}")
    
    # Content
    typer.secho("\nContent:", fg=get_heading_color(), bold=True)
    role_display = format_role_display(node.get('role', 'user'))
    color = get_llm_color() if node.get('role') == 'assistant' else get_text_color()
    typer.secho(f"\n{role_display}:", fg=color, bold=True)
    wrapped_text_print_with_indent(node['content'], 2, fg=color)
    
    # Children
    from episodic.db import get_children
    children = get_children(node['id'])
    if children:
        typer.secho("\nChildren:", fg=get_heading_color(), bold=True)
        for child in children:
            preview = child['content'][:50] if child.get('content') else "Empty"
            typer.echo(f"  - {child['short_id']}: {preview}...")
    
    typer.echo("")  # Final newline


def init(erase: bool = typer.Option(False, "--erase", "-e", help="Erase existing database")):
    """Initialize the database."""
    from episodic.benchmark import benchmark_operation
    from episodic.db import database_exists
    
    with benchmark_operation("Database initialization"):
        # Check if database already exists
        if database_exists() and not erase:
            typer.secho("⚠️  Database already exists!", fg="yellow")
            typer.secho("Use '/init --erase' or '/init -e' to erase and start fresh.", fg=get_system_color())
            return
        
        if erase:
            typer.secho("🗑️  Erasing existing database...", fg=get_system_color())
        
        init_db(erase=erase)
        
        if erase:
            typer.secho("✅ Database erased and reinitialized", fg=get_system_color())
            # Reset conversation manager state
            from episodic.conversation import conversation_manager
            conversation_manager.current_node_id = None
            conversation_manager.current_topic = None
            conversation_manager.reset_session_costs()
        else:
            typer.secho("✅ Database initialized", fg=get_system_color())


def add(content: str, parent: Optional[str] = typer.Option(None, "--parent", "-p", help="Parent node ID")):
    """Add a new node to the conversation graph."""
    parent_id = None
    if parent:
        parent_node = get_node(parent)
        if not parent_node:
            typer.secho(f"Error: Parent node '{parent}' not found", fg="red", err=True)
            raise typer.Exit(1)
        parent_id = parent_node['id']
    
    node_id = add_node(content, parent_id, role="user")
    node = get_node(node_id)
    typer.secho(f"✅ Added node {node['short_id']}", fg=get_system_color())


def show(node_id: str):
    """Show details of a specific node."""
    # get_node handles both short ID and full ID
    node = get_node(node_id)
    
    if not node:
        typer.secho(f"Error: Node '{node_id}' not found", fg="red", err=True)
        raise typer.Exit(1)
    
    _display_node_details(node)


def print_node(node_id: Optional[str] = None):
    """Print the content of a specific node or the current node."""
    from episodic.conversation import wrapped_llm_print
    
    if node_id is None:
        current_id = conversation_manager.get_current_node_id()
        if not current_id:
            typer.secho("No current node. Use 'head <node_id>' to set one.", 
                       fg=get_system_color())
            return
        node = get_node(current_id)
    else:
        node = get_node(node_id)
    
    if not node:
        typer.secho(f"Error: Node not found", fg="red", err=True)
        return
    
    if node.get('content'):
        wrapped_llm_print(node['content'])
    else:
        typer.secho("No content in this node", fg=get_system_color())


def head(node_id: Optional[str] = None):
    """Set or show the current head node."""
    if node_id is None:
        # Show current head
        current_id = conversation_manager.get_current_node_id()
        if current_id:
            node = get_node(current_id)
            if node:
                # Get parent for context
                parent_msg = ""
                if node.get('parent_id'):
                    parent = get_node(node['parent_id'])
                    if parent:
                        parent_msg = f" (parent: {parent['short_id']})"
                
                typer.secho(f"Current node: {node['short_id']}{parent_msg}", 
                           fg=get_system_color())
                
                # Show the content
                if node.get('content'):
                    role = node.get('role', 'user')
                    color = get_llm_color() if role == 'assistant' else get_text_color()
                    label = "Response" if role == 'assistant' else "Message"
                    typer.secho(f"\n{label}:", fg=color, bold=True)
                    from episodic.conversation import wrapped_text_print
                    preview = node['content'][:150]
                    if len(node['content']) > 150:
                        preview += "..."
                    wrapped_text_print(f"  {preview}", fg=color)
        else:
            typer.secho("No current node set", fg=get_system_color())
    else:
        # Set new head
        node = get_node(node_id)
        if not node:
            typer.secho(f"Error: Node '{node_id}' not found", fg="red", err=True)
            raise typer.Exit(1)
        
        conversation_manager.set_current_node_id(node['id'])
        typer.secho(f"Current node changed to: {node['short_id']}", 
                   fg=get_system_color())


def list(count: int = typer.Option(DEFAULT_LIST_COUNT, "--count", "-c", help="Number of recent nodes to list")):
    """List recent nodes from the conversation graph."""
    
    nodes = get_recent_nodes(count)
    
    if not nodes:
        typer.secho("No nodes found. Start a conversation or add nodes.", 
                   fg=get_system_color())
        return
    
    typer.secho(f"\n📜 Recent {len(nodes)} nodes:", fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    current_node_id = conversation_manager.get_current_node_id()
    
    for node in nodes:
        # Node ID and indicator
        is_current = node['id'] == current_node_id
        current_indicator = " 👈" if is_current else ""
        
        typer.secho(f"\n[{node['short_id']}]{current_indicator}", 
                   fg=get_heading_color(), bold=True)
        
        # Parent info
        if node.get('parent_id'):
            parent = get_node(node['parent_id'])
            if parent:
                typer.secho(f"  Parent: ", fg=get_text_color(), dim=True, nl=False)
                typer.secho(f"[{parent['short_id']}]", fg=get_text_color(), bold=True)
        
        # Content preview based on role
        if node['content']:
            role_display = format_role_display(node.get('role', 'user'))
            color = get_llm_color() if node.get('role') == 'assistant' else get_text_color()
            typer.secho(f"  {role_display}: ", fg=color, bold=True, nl=False)
            
            # Truncate long content
            content = node['content'].replace('\n', ' ')
            if len(content) > 60:
                content = content[:57] + "..."
            typer.secho(content, fg=color)
        
        # Show children count
        from episodic.db import get_children
        children = get_children(node['id'])
        if children:
            child_ids = [c['short_id'] for c in children[:3]]
            more = f" +{len(children)-3} more" if len(children) > 3 else ""
            typer.secho(f"  → Children: {', '.join(child_ids)}{more}", 
                       fg=get_text_color(), dim=True)
    
    typer.secho("\n" + "─" * 60, fg=get_heading_color())
    typer.secho("💡 Use 'show <id>' for full details or 'head <id>' to continue from a node", 
               fg=get_system_color(), dim=True)


def ancestry(node_id: str):
    """Show the ancestry (parent chain) of a node."""
    
    # get_node handles both short ID and full ID
    node = get_node(node_id)
    
    if not node:
        typer.secho(f"Error: Node '{node_id}' not found", fg="red", err=True)
        raise typer.Exit(1)
    
    ancestors = get_ancestry(node['id'])
    
    typer.secho(f"\n🌳 Ancestry for node {node['short_id']}:", 
               fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    for i, ancestor in enumerate(ancestors):
        # Indentation to show hierarchy
        indent = "  " * i
        
        # Node info
        typer.secho(f"{indent}[{ancestor['short_id']}]", 
                   fg=get_heading_color(), bold=True, nl=False)
        
        # Timestamp
        from datetime import datetime
        try:
            timestamp = datetime.fromisoformat(ancestor['timestamp'].replace('Z', '+00:00'))
            time_str = timestamp.strftime(" - %Y-%m-%d %H:%M")
            typer.secho(time_str, fg=get_text_color(), dim=True)
        except:
            typer.echo("")
        
        # Content preview
        if ancestor.get('content'):
            content = ancestor['content'].replace('\n', ' ')
            if len(content) > 50:
                content = content[:47] + "..."
            emoji = "🤖" if ancestor.get('role') == 'assistant' else "💬"
            color = get_llm_color() if ancestor.get('role') == 'assistant' else get_text_color()
            typer.secho(f"{indent}  {emoji} {content}", fg=color)
        
        if i < len(ancestors) - 1:
            typer.secho(f"{indent}  ↓", fg=get_text_color(), dim=True)
    
    typer.secho("─" * 60, fg=get_heading_color())