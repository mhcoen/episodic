"""
Enhanced CLI command handling using the command registry.

This module provides a cleaner command handling system that uses
the centralized command registry.
"""

import shlex
import typer
from typing import List
from episodic.commands.registry import command_registry
from episodic.configuration import (
    EXIT_COMMANDS, get_system_color, get_heading_color, get_text_color
)


def handle_command_with_registry(command_str: str) -> bool:
    """
    Handle a command string using the command registry.
    
    Returns:
        bool: True if should exit, False otherwise
    """
    # Parse the command
    try:
        parts = shlex.split(command_str)
    except ValueError as e:
        typer.secho(f"Error parsing command: {e}", fg="red")
        return False
    
    if not parts:
        return False
    
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []
    
    # Remove leading slash if present
    if cmd.startswith('/'):
        cmd = cmd[1:]
    
    # Check for exit commands
    if cmd in EXIT_COMMANDS or cmd == "q":
        return True
    
    # Look up command in registry
    cmd_info = command_registry.get_command(cmd)
    
    if not cmd_info:
        typer.secho(f"Unknown command: /{cmd}", fg="red")
        typer.echo("Type /help for available commands")
        return False
    
    # Check if deprecated
    if cmd_info.deprecated:
        typer.secho(
            f"⚠️  Warning: /{cmd} is deprecated. Use /{cmd_info.replacement} instead.",
            fg="yellow"
        )
    
    # Handle the command based on its type
    try:
        # Special handling for unified commands
        if cmd in ["topics", "compression"]:
            # These commands expect action as first argument
            if args:
                action = args[0]
                remaining_args = args[1:]
                # Call with action and parse remaining args
                cmd_info.handler(action, *remaining_args)
            else:
                # Default action
                cmd_info.handler()
        else:
            # Legacy command handling - needs specific argument parsing
            # This is where we'd need command-specific logic
            # For now, pass through to original handler
            return handle_legacy_command(cmd, args)
    
    except Exception as e:
        typer.secho(f"Error executing command: {e}", fg="red")
        if typer.get_app().get("debug", False):
            import traceback
            traceback.print_exc()
    
    return False


def handle_legacy_command(cmd: str, args: List[str]) -> bool:
    """Handle legacy commands that aren't yet converted to new style."""
    # Import the original handle_command logic
    from episodic.cli import handle_command
    
    # Reconstruct command string
    if args:
        # Properly quote arguments that contain spaces
        quoted_args = []
        for arg in args:
            if ' ' in arg:
                quoted_args.append(f'"{arg}"')
            else:
                quoted_args.append(arg)
        command_str = f"/{cmd} {' '.join(quoted_args)}"
    else:
        command_str = f"/{cmd}"
    
    # Use original handler
    return handle_command(command_str)


def show_help_with_categories():
    """Show basic help information with common commands."""
    typer.secho("\n⌨️  Type messages directly to chat.", fg=get_heading_color(), bold=True)

    # Mode switching - most prominent
    typer.secho("\n🎭 Mode Selection:", fg=get_heading_color(), bold=True)
    mode_commands = [
        ("/muse on", "Enable Muse mode (Perplexity-like web search)"),
        ("/chat on", "Enable Chat mode (normal LLM conversation)"),
    ]
    
    for cmd, desc in mode_commands:
        typer.secho(f"  {cmd:<25} ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Basic commands for new users
    typer.secho("\n🚀 Getting Started:", fg=get_heading_color(), bold=True)
    basic_commands = [
        ("/init", "Initialize the database"),
        ("/model", "Show/change current chat model"),
        ("/web", "Manage web search providers"),
    ]
    
    for cmd, desc in basic_commands:
        typer.secho(f"  {cmd:<25} ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Common conversation commands
    typer.secho("\n💬 Common Commands:", fg=get_heading_color(), bold=True)
    conversation_commands = [
        ("/topics", "List conversation topics"),
        ("/summary", "Summarize recent conversation"),
        ("/cost", "Show session cost"),
    ]
    
    for cmd, desc in conversation_commands:
        typer.secho(f"  {cmd:<25} ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    #   typer.secho("\n" + "─" * 60, fg=get_heading_color())
    typer.secho("\n📖 Type '/help --all' to see all available commands", 
               fg=get_text_color(), bold=True)
    typer.secho("🚪 Type '/exit' or '/quit' to leave", fg=get_text_color(), dim=True)


def show_advanced_help():
    """Show all available commands organized by categories."""
    typer.secho("\n📚 Episodic Commands (Advanced)", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())
    
    # Get commands by category
    categories = command_registry.get_commands_by_category()
    
    # Define category order for advanced view
    category_order = [
        "Navigation", "Conversation", "Topics", "Configuration",
        "Knowledge Base", "Compression", "Utility"
    ]
    
    for category in category_order:
        if category not in categories:
            continue
            
        commands = categories[category]
        if not commands:
            continue
        
        # Skip deprecated commands in count
        active_commands = [cmd for cmd in commands if not cmd.deprecated]
        if not active_commands:
            continue
        
        # Category header
        typer.secho(f"\n{get_category_icon(category)} {category}:", 
                   fg=get_heading_color(), bold=True)
        
        # Show commands
        for cmd_info in active_commands:
            # Format command name with aliases
            cmd_display = f"/{cmd_info.name}"
            if cmd_info.aliases:
                cmd_display += f" (/{', /'.join(cmd_info.aliases)})"
            
            typer.secho(f"  {cmd_display:<30} ", 
                       fg=get_system_color(), bold=True, nl=False)
            typer.secho(cmd_info.description, fg=get_text_color())
    
    # Show deprecated commands separately
    deprecated_cmds = [
        cmd for cmds in categories.values() 
        for cmd in cmds if cmd.deprecated
    ]
    
    if deprecated_cmds:
        typer.secho("\n⚠️  Deprecated Commands:", fg="yellow", bold=True)
        for cmd_info in deprecated_cmds:
            typer.secho(f"  /{cmd_info.name:<25} → Use /{cmd_info.replacement}",
                       fg="yellow")
    
    typer.secho("\n" + "─" * 60, fg=get_heading_color())
    typer.secho("💡 Type messages directly to chat", fg=get_text_color(), dim=True)
    typer.secho("📝 Common settings: /set debug off, /set cost on, /set topics on", 
               fg=get_text_color(), dim=True)
    typer.secho("🚪 Type '/exit' or '/quit' to leave", fg=get_text_color(), dim=True)


def get_category_icon(category: str) -> str:
    """Get emoji icon for command category."""
    icons = {
        "Navigation": "🧭",
        "Conversation": "💬",
        "Topics": "📑",
        "Configuration": "⚙️",
        "Knowledge Base": "📚",
        "Compression": "📦",
        "Utility": "🛠️"
    }
    return icons.get(category, "📌")
