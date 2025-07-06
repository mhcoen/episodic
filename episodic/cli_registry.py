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


def show_help_with_categories(advanced=False):
    """Show help information organized by categories."""
    from episodic.color_utils import secho_color
    
    # Header
    typer.echo()
    secho_color("🤖 Episodic Help", fg=get_heading_color(), bold=True)
    secho_color("═" * 50, fg=get_heading_color())
    
    # Quick start
    secho_color("\n✨ GETTING STARTED", fg=get_heading_color(), bold=True)
    secho_color("─" * 18, fg=get_heading_color())
    secho_color("  Just type to chat ", fg=get_text_color(), nl=False)
    secho_color("(no / needed)", fg=get_system_color())
    secho_color("  /topics              ", fg=get_system_color(), bold=True, nl=False)
    secho_color("See your conversation topics", fg=get_text_color())
    secho_color("  /set cost on         ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Show token usage", fg=get_text_color())
    secho_color("  /help                ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Show this help", fg=get_text_color())
    
    # Essential conversation commands
    secho_color("\n💬 CONVERSATION", fg=get_heading_color(), bold=True)
    secho_color("─" * 15, fg=get_heading_color())
    secho_color("  /model               ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Choose AI model", fg=get_text_color())
    secho_color("  /cost                ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Show session costs", fg=get_text_color())
    secho_color("  /summary [n]         ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Summarize last n messages", fg=get_text_color())
    secho_color("  /topics              ", fg=get_system_color(), bold=True, nl=False)
    secho_color("View conversation topics", fg=get_text_color())
    
    # Basic settings
    secho_color("\n⚙️  BASIC SETTINGS", fg=get_heading_color(), bold=True)
    secho_color("─" * 17, fg=get_heading_color())
    secho_color("  /set <param> <value> ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Change settings ", fg=get_text_color(), nl=False)
    secho_color("(use dashes: stream-rate)", fg=get_system_color(), dim=True)
    secho_color("  /set                 ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Show all current settings", fg=get_text_color())
    secho_color("  /reset all           ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Reset to defaults", fg=get_text_color())
    
    # Knowledge and search
    secho_color("\n📚 KNOWLEDGE & SEARCH", fg=get_heading_color(), bold=True)
    secho_color("─" * 21, fg=get_heading_color())
    
    # RAG section
    secho_color("  Knowledge Base (RAG):", fg=get_text_color(), bold=True)
    secho_color("    /rag on/off        ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Enable knowledge base", fg=get_text_color())
    secho_color("    /search <query>    ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Search your documents ", fg=get_text_color(), nl=False)
    secho_color("(/s)", fg=get_system_color(), dim=True)
    secho_color("    /index <file>      ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Add document to knowledge ", fg=get_text_color(), nl=False)
    secho_color("(/i)", fg=get_system_color(), dim=True)
    secho_color("    /docs              ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Manage indexed documents", fg=get_text_color())
    
    # Web search section
    secho_color("\n  Web Search:", fg=get_text_color(), bold=True)
    secho_color("    /websearch <query> ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Search the web ", fg=get_text_color(), nl=False)
    secho_color("(/ws)", fg=get_system_color(), dim=True)
    secho_color("    /websearch on/off  ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Enable web search", fg=get_text_color())
    
    if not advanced:
        # Show hint about advanced commands
        secho_color("\n" + "─" * 50, fg=get_heading_color())
        secho_color("💡 ", fg="bright_yellow", nl=False)
        secho_color("Type ", fg=get_text_color(), nl=False)
        secho_color("/help --all", fg=get_system_color(), bold=True, nl=False)
        secho_color(" for advanced commands", fg=get_text_color())
        secho_color("🔍 ", fg="bright_cyan", nl=False)
        secho_color("Type ", fg=get_text_color(), nl=False)
        secho_color("/help <command>", fg=get_system_color(), bold=True, nl=False)
        secho_color(" for detailed help", fg=get_text_color())
    else:
        # Show advanced commands
        show_advanced_help()
    
    typer.echo()  # Final newline


def show_advanced_help():
    """Show advanced commands for power users."""
    from episodic.color_utils import secho_color
    
    # Advanced configuration
    secho_color("\n🎛️  ADVANCED CONFIGURATION", fg=get_heading_color(), bold=True)
    secho_color("─" * 26, fg=get_heading_color())
    secho_color("  /set main.temp 0.7   ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Set model temperature", fg=get_text_color())
    secho_color("  /model-params        ", fg=get_system_color(), bold=True, nl=False)
    secho_color("View all model parameters", fg=get_text_color())
    secho_color("  /prompt <name>       ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Change system prompt", fg=get_text_color())
    
    # Topic management
    secho_color("\n📑 TOPIC MANAGEMENT", fg=get_heading_color(), bold=True)
    secho_color("─" * 19, fg=get_heading_color())
    secho_color("  /topics rename       ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Rename ongoing topics", fg=get_text_color())
    secho_color("  /topics compress     ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Manually compress topic", fg=get_text_color())
    secho_color("  /topics stats        ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Topic statistics", fg=get_text_color())
    
    # DAG navigation
    secho_color("\n🧭 DAG NAVIGATION", fg=get_heading_color(), bold=True)
    secho_color("─" * 17, fg=get_heading_color())
    secho_color("  /show <id>           ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Show node details", fg=get_text_color())
    secho_color("  /list [--count n]    ", fg=get_system_color(), bold=True, nl=False)
    secho_color("List recent nodes", fg=get_text_color())
    secho_color("  /ancestry <id>       ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Show conversation path", fg=get_text_color())
    secho_color("  /head [<id>]         ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Show/set current node", fg=get_text_color())
    
    # System commands
    secho_color("\n🔧 SYSTEM", fg=get_heading_color(), bold=True)
    secho_color("─" * 9, fg=get_heading_color())
    secho_color("  /verify              ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Check system integrity", fg=get_text_color())
    secho_color("  /benchmark           ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Performance statistics", fg=get_text_color())
    secho_color("  /visualize           ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Open graph visualization", fg=get_text_color())
    secho_color("  /init [--erase]      ", fg=get_system_color(), bold=True, nl=False)
    secho_color("Initialize database", fg=get_text_color())
    
    # Show hint about compression
    secho_color("\n" + "─" * 50, fg=get_heading_color())
    secho_color("📦 ", fg="bright_magenta", nl=False)
    secho_color("For compression commands, use ", fg=get_text_color(), nl=False)
    secho_color("/compression stats", fg=get_system_color(), bold=True)


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