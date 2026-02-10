"""
Enhanced CLI command handling using the command registry.

This module provides a cleaner command handling system that uses
the centralized command registry.

Help text display functions live in cli_help_text.py.
"""

import shlex
import typer
from typing import List
from episodic.commands.registry import command_registry, register_all_commands
from episodic.configuration import EXIT_COMMANDS

# Re-export help functions for backward compatibility.
# Other modules import these names from cli_registry; the re-exports keep
# those import sites working without changes.
from episodic.cli_help_text import (  # noqa: F401
    show_help_with_categories,
    show_category_help,
    show_chat_help,
    show_settings_help,
    show_search_help,
    show_history_help,
    show_topics_help,
    show_markdown_help,
    show_voice_help,
    show_assistant_help,
    show_mcp_help,
    show_advanced_help,
    show_simple_help,
    get_category_icon,
    _display_aligned_commands,
)

# Ensure commands are registered
_registry_initialized = False

def _ensure_registry_initialized():
    global _registry_initialized
    if not _registry_initialized:
        register_all_commands()
        _registry_initialized = True


def handle_command_with_registry(command_str: str) -> bool:
    """
    Handle a command string using the command registry.

    Returns:
        bool: True if should exit, False otherwise
    """
    _ensure_registry_initialized()

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

    # Check if we're in simple mode
    from episodic.commands.interface_mode import is_simple_mode, get_simple_mode_commands

    # Look up command in registry
    cmd_info = command_registry.get_command(cmd)

    if not cmd_info:
        typer.secho(f"Unknown command: /{cmd}", fg="red")
        typer.echo("Type /help for available commands")
        return False

    # In simple mode, restrict to allowed commands
    if is_simple_mode() and cmd not in get_simple_mode_commands():
        from episodic.configuration import get_text_color
        typer.secho(f"Command /{cmd} is not available in simple mode.", fg="red")
        typer.secho("Available: /chat, /muse, /voice, /new, /save, /load, /files, /style, /format, /theme, /help, /exit", fg="yellow")
        typer.secho("💡 Type /advanced to access all commands", fg=get_text_color(), dim=True)
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
        if cmd in ["topics", "compression", "kg"]:
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
