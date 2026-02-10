"""
Command routing for Episodic CLI.

This module handles parsing and routing of commands to their respective handlers.
"""

import shlex
import typer
from typing import List, Tuple

from episodic.config import config
from episodic.configuration import (
    EXIT_COMMANDS, get_text_color, get_error_color, get_warning_color,
)
from episodic.benchmark import display_pending_benchmark


def parse_command(command_str: str) -> Tuple[str, List[str]]:
    """
    Parse a command string into command and arguments.

    Returns:
        Tuple of (command, arguments)
    """
    try:
        parts = shlex.split(command_str)
    except ValueError:
        # If shlex fails (e.g., unmatched quotes), fall back to simple split
        # This handles contractions like "what's" better
        parts = command_str.split()

    if not parts:
        return "", []

    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    return cmd, args


def handle_command(command_str: str) -> bool:
    """
    Handle a command string.

    Returns:
        bool: True if should exit, False otherwise
    """
    from episodic.cli_command_handlers import COMMAND_MAP, handle_deprecated_commands

    cmd, args = parse_command(command_str)

    if not cmd:
        return False

    # Check for exit commands (remove leading slash for comparison)
    cmd_without_slash = cmd[1:] if cmd.startswith('/') else cmd
    if cmd_without_slash in EXIT_COMMANDS or cmd_without_slash == "q":
        return True

    # Check if we're in simple mode
    from episodic.commands.interface_mode import is_simple_mode, get_simple_mode_commands

    # Developer commands are always available
    developer_commands = ['dev', 'debug', 'test']

    # In simple mode, restrict to allowed commands (except developer commands)
    if is_simple_mode() and cmd_without_slash not in get_simple_mode_commands() and cmd_without_slash not in developer_commands:
        typer.secho(f"Command {cmd} is not available in simple mode.", fg=get_error_color())
        typer.secho("Available: /chat, /muse, /critique, /new, /save, /load, /files, /style, /format, /help, /exit", fg=get_warning_color())
        typer.secho("💡 Type /advanced to access all commands", fg=get_text_color(), dim=True)
        return False

    try:
        # Check if this is a utility command first
        from episodic.utility.cli_integration import (
            is_utility_command, handle_utility_command, display_utility_result
        )

        if is_utility_command(cmd_without_slash):
            args_str = " ".join(args) if args else ""
            result = handle_utility_command(cmd_without_slash, args_str)
            if result is not None:
                display_utility_result(result)
                return False

        # Route to appropriate handler via dispatch map
        handler = COMMAND_MAP.get(cmd)
        if handler is not None:
            handler(args)
        else:
            # Check if it's a deprecated command
            handle_deprecated_commands(cmd, args)
    except Exception as e:
        typer.secho(f"Error executing command: {e}", fg=get_error_color())
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg=get_error_color())

    # Display any pending benchmarks after command execution
    display_pending_benchmark()

    # Add blank line before next prompt
    typer.echo()

    return False
