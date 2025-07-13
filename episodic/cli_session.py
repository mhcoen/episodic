"""
Session management for Episodic CLI.

This module handles session scripts, history, and session state.
"""

import os
import typer
from typing import List
from datetime import datetime

from episodic.config import config
from episodic.configuration import get_system_color


# Global session state
session_commands: List[str] = []


def add_to_session_commands(command: str):
    """Add a command to the session history."""
    global session_commands
    session_commands.append(command)


def get_session_commands() -> List[str]:
    """Get the current session commands."""
    return session_commands


def clear_session_commands():
    """Clear the session commands."""
    global session_commands
    session_commands = []


def save_session_script(filename: str):
    """Save the current session's commands to a script file."""
    # Ensure scripts directory exists
    scripts_dir = "scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    filepath = os.path.join(scripts_dir, filename)
    
    # Write commands to file
    with open(filepath, 'w') as f:
        f.write("# Episodic Session Script\n")
        f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        f.write("# This script contains all commands from your session.\n")
        f.write("# Execute with: /execute <filename>\n")
        f.write("#\n")
        f.write("# Lines starting with # are comments and will be skipped.\n")
        f.write("# Empty lines will also be skipped.\n")
        f.write("\n")
        
        # Write all session commands
        for cmd in session_commands:
            # Skip meta-commands that shouldn't be in scripts
            if not cmd.startswith('/save') and not cmd.startswith('/execute'):
                f.write(f"{cmd}\n")
    
    return filepath


def execute_script(filename: str):
    """Execute commands from a script file."""
    # Look for the file in various locations
    search_paths = [
        filename,  # As given
        f"scripts/{filename}",  # In scripts directory
        f"scripts/{filename}.txt",  # With .txt extension
        f"{filename}.txt",  # With .txt extension in current dir
    ]
    
    filepath = None
    for path in search_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if not filepath:
        typer.secho(f"Script file not found: {filename}", fg="red")
        typer.secho("Searched in:", fg="yellow")
        for path in search_paths:
            typer.secho(f"  - {path}", fg="yellow")
        return
    
    typer.secho(f"Executing script: {filepath}", fg=get_system_color())
    typer.echo()
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Track execution progress
        total_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        executed = 0
        
        # Import here to avoid circular imports
        from episodic.cli_command_router import handle_command
        from episodic.cli import handle_chat_message
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            executed += 1
            
            # Show progress
            typer.secho(f"[{executed}/{total_lines}] {line}", fg="cyan", bold=True)
            
            # Execute the line
            if line.startswith('/'):
                # It's a command
                should_exit = handle_command(line)
                if should_exit:
                    typer.secho("Script requested exit, stopping execution.", fg="yellow")
                    break
            else:
                # It's a chat message
                handle_chat_message(line)
            
            # Add a small delay between commands to avoid overwhelming the system
            if config.get("script_delay", 0.1) > 0:
                import time
                time.sleep(config.get("script_delay", 0.1))
        
        typer.echo()
        typer.secho(f"✅ Script execution complete: {executed} commands executed", fg="green")
        
    except FileNotFoundError:
        typer.secho(f"Script file not found: {filepath}", fg="red")
    except Exception as e:
        typer.secho(f"Error executing script: {e}", fg="red")
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg="red")


def save_to_history(message: str):
    """Save a message to the history file."""
    from episodic.configuration import DEFAULT_HISTORY_FILE
    history_file = config.get("history_file", DEFAULT_HISTORY_FILE)
    # Expand tilde to home directory
    history_file = os.path.expanduser(history_file)
    
    try:
        # Ensure the directory exists
        history_dir = os.path.dirname(history_file)
        if history_dir and not os.path.exists(history_dir):
            os.makedirs(history_dir, exist_ok=True)
        
        # Append to history file
        with open(history_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: {message}\n")
    except Exception as e:
        # Don't let history errors break the main flow
        if config.get("debug"):
            typer.secho(f"Failed to save to history: {e}", fg="yellow")