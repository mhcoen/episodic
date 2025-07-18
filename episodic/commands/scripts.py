"""
Script management commands.

This module provides commands for saving and executing session scripts.
"""

import os
import typer
from typing import Optional
from datetime import datetime
from episodic.configuration import get_system_color, get_text_color
from episodic.cli_session import session_commands


def scripts_command(subcommand: Optional[str] = None, filename: Optional[str] = None):
    """
    Manage session scripts.
    
    Usage:
        /scripts                    # Show available scripts
        /scripts save <filename>    # Save current session to script
        /scripts run <filename>     # Execute a script file
        /scripts list              # List available scripts
    """
    if not subcommand:
        list_scripts()
        return
    
    subcommand = subcommand.lower()
    
    if subcommand == "save":
        if not filename:
            typer.secho("Please provide a filename: /scripts save <filename>", fg="red")
            return
        save_script(filename)
    elif subcommand == "run":
        if not filename:
            typer.secho("Please provide a filename: /scripts run <filename>", fg="red")
            return
        run_script(filename)
    elif subcommand == "list":
        list_scripts()
    else:
        typer.secho(f"Unknown subcommand: {subcommand}", fg="red")
        typer.secho("Valid subcommands: save, run, list", fg=get_text_color())


def save_script(filename: str):
    """Save the current session's commands to a script file."""
    # Ensure scripts directory exists
    scripts_dir = "scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    filepath = os.path.join(scripts_dir, filename)
    
    # Get session commands
    commands = session_commands
    
    # Filter out unwanted commands
    filtered_commands = [
        cmd for cmd in commands 
        if not cmd.startswith('/scripts') and not cmd.startswith('/save')
    ]
    
    # Write commands to file
    with open(filepath, 'w') as f:
        f.write("# Episodic Session Script\n")
        f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        f.write("# This script contains all commands from your session.\n")
        f.write("# Execute with: /scripts run <filename>\n")
        f.write("#\n")
        f.write("# Lines starting with # are comments and will be skipped.\n")
        f.write("# Empty lines will also be skipped.\n")
        f.write("\n")
        
        # Write all session commands
        for cmd in filtered_commands:
            f.write(f"{cmd}\n")
    
    typer.secho(f"✅ Script saved to: {filepath}", fg="green")
    typer.secho(f"   Commands saved: {len(filtered_commands)}", fg=get_text_color())
    typer.secho(f"   Run with: /scripts run {filename}", fg=get_text_color())


def run_script(filename: str):
    """Execute commands from a script file."""
    # Import execute_script from cli_session
    from episodic.cli_session import execute_script
    execute_script(filename)


def list_scripts():
    """List available script files."""
    scripts_dir = "scripts"
    
    typer.secho("\n📄 Available Scripts:", fg=get_system_color(), bold=True)
    typer.secho("─" * 50, fg=get_system_color())
    
    if not os.path.exists(scripts_dir):
        typer.secho("No scripts directory found.", fg=get_text_color())
        typer.secho("Save your first script with: /scripts save <filename>", fg=get_text_color())
        return
    
    script_files = []
    for file in sorted(os.listdir(scripts_dir)):
        if file.endswith('.txt'):
            filepath = os.path.join(scripts_dir, file)
            stat = os.stat(filepath)
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            # Count non-comment lines
            with open(filepath, 'r') as f:
                lines = f.readlines()
                command_count = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            
            script_files.append((file, size, mtime, command_count))
    
    if not script_files:
        typer.secho("No script files found.", fg=get_text_color())
        typer.secho("Save your first script with: /scripts save <filename>", fg=get_text_color())
        return
    
    for filename, size, mtime, cmd_count in script_files:
        # Format the display
        name_display = f"  {filename:<30}"
        size_display = f"{size:>8,} bytes"
        time_display = mtime.strftime("%Y-%m-%d %H:%M")
        cmd_display = f"{cmd_count} commands"
        
        typer.secho(name_display, fg=get_system_color(), bold=True, nl=False)
        typer.secho(f"{size_display}  {time_display}  {cmd_display}", fg=get_text_color())
    
    typer.echo()
    typer.secho("Run a script with: /scripts run <filename>", fg=get_text_color())