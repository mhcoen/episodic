"""
Interface mode commands for switching between simple and advanced modes.
"""

import typer
from episodic.config import config
from episodic.configuration import get_system_color, get_text_color


def simple_mode_command():
    """Switch to simple mode with just essential commands."""
    config.set("interface_mode", "simple")
    
    typer.secho("\n🎯 Simple Mode Activated", fg=get_system_color(), bold=True)
    typer.secho("─" * 50, fg=get_system_color())
    typer.secho("Just the essentials - 10 commands to get things done:", fg=get_text_color())
    typer.echo()
    
    commands = [
        ("/chat", "Normal conversation mode"),
        ("/muse", "Web search mode (like Perplexity)"),
        ("/new", "Start fresh topic"),
        ("/save", "Save current topic"),
        ("/load", "Load conversation"),
        ("/files", "List saved conversations"),
        ("/style", "Set response length (concise/standard/comprehensive)"),
        ("/format", "Set response format (paragraph/bullet-points)"),
        ("/help", "Show commands"),
        ("/exit", "Leave Episodic"),
    ]
    
    for cmd, desc in commands:
        typer.secho(f"  {cmd:<10}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(f" - {desc}", fg=get_text_color())
    
    typer.echo()
    typer.secho("Advanced features hidden. Type /advanced to access all commands.", fg=get_text_color())
    typer.secho("Tab completion now shows only simple mode commands.", fg=get_text_color(), dim=True)


def advanced_mode_command():
    """Switch to advanced mode with all commands available."""
    config.set("interface_mode", "advanced")
    
    typer.secho("\n🔓 Advanced Mode Activated", fg=get_system_color(), bold=True)
    typer.secho("─" * 50, fg=get_system_color())
    typer.secho("All 50+ commands are now available!", fg=get_text_color())
    typer.secho("Type /help to see all commands.", fg=get_text_color())
    typer.echo()
    typer.secho("💡 Want simplicity? Type /simple", fg=get_text_color(), dim=True)


def is_simple_mode() -> bool:
    """Check if we're in simple mode."""
    return config.get("interface_mode", "advanced") == "simple"


def get_simple_mode_commands() -> set:
    """Get the set of commands available in simple mode."""
    return {
        "chat", "muse", "save", "load", "files", "new",
        "style", "format", "help", "exit", "quit", "simple", "advanced"
    }