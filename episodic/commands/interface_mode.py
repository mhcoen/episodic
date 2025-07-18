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
    typer.echo()
    
    # Show grouped commands
    typer.secho("💬 Conversation:", fg=get_text_color(), bold=True)
    typer.secho("   /chat   - Normal mode  |  /muse - Web search  |  /new - Fresh topic", fg=get_text_color())
    typer.echo()
    
    typer.secho("📁 Files:", fg=get_text_color(), bold=True)
    typer.secho("   /save   - Save topic   |  /load - Load file   |  /files - List saved", fg=get_text_color())
    typer.echo()
    
    typer.secho("✨ Style:", fg=get_text_color(), bold=True)
    typer.secho("   /style  - Response length  |  /format - Response format", fg=get_text_color())
    typer.echo()
    
    typer.secho("⚙️  System:", fg=get_text_color(), bold=True)
    typer.secho("   /help   - Show commands    |  /exit - Leave", fg=get_text_color())
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