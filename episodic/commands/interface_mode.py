"""
Interface mode commands for switching between simple and advanced modes.
"""

import typer
from episodic.config import config
from episodic.configuration import get_system_color, get_text_color, get_heading_color


def simple_mode_command():
    """Switch to simple mode with just essential commands."""
    config.set("interface_mode", "simple")
    
    # Disable technical output in simple mode
    config.set("show_drift", False)
    config.set("show_topics", False)
    config.set("debug", False)
    
    # Persist only the interface_mode setting
    config.config["interface_mode"] = "simple"
    config._save()
    
    typer.secho("\n✨ Simple Mode", fg=get_system_color(), bold=True)
    typer.secho("Everything you need, nothing you don't.\n", fg=get_text_color())
    
    # Conversation commands
    typer.secho("💬 Conversation", fg=get_heading_color(), bold=True)
    typer.secho("   /chat   ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Normal conversation mode", fg=get_text_color())
    typer.secho("   /muse   ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Web-enhanced mode (like Perplexity)", fg=get_text_color())
    typer.secho("   /new    ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Start a fresh topic", fg=get_text_color())
    typer.echo()
    
    # File commands
    typer.secho("📁 Files", fg=get_heading_color(), bold=True)
    typer.secho("   /save   ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Save current topic", fg=get_text_color())
    typer.secho("   /load   ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Load a conversation", fg=get_text_color())
    typer.secho("   /files  ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("List saved conversations", fg=get_text_color())
    typer.echo()
    
    # Style commands
    typer.secho("✨ Style", fg=get_heading_color(), bold=True)
    typer.secho("   /style  ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Response length (concise/standard/comprehensive)", fg=get_text_color())
    typer.secho("   /format ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Response format (paragraph/bulleted/mixed/academic)", fg=get_text_color())
    typer.echo()
    
    # System commands
    typer.secho("⚙️  System", fg=get_heading_color(), bold=True)
    typer.secho("   /theme  ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Change color theme", fg=get_text_color())
    typer.secho("   /help   ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Show this help", fg=get_text_color())
    typer.secho("   /exit   ", fg=get_system_color(), bold=True, nl=False)
    typer.secho("Leave Episodic", fg=get_text_color())
    typer.echo()
    
    typer.secho("──────────────────────────────────────────────────", fg=get_system_color(), dim=True)
    typer.secho("💡 Type ", fg=get_text_color(), nl=False)
    typer.secho("/advanced", fg=get_system_color(), bold=True, nl=False)
    typer.secho(" to access all 50+ commands", fg=get_text_color())


def advanced_mode_command():
    """Switch to advanced mode with all commands available."""
    config.set("interface_mode", "advanced")
    
    # Re-enable technical output in advanced mode (restore defaults)
    config.set("show_drift", True)
    config.set("show_topics", False)  # Keep topics off by default
    # Don't change debug - let user control that
    
    # Persist only the interface_mode setting
    config.config["interface_mode"] = "advanced"
    config._save()
    
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
        "style", "format", "theme", "help", "exit", "quit", "simple", "advanced"
    }