"""
Mode switching commands for Episodic (muse vs chat mode).
"""

import typer
from episodic.config import config
from episodic.configuration import get_system_color, get_text_color


def muse_command(enable: bool = True):
    """Enable or disable muse mode (all input becomes web searches)."""
    if enable:
        config.set("muse_mode", True)
        # Also enable web search if not already enabled
        if not config.get("web_search_enabled", False):
            config.set("web_search_enabled", True)
            typer.secho("✓ Web search enabled automatically", fg="green")
        typer.secho("🎭 Muse mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("All input will be treated as web search queries", fg=get_text_color())
        typer.secho("(Like Perplexity - synthesized answers from web search)", fg=typer.colors.WHITE, dim=True)
    else:
        config.set("muse_mode", False)
        typer.secho("💬 Chat mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("Input will be sent to the LLM", fg=get_text_color())


def chat_command(enable: bool = True):
    """Enable or disable chat mode (normal LLM conversation)."""
    if enable:
        # Chat on means muse off
        config.set("muse_mode", False)
        typer.secho("💬 Chat mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("Input will be sent to the LLM", fg=get_text_color())
    else:
        # Chat off means muse on
        config.set("muse_mode", True)
        # Also enable web search if not already enabled
        if not config.get("web_search_enabled", False):
            config.set("web_search_enabled", True)
            typer.secho("✓ Web search enabled automatically", fg="green")
        typer.secho("🎭 Muse mode ", nl=False, fg=get_system_color(), bold=True)
        typer.secho("ENABLED", fg="bright_green", bold=True)
        typer.secho("All input will be treated as web search queries", fg=get_text_color())
        typer.secho("(Like Perplexity - synthesized answers from web search)", fg=typer.colors.WHITE, dim=True)


def handle_muse(args: list):
    """Handle /muse command with on/off argument."""
    if not args:
        # Show current status
        muse_enabled = config.get("muse_mode", False)
        if muse_enabled:
            typer.secho("🎭 Muse mode is ", nl=False, fg=get_system_color())
            typer.secho("ENABLED", fg="bright_green", bold=True)
            typer.secho("All input is being treated as web searches", fg=get_text_color())
        else:
            typer.secho("💬 Chat mode is ", nl=False, fg=get_system_color())
            typer.secho("ENABLED", fg="bright_green", bold=True)
            typer.secho("Input is being sent to the LLM", fg=get_text_color())
    elif args[0].lower() == "on":
        muse_command(True)
    elif args[0].lower() == "off":
        muse_command(False)
    else:
        typer.secho("Usage: /muse [on|off]", fg="red")


def handle_chat(args: list):
    """Handle /chat command with on/off argument."""
    if not args:
        # Show current status
        muse_enabled = config.get("muse_mode", False)
        if muse_enabled:
            typer.secho("💬 Chat mode is ", nl=False, fg=get_system_color())
            typer.secho("DISABLED", fg="bright_red", bold=True)
            typer.secho("Muse mode is active - all input is treated as web searches", fg=get_text_color())
        else:
            typer.secho("💬 Chat mode is ", nl=False, fg=get_system_color())
            typer.secho("ENABLED", fg="bright_green", bold=True)
            typer.secho("Input is being sent to the LLM", fg=get_text_color())
    elif args[0].lower() == "on":
        chat_command(True)
    elif args[0].lower() == "off":
        chat_command(False)
    else:
        typer.secho("Usage: /chat [on|off]", fg="red")