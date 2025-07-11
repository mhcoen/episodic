"""
Muse mode command handlers for Episodic.

This module provides command handlers for muse mode - Perplexity-like web search synthesis.
"""

from typing import Optional
import typer

from episodic.config import config
from episodic.configuration import get_system_color, get_text_color


def muse(action: Optional[str] = None):
    """
    Handle muse mode toggling - treat all input as web search queries.
    
    Args:
        action: Optional action - "on", "off", or None to show status
    """
    if action is None:
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
    elif action == "on":
        muse_on()
    elif action == "off":
        muse_off()
    else:
        typer.secho("Usage: /muse [on|off]", fg="red")


def muse_on():
    """Enable muse mode - all input becomes web searches."""
    config.set("muse_mode", True)
    
    # Also enable web search if not already enabled
    if not config.get("web_search_enabled", False):
        config.set("web_search_enabled", True)
        typer.secho("✓ Web search enabled automatically", fg="green")
    
    typer.secho("🎭 Muse mode ", nl=False, fg=get_system_color(), bold=True)
    typer.secho("ENABLED", fg="bright_green", bold=True)
    typer.secho("All input will be treated as web search queries", fg=get_text_color())
    typer.secho("(Like Perplexity - synthesized answers from web search)", 
                fg=typer.colors.WHITE, dim=True)


def muse_off():
    """Disable muse mode - return to normal chat mode."""
    config.set("muse_mode", False)
    
    typer.secho("💬 Chat mode ", nl=False, fg=get_system_color(), bold=True)
    typer.secho("ENABLED", fg="bright_green", bold=True)
    typer.secho("Input will be sent to the LLM", fg=get_text_color())