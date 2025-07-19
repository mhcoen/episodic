"""
Display utilities for Episodic CLI.

This module handles welcome messages, prompts, and other display functions.
"""

import os
import typer
from prompt_toolkit.formatted_text import HTML

from episodic.config import config
from episodic.configuration import (
    get_system_color, get_text_color, get_llm_color
)
from episodic.db import get_head, get_recent_topics


def setup_environment():
    """Set up the environment for the CLI."""
    # Clear the screen on startup (portable across platforms)
    typer.clear()
    
    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    
    # Apply simple mode display settings if starting in simple mode
    if config.get("interface_mode", "advanced") == "simple":
        config.set("show_drift", False)
        config.set("show_topics", False)
        config.set("debug", False)
    
    # Set up any other environment needs
    # Note: clear_on_start config option is now deprecated as we always clear


def display_welcome():
    """Display welcome message immediately."""
    typer.secho("Welcome to Episodic!", fg=get_system_color(), bold=True)
    
    # Check if we're in simple mode
    interface_mode = config.get("interface_mode", "advanced")
    if interface_mode == "simple":
        typer.secho("Running in Simple mode.", fg=get_text_color())
    else:
        typer.echo()  # Add blank line for better spacing
        typer.secho("💡 New to Episodic? Type /simple for a streamlined experience.", fg=get_text_color(), dim=True)

def display_model_info():
    """Display model and pricing information."""
    # Display current model and pricing information
    from episodic.llm_config import get_current_provider
    from litellm import cost_per_token
    
    # Use the same config key as /model command uses
    current_model = config.get("model", "gpt-3.5-turbo")
    provider = get_current_provider()
    
    # Check if it's a local provider
    LOCAL_PROVIDERS = ["ollama", "lmstudio", "local"]
    
    # Display model info
    typer.secho("Using model: ", nl=False, fg=get_text_color())
    typer.secho(f"{current_model}", nl=False, fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.secho(" (Provider: ", nl=False, fg=get_text_color())
    typer.secho(f"{provider}", nl=False, fg=typer.colors.BRIGHT_YELLOW, bold=True)
    typer.secho(")", fg=get_text_color())
    
    # Display pricing info
    if provider in LOCAL_PROVIDERS:
        typer.secho("Pricing: ", nl=False, fg=get_text_color())
        typer.secho("Local model", fg=typer.colors.BRIGHT_GREEN, bold=True)
    else:
        try:
            input_cost, output_cost = cost_per_token(model=current_model, prompt_tokens=1000, completion_tokens=1000)
            typer.secho("Pricing: ", nl=False, fg=get_text_color())
            typer.secho(f"${input_cost:.6f}/1K input, ${output_cost:.6f}/1K output", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
        except Exception:
            typer.secho("Pricing: ", nl=False, fg=get_text_color())
            typer.secho("Not available", fg=typer.colors.YELLOW)
    
    # Check if we're in muse mode
    if config.get("muse_mode", False):
        typer.secho("🔮 Muse mode active - type queries for web-enhanced responses", 
                   fg=get_system_color())
        # Also show web search status
        if config.get("web_search_enabled", False):
            typer.secho("🌐 Web search enabled", fg=get_system_color())
        else:
            typer.secho("⚠️  Web search disabled - enable with '/muse on'", 
                       fg="yellow")
    
    # Display current head if it exists
    head_id = get_head()
    if head_id:
        # Get recent topics to show current context
        topics = get_recent_topics(limit=10)
        current_topic = None
        
        # Find current topic (one without end_node_id)
        for topic in topics:
            if not topic.get('end_node_id'):
                current_topic = topic
                break
        
        if current_topic:
            typer.secho(f"Current topic: {current_topic['name']}", fg=get_system_color())

    typer.echo()
    typer.secho("Just start typing to chat or /help for commands.", fg=get_text_color())
    typer.echo()


def get_prompt() -> str:
    """Get the current prompt string."""
    # Check if we're in muse mode
    if config.get("muse_mode", False):
        prompt_text = "» "
        # Return as HTML for color support
        return HTML(f'<ansibrightmagenta><b>{prompt_text}</b></ansibrightmagenta>')
    
    # Normal mode prompt
    prompt_text = "> "
    return HTML(f'<ansigreen><b>{prompt_text}</b></ansigreen>')


def display_thinking_indicator():
    """Display a thinking indicator."""
    typer.secho("🤔 Thinking...", fg=get_llm_color(), nl=False)


def clear_thinking_indicator():
    """Clear the thinking indicator."""
    # Move cursor back and clear the line
    typer.echo("\r" + " " * 20 + "\r", nl=False)
