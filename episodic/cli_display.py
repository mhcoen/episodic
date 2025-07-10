"""
Display utilities for Episodic CLI.

This module handles welcome messages, prompts, and other display functions.
"""

import os
import typer
from datetime import datetime
from prompt_toolkit.formatted_text import HTML

from episodic.config import config
from episodic.configuration import (
    get_system_color, get_text_color, get_heading_color, get_llm_color
)
from episodic.db import get_head, get_recent_topics


def setup_environment():
    """Set up the environment for the CLI."""
    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    
    # Set up any other environment needs
    if config.get("clear_on_start", False):
        typer.clear()


def display_welcome():
    """Display the welcome message."""
    typer.secho("=" * 50, fg=get_heading_color(), bold=True)
    typer.secho("Welcome to Episodic CLI", fg=get_heading_color(), bold=True)
    typer.secho("Type /help for available commands", fg=get_text_color())
    typer.secho("=" * 50, fg=get_heading_color(), bold=True)


def display_model_info():
    """Display current model information."""
    # Try to get the model from config
    model = config.get("model")
    
    if model:
        typer.secho(f"Using model: {model}", fg=get_system_color())
        
        # Check if we're in muse mode
        if config.get("muse_mode", False):
            typer.secho("🔮 Muse mode active - type queries for web-enhanced responses", 
                       fg=get_system_color())
            # Also show web search status
            if config.get("web_search_enabled", False):
                typer.secho("🌐 Web search enabled", fg=get_system_color())
            else:
                typer.secho("⚠️  Web search disabled - enable with '/websearch on'", 
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


def get_prompt() -> str:
    """Get the current prompt string."""
    # Check if we're in muse mode
    if config.get("muse_mode", False):
        prompt_text = "muse> "
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