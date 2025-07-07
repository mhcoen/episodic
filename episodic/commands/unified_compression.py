"""
Unified compression management commands.

This module consolidates all compression-related commands into a single
cohesive interface with subcommands.
"""

import typer
from typing import Optional

# Import existing compression commands
from .compression import (
    compress as compress_impl,
    compression_stats as stats_impl,
    compression_queue as queue_impl,
    api_call_stats as api_stats_impl,
    reset_api_stats as reset_api_impl
)
from .topics import compress_current_topic as compress_topic_impl


def compression_command(
    action: str = typer.Argument("stats", help="Action: stats|queue|api-stats|reset-api"),
    # Stats-specific options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed statistics")
):
    """
    Compression information and management command.
    
    Actions:
      stats       - Show compression statistics (default)
      queue       - Show pending compressions
      api-stats   - Show API call statistics
      reset-api   - Reset API call statistics
      
    Examples:
      /compression                    # Show compression stats
      /compression queue              # Show pending compressions
      /compression api-stats          # Show API usage
      
    For actual compression, use:
      /compress                      # Compress current branch
      /compress-current-topic        # Compress current topic
    """
    
    if action == "stats":
        stats_impl()
        
    elif action == "queue":
        queue_impl()
            
    elif action == "api-stats":
        api_stats_impl()
        
    elif action == "reset-api":
        reset_api_impl()
        
    else:
        typer.secho(f"Unknown action: {action}", fg="red")
        typer.echo("\nAvailable actions: stats, queue, api-stats, reset-api")


# Backward compatibility aliases
def compression_stats():
    """Show compression statistics."""
    compression_command("stats")


def compression_queue():
    """Show pending compressions."""
    compression_command("queue")


def api_stats():
    """Show API call statistics."""
    compression_command("api-stats")


def reset_api_stats():
    """Reset API call statistics."""
    compression_command("reset-api")