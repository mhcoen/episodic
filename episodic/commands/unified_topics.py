"""
Unified topic management commands.

This module consolidates all topic-related commands into a single
cohesive interface with subcommands.
"""

import typer
from typing import Optional
from episodic.configuration import (
    get_heading_color, get_system_color
)
from episodic.db import get_recent_topics

# Import existing topic commands
from .topics import topics as list_topics_impl, compress_current_topic as compress_topic_impl
from .topic_rename import rename_ongoing_topics as rename_topics_impl
from .index_topics import index_topics as index_topics_impl
from .debug_topics import topic_scores as topic_scores_impl


def handle_topics_action(action: str = "list", **kwargs):
    """
    Handle topics actions without Typer decorators.
    This is for direct function calls from the CLI router.
    """
    if action == "list":
        list_topics_impl()
    elif action == "rename":
        rename_topics_impl()
    elif action == "compress":
        compress_topic_impl()
    elif action == "index":
        window_size = kwargs.get('window_size', 5)
        apply = kwargs.get('apply', False)
        verbose = kwargs.get('verbose', False)
        index_topics_impl(window_size=window_size, apply=apply, verbose=verbose)
    elif action == "scores":
        node_id = kwargs.get('node_id', None)
        topic_scores_impl(node_id=node_id)
    elif action == "stats":
        verbose = kwargs.get('verbose', False)
        show_topic_stats(verbose=verbose)
    else:
        typer.secho(f"Unknown action: {action}", fg="red")
        typer.secho("\nAvailable actions: list, rename, compress, index, scores, stats", fg="yellow")


def topics_command(
    action: str = typer.Argument("list", help="Action to perform: list|rename|compress|index|scores|stats"),
    # Common options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    # Index-specific options
    window_size: Optional[int] = typer.Option(None, "--window-size", "-w", help="Window size for index action"),
    apply: bool = typer.Option(False, "--apply", "-a", help="Apply changes (for index action)"),
    # List-specific options
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Number of topics to show"),
    # Scores-specific options
    node_id: Optional[str] = typer.Option(None, "--node", help="Node ID for score analysis")
):
    """
    Unified topic management command.
    
    Actions:
      list     - List all topics (default)
      rename   - Rename ongoing topics based on content
      compress - Compress the current topic
      index    - Manually detect topics using sliding windows
      scores   - Show topic detection scores
      stats    - Show topic statistics
    
    Examples:
      /topics              # List all topics
      /topics rename       # Rename ongoing topics
      /topics index 5      # Detect topics with window size 5
      /topics stats        # Show topic statistics
    """
    
    if action == "list":
        # Use existing implementation
        list_topics_impl()
        
    elif action == "rename":
        # Use existing implementation
        rename_topics_impl()
        
    elif action == "compress":
        # Use existing implementation
        compress_topic_impl()
        
    elif action == "index":
        # For index, window_size can be provided as argument or option
        if window_size is None:
            # Try to parse from remaining args
            try:
                pass
                # This is a bit of a hack but works for now
                window_size = 5  # Default
            except:
                window_size = 5
        index_topics_impl(window_size=window_size, apply=apply, verbose=verbose)
        
    elif action == "scores":
        # Use existing implementation
        topic_scores_impl(node_id=node_id)
        
    elif action == "stats":
        # New implementation for statistics
        show_topic_stats(verbose=verbose)
        
    else:
        typer.secho(f"Unknown action: {action}", fg="red")
        typer.echo("\nAvailable actions: list, rename, compress, index, scores, stats")


def show_topic_stats(verbose: bool = False):
    """Show statistics about topics in the conversation."""
    topics = get_recent_topics(limit=1000)  # Get all topics
    
    if not topics:
        typer.secho("No topics found.", fg=get_system_color())
        return
    
    typer.secho("\n📊 Topic Statistics", fg=get_heading_color(), bold=True)
    typer.secho("=" * 50, fg=get_heading_color())
    
    # Basic stats
    total_topics = len(topics)
    ongoing_topics = sum(1 for t in topics if t['end_node_id'] is None)
    completed_topics = total_topics - ongoing_topics
    
    typer.echo(f"\nTotal topics: {total_topics}")
    typer.echo(f"Completed: {completed_topics}")
    typer.echo(f"Ongoing: {ongoing_topics}")
    
    if verbose:
        # Topic name analysis
        typer.secho("\n📝 Topic Names:", fg=get_heading_color())
        topic_names = {}
        for topic in topics:
            name = topic['name']
            topic_names[name] = topic_names.get(name, 0) + 1
        
        # Show most common topics
        sorted_names = sorted(topic_names.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_names[:10]:
            if count > 1:
                typer.echo(f"  {name}: {count} occurrences")
        
        # Show average messages per topic
        typer.secho("\n📈 Topic Sizes:", fg=get_heading_color())
        
        # This would require counting nodes per topic
        # For now, just show a placeholder
        typer.echo("  Average messages per topic: (calculation pending)")
        
    typer.echo()


# Create aliases for backward compatibility
def topics_list():
    """Backward compatibility wrapper."""
    topics_command("list")


def topics_rename():
    """Backward compatibility wrapper."""
    topics_command("rename")


def topics_compress():
    """Backward compatibility wrapper."""
    topics_command("compress")


def topics_index(window_size: int = 5, apply: bool = False, verbose: bool = False):
    """Backward compatibility wrapper."""
    topics_command("index", window_size=window_size, apply=apply, verbose=verbose)