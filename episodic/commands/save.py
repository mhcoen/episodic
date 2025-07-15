"""Save command for exporting conversations to markdown."""

import typer
from typing import Optional
from episodic.markdown_export import export_topics_to_markdown
from episodic.configuration import get_system_color

def save_command(args: str):
    """
    Save conversations to markdown file.
    
    Usage:
        /save                     # Save current topic
        /save 3                   # Save topic 3
        /save 1-3                 # Save topics 1-3
        /save all conversation.md # Save all topics to specific file
    """
    parts = args.split() if args else []
    
    # Parse arguments
    topic_spec = "current"  # default
    filename = None
    
    if not parts:
        # Just /save - use current topic
        topic_spec = "current"
    elif len(parts) == 1:
        # Could be topic spec or filename
        if parts[0].endswith('.md'):
            filename = parts[0]
        else:
            topic_spec = parts[0]
    elif len(parts) == 2:
        # Topic spec and filename
        topic_spec = parts[0]
        filename = parts[1]
    
    try:
        filepath = export_topics_to_markdown(topic_spec, filename)
        typer.secho(f"✅ Conversation saved to: {filepath}", fg=get_system_color())
    except ValueError as e:
        typer.secho(f"❌ {str(e)}", fg="red")
    except Exception as e:
        typer.secho(f"❌ Export failed: {str(e)}", fg="red")