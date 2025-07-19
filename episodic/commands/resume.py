"""Resume command for loading markdown conversations."""

import typer
from episodic.markdown_import import import_markdown_file
from episodic.configuration import get_system_color, get_text_color
from episodic.commands.interface_mode import is_simple_mode

def resume_command(filepath: str):
    """
    Load and continue a markdown conversation.
    
    Usage:
        /resume conversation.md   # Continue from loaded conversation
    """
    if not filepath:
        typer.secho("❌ Please specify a markdown file to load", fg="red")
        return
    
    try:
        # Import the markdown file
        from episodic.conversation import conversation_manager
        
        typer.secho(f"📄 Loading conversation from {filepath}...", 
                   fg=get_system_color())
        
        last_node_id = import_markdown_file(filepath, conversation_manager)
        
        # Update conversation position
        conversation_manager.current_node_id = last_node_id
        
        typer.secho(f"✅ Conversation loaded successfully!", fg=get_system_color())
        if is_simple_mode():
            typer.secho(f"   Continue chatting to pick up where you left off", 
                       fg=get_text_color())
        else:
            typer.secho(f"   Continue chatting or use /list to see recent messages", 
                       fg=get_text_color())
        
    except FileNotFoundError:
        typer.secho(f"❌ File not found: {filepath}", fg="red")
    except ValueError as e:
        typer.secho(f"❌ {str(e)}", fg="red")
    except Exception as e:
        typer.secho(f"❌ Failed to load: {str(e)}", fg="red")