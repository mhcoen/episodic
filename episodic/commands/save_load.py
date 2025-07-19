"""
Save and load commands for conversations.

This module provides simple commands to save and load conversations
as markdown files for the simple mode interface.
"""

import os
import typer
from typing import Optional
from datetime import datetime
from pathlib import Path

from episodic.config import config
from episodic.configuration import get_system_color, get_text_color
from episodic.markdown_export import export_topics_to_markdown
from episodic.db import get_recent_topics, get_head, get_recent_nodes
from episodic.llm import query_llm


def generate_filename_with_llm(topics: list, recent_messages: list) -> Optional[str]:
    """
    Use an instruct LLM to generate a descriptive filename based on conversation content.
    Returns None if generation fails.
    """
    try:
        # Get the topic detection model (fast and cheap)
        model = config.get("topic_detection_model", "huggingface/tiiuae/falcon-7b-instruct")
        
        # Build a summary of the conversation
        topic_names = [t['name'] for t in topics if t['name'] != "ongoing-discussion"][:3]
        
        # Get last few messages for context
        context_messages = []
        for msg in recent_messages[-6:]:  # Last 6 messages
            if msg['node_type'] == 'user':
                context_messages.append(f"User: {msg['content'][:100]}...")
        
        prompt = f"""Generate a short, descriptive filename (2-4 words) for a conversation about these topics:
Topics: {', '.join(topic_names) if topic_names else 'general discussion'}
Recent questions: {' '.join(context_messages[-3:]) if context_messages else 'various topics'}

Rules:
- Use only letters, numbers, and hyphens
- Make it descriptive but concise
- No dates or timestamps
- Lowercase only
- Example good filenames: quantum-physics-basics, recipe-collection, travel-planning

Filename:"""

        response = query_llm(
            prompt,
            model=model,
            temperature=0.3,
            max_tokens=20,
            system_prompt="You are a helpful assistant that generates concise, descriptive filenames."
        )
        
        # Clean the response
        filename = response.strip().lower()
        filename = filename.replace(' ', '-').replace('_', '-')
        filename = ''.join(c for c in filename if c.isalnum() or c == '-')
        
        # Validate it's reasonable
        if 2 <= len(filename) <= 50 and filename[0].isalnum():
            return filename
            
    except Exception as e:
        if config.get("debug"):
            typer.secho(f"Could not generate filename with LLM: {e}", fg="yellow")
    
    return None




def save_command(filename: Optional[str] = None):
    """
    Save the current topic to a markdown file.
    
    Usage:
        /save                  # Save current topic with auto-generated name
        /save my-chat          # Save current topic as my-chat.md
        /save "Project Notes"  # Save current topic as project-notes.md
        
    Note: Saves only the current topic to keep file sizes manageable.
    Use /out in advanced mode for more export options.
    """
    # Get current topic for auto-naming
    topics = get_recent_topics(limit=50)
    current_head = get_head()
    
    # Find current topic name - the most recent ongoing topic
    current_topic_name = "conversation"
    for topic in topics:
        if topic['end_node_id'] is None:  # Ongoing topic
            current_topic_name = topic['name']
            break
    
    # Generate filename if not provided
    if not filename:
        # Try to use LLM to generate a descriptive filename
        generated_filename = None
        if config.get("use_llm_filenames", True):  # New config option
            recent_messages = get_recent_nodes(limit=10)
            generated_filename = generate_filename_with_llm(topics, recent_messages)
            
            if generated_filename:
                # Add a short timestamp to ensure uniqueness
                timestamp = datetime.now().strftime("%H%M")
                filename = f"{generated_filename}-{timestamp}"
                typer.secho(f"Generated filename: {generated_filename}", fg=get_text_color(), dim=True)
        
        # Fall back to topic-based naming if LLM fails or is disabled
        if not filename:
            safe_topic = current_topic_name.lower().replace(' ', '-').replace('_', '-')
            safe_topic = ''.join(c for c in safe_topic if c.isalnum() or c == '-')
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            filename = f"{safe_topic}-{timestamp}"
    
    # Clean up filename
    filename = filename.strip()
    if filename.endswith('.md'):
        filename = filename[:-3]
    
    # Make filename filesystem-safe
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_filename = safe_filename.replace(' ', '-').lower()
    
    # Add .md extension
    full_filename = f"{safe_filename}.md"
    
    try:
        # Export only the current topic by default (not entire history!)
        # This prevents massive file sizes as conversations grow
        # Note: export_topics_to_markdown will use the configured export directory
        filepath = export_topics_to_markdown("current", full_filename)
        
        # Get file stats for display
        stat = os.stat(filepath)
        size_kb = stat.st_size / 1024
        
        typer.secho(f"✅ Current topic saved to: {filepath}", fg="green")
        typer.secho(f"   Size: {size_kb:.1f} KB", fg=get_text_color())
        typer.secho(f"   Topic: {current_topic_name}", fg=get_text_color())
        
        # Show how to load it back
        typer.secho(f"\n   Load with: /load {full_filename}", fg=get_text_color())
        
    except Exception as e:
        typer.secho(f"❌ Error saving conversation: {str(e)}", fg="red")
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg="red")


def load_command(filename: str):
    """
    Load a conversation from a markdown file.
    
    Usage:
        /load quantum-computing.md
        /load project-notes
    """
    from episodic.commands.resume import resume_command
    
    # Add .md extension if not present
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    
    # Get configured export directory
    export_dir = Path(os.path.expanduser(config.get("export_directory", "~/.episodic/exports")))
    filepath = export_dir / filename
    
    if not filepath.exists():
        # Try direct path
        filepath = Path(filename)
    
    if not filepath.exists():
        typer.secho(f"❌ File not found: {filename}", fg="red")
        typer.secho("Try /files to see available conversations", fg=get_text_color())
        return
    
    # Use the existing resume command
    resume_command(str(filepath))


def files_command():
    """
    List saved conversation files.
    
    Usage:
        /files    # List all saved conversations
    """
    export_dir = Path(os.path.expanduser(config.get("export_directory", "~/.episodic/exports")))
    
    typer.secho("\n📁 Saved Conversations:", fg=get_system_color(), bold=True)
    typer.secho("─" * 60, fg=get_system_color())
    
    if not export_dir.exists():
        typer.secho("No saved conversations yet.", fg=get_text_color())
        typer.secho("Save your first conversation with: /save", fg=get_text_color())
        return
    
    # Get all markdown files
    md_files = []
    for file in sorted(export_dir.glob("*.md")):
        stat = file.stat()
        size_kb = stat.st_size / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        # Try to get preview from first few lines
        preview = ""
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Look for first topic header
                for line in lines[:10]:
                    if line.startswith("## Topic:"):
                        preview = line[9:].strip()
                        break
                    elif line.startswith("# "):
                        preview = line[2:].strip()
                        break
        except:
            pass
        
        md_files.append((file.name, size_kb, mtime, preview))
    
    if not md_files:
        typer.secho("No saved conversations yet.", fg=get_text_color())
        typer.secho("Save your first conversation with: /save", fg=get_text_color())
        return
    
    # Display files
    for filename, size, mtime, preview in md_files:
        # Format time difference
        time_diff = datetime.now() - mtime
        if time_diff.days > 0:
            time_str = f"{time_diff.days} days ago"
        elif time_diff.seconds > 3600:
            hours = time_diff.seconds // 3600
            time_str = f"{hours} hours ago"
        else:
            minutes = time_diff.seconds // 60
            time_str = f"{minutes} minutes ago"
        
        # Display file info
        typer.secho(f"  {filename:<40}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(f" ({time_str}, {size:.1f} KB)", fg=get_text_color())
        
        if preview:
            typer.secho(f"     Preview: {preview[:50]}...", fg=get_text_color(), dim=True)
    
    typer.echo()
    typer.secho("Load a conversation with: /load <filename>", fg=get_text_color())