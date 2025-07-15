"""List markdown files in a directory."""

import os
import typer
from datetime import datetime
from typing import Optional, List
from episodic.configuration import get_system_color, get_text_color, get_heading_color


def ls_command(directory: Optional[str] = None):
    """
    List markdown files in a directory.
    
    Usage:
        /ls              # List markdown files in current directory
        /ls exports      # List markdown files in exports directory
        /ls ~/Documents  # List markdown files in Documents
    """
    # Default to current directory
    if not directory:
        directory = "."
    
    # Expand user home directory
    directory = os.path.expanduser(directory)
    
    # Check if directory exists
    if not os.path.exists(directory):
        typer.secho(f"❌ Directory not found: {directory}", fg="red")
        return
    
    if not os.path.isdir(directory):
        typer.secho(f"❌ Not a directory: {directory}", fg="red")
        return
    
    # Find all markdown files
    markdown_files = find_markdown_files(directory)
    
    if not markdown_files:
        typer.secho(f"No markdown files found in {directory}", fg=get_text_color())
        return
    
    # Display header
    display_path = directory if directory != "." else "current directory"
    typer.secho(f"\n📁 Markdown files in {display_path}", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())
    
    # Display files with details
    for file_info in markdown_files:
        display_file_info(file_info)
    
    # Summary
    typer.secho(f"\nTotal: {len(markdown_files)} markdown files", fg=get_system_color())


def find_markdown_files(directory: str) -> List[dict]:
    """Find all markdown files in the directory."""
    markdown_files = []
    
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.md', '.markdown')):
                filepath = os.path.join(directory, filename)
                
                # Get file stats
                try:
                    stats = os.stat(filepath)
                    size = stats.st_size
                    modified = stats.st_mtime
                    
                    # Try to read first few lines for preview
                    preview = get_file_preview(filepath)
                    
                    markdown_files.append({
                        'name': filename,
                        'path': filepath,
                        'size': size,
                        'modified': modified,
                        'preview': preview
                    })
                except (OSError, IOError):
                    # If we can't read the file, just add basic info
                    markdown_files.append({
                        'name': filename,
                        'path': filepath,
                        'size': 0,
                        'modified': 0,
                        'preview': None
                    })
    
    except PermissionError:
        typer.secho(f"⚠️  Permission denied accessing directory", fg="yellow")
        return []
    
    # Sort by modification time (newest first)
    markdown_files.sort(key=lambda x: x['modified'], reverse=True)
    
    return markdown_files


def get_file_preview(filepath: str) -> Optional[str]:
    """Get a preview of the file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 5:  # Read up to 5 lines
                    break
                line = line.strip()
                if line:  # Skip empty lines
                    lines.append(line)
            
            # Look for title or topic
            for line in lines:
                if line.startswith('# '):
                    return line[2:].strip()
                elif line.startswith('## '):
                    return line[3:].strip()
            
            # Return first non-empty line if no title found
            return lines[0] if lines else None
    
    except (OSError, IOError, UnicodeDecodeError):
        return None


def display_file_info(file_info: dict):
    """Display information about a markdown file."""
    name = file_info['name']
    size = file_info['size']
    modified = file_info['modified']
    preview = file_info['preview']
    
    # Format size
    if size < 1024:
        size_str = f"{size} B"
    elif size < 1024 * 1024:
        size_str = f"{size / 1024:.1f} KB"
    else:
        size_str = f"{size / (1024 * 1024):.1f} MB"
    
    # Format modified time
    if modified > 0:
        mod_dt = datetime.fromtimestamp(modified)
        # Show relative time for recent files
        now = datetime.now()
        diff = now - mod_dt
        
        if diff.days == 0:
            if diff.seconds < 3600:
                time_str = f"{diff.seconds // 60} minutes ago"
            else:
                time_str = f"{diff.seconds // 3600} hours ago"
        elif diff.days == 1:
            time_str = "yesterday"
        elif diff.days < 7:
            time_str = f"{diff.days} days ago"
        else:
            time_str = mod_dt.strftime("%Y-%m-%d")
    else:
        time_str = "unknown"
    
    # Display file info
    typer.secho(f"\n📄 {name}", fg=get_text_color(), bold=True)
    typer.secho(f"   Size: {size_str} • Modified: {time_str}", fg=get_text_color(), dim=True)
    
    if preview:
        # Truncate preview if too long
        if len(preview) > 60:
            preview = preview[:57] + "..."
        typer.secho(f"   Preview: {preview}", fg=get_system_color())


def format_relative_path(filepath: str) -> str:
    """Format a file path relative to current directory if possible."""
    try:
        cwd = os.getcwd()
        relpath = os.path.relpath(filepath, cwd)
        # If the relative path is shorter or cleaner, use it
        if len(relpath) < len(filepath) and not relpath.startswith('..'):
            return relpath
    except ValueError:
        # Different drives on Windows, can't make relative
        pass
    
    return filepath