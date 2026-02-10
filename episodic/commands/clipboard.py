"""
Clipboard command for copying conversation content.

Provides cross-platform clipboard support for copying the last LLM response
or specific node content.
"""

import subprocess
import typer
from typing import Optional

from episodic.db import get_head, get_node, get_recent_nodes
from episodic.configuration import get_success_color, get_error_color


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard.

    Tries multiple clipboard commands in order for cross-platform support:
    - pbcopy (macOS)
    - xclip (Linux X11)
    - wl-copy (Linux Wayland)
    - clip.exe (Windows/WSL)
    - PowerShell fallback (Windows)

    Returns True if successful, False otherwise.
    """
    clipboard_commands = [
        ["pbcopy"],                                    # macOS
        ["xclip", "-selection", "clipboard"],         # Linux X11
        ["wl-copy"],                                  # Linux Wayland
        ["clip.exe"],                                 # Windows/WSL
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard"],  # Windows fallback
    ]

    for cmd in clipboard_commands:
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            process.communicate(input=text.encode('utf-8'), timeout=3)
            if process.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

    return False


def copy_command(node_id: Optional[str] = None):
    """
    Copy conversation content to clipboard.

    Usage:
        /copy           Copy the last assistant response
        /copy <node_id> Copy content from a specific node

    Args:
        node_id: Optional node ID to copy from. If not provided, copies
                 the most recent assistant response.
    """
    if node_id:
        # Copy specific node
        node = get_node(node_id)
        if not node:
            typer.secho(f"Node not found: {node_id}", fg=get_error_color())
            return

        content = node.get('content', '')
        if not content:
            typer.secho("Node has no content to copy.", fg=get_error_color())
            return

        if copy_to_clipboard(content):
            role = node.get('role', 'unknown')
            short_id = node.get('short_id', node_id[:8])
            preview = content[:50] + "..." if len(content) > 50 else content
            typer.secho(f"✓ Copied {role} message ({short_id}): {preview}", fg=get_success_color())
        else:
            typer.secho("Failed to copy to clipboard. No clipboard tool available.", fg=get_error_color())
        return

    # No node_id provided - find the last assistant response
    head = get_head()
    if not head:
        typer.secho("No conversation history. Start chatting first.", fg=get_error_color())
        return

    # If head is an assistant message, use it
    if head.get('role') == 'assistant':
        content = head.get('content', '')
        if content:
            if copy_to_clipboard(content):
                preview = content[:50] + "..." if len(content) > 50 else content
                typer.secho(f"✓ Copied last response: {preview}", fg=get_success_color())
            else:
                typer.secho("Failed to copy to clipboard. No clipboard tool available.", fg=get_error_color())
            return

    # Head is a user message - look for the most recent assistant message
    recent = get_recent_nodes(limit=10)
    for node in recent:
        if node.get('role') == 'assistant':
            content = node.get('content', '')
            if content:
                if copy_to_clipboard(content):
                    preview = content[:50] + "..." if len(content) > 50 else content
                    typer.secho(f"✓ Copied last response: {preview}", fg=get_success_color())
                else:
                    typer.secho("Failed to copy to clipboard. No clipboard tool available.", fg=get_error_color())
                return

    typer.secho("No assistant response found to copy.", fg=get_error_color())
