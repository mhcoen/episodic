"""
Prompt management commands for the Episodic CLI.
"""

import typer
from typing import Optional
from episodic.prompt_manager import get_available_prompts, load_prompt, get_active_prompt
from episodic.config import config
from episodic.configuration import get_heading_color, get_text_color, get_system_color
from episodic.conversation import conversation_manager


def prompts(action: Optional[str] = None, name: Optional[str] = None):
    """Manage system prompts for different conversation styles."""
    if action is None or action == "list":
        # List available prompts
        available = get_available_prompts()
        active = get_active_prompt()
        
        typer.secho("\n📝 Available Prompts", fg=get_heading_color(), bold=True)
        typer.secho("─" * 50, fg=get_heading_color())
        
        for prompt_name in sorted(available):
            prompt_data = load_prompt(prompt_name)
            if prompt_data:
                is_active = prompt_name == active
                prefix = "→ " if is_active else "  "
                status = " (active)" if is_active else ""
                
                typer.secho(f"{prefix}{prompt_name}{status}", 
                           fg=get_system_color() if is_active else get_text_color(), 
                           bold=is_active)
                
                if prompt_data.get('description'):
                    typer.secho(f"   {prompt_data['description']}", 
                               fg=get_text_color(), dim=True)
        
        typer.secho("\n💡 Use '/prompts use <name>' to switch prompts", 
                   fg=get_text_color(), dim=True)
        
    elif action == "use" and name:
        # Switch to a different prompt
        available = get_available_prompts()
        
        if name not in available:
            typer.secho(f"Error: Prompt '{name}' not found", fg="red")
            typer.secho("Use '/prompts list' to see available prompts", fg=get_text_color())
            return
        
        # Load the prompt to verify it's valid
        prompt_data = load_prompt(name)
        if not prompt_data or 'content' not in prompt_data:
            typer.secho(f"Error: Prompt '{name}' is invalid or empty", fg="red")
            return
        
        # Update config
        config.set("active_prompt", name)
        
        # Update the conversation manager's system prompt
        conversation_manager.system_prompt = prompt_data['content']
        
        typer.secho(f"✅ Switched to '{name}' prompt", fg=get_system_color())
        
        if prompt_data.get('description'):
            typer.secho(f"   {prompt_data['description']}", fg=get_text_color(), dim=True)
        
        # Show a preview of the prompt
        preview = prompt_data['content'][:150]
        if len(prompt_data['content']) > 150:
            preview += "..."
        typer.secho("\nPrompt preview:", fg=get_text_color(), dim=True)
        typer.secho(f"   {preview}", fg=get_text_color(), dim=True)
        
    elif action == "show" and name:
        # Show details of a specific prompt
        if name not in get_available_prompts():
            typer.secho(f"Error: Prompt '{name}' not found", fg="red")
            return
        
        prompt_data = load_prompt(name)
        if not prompt_data:
            typer.secho(f"Error: Could not load prompt '{name}'", fg="red")
            return
        
        typer.secho(f"\n📝 Prompt: {name}", fg=get_heading_color(), bold=True)
        typer.secho("─" * 50, fg=get_heading_color())
        
        if prompt_data.get('description'):
            typer.secho("Description:", fg=get_text_color(), bold=True)
            typer.secho(f"  {prompt_data['description']}", fg=get_text_color())
        
        typer.secho("\nContent:", fg=get_text_color(), bold=True)
        for line in prompt_data['content'].split('\n'):
            typer.secho(f"  {line}", fg=get_text_color())
        
        if prompt_data.get('author'):
            typer.secho(f"\nAuthor: {prompt_data['author']}", fg=get_text_color(), dim=True)
        
        if prompt_data.get('version'):
            typer.secho(f"Version: {prompt_data['version']}", fg=get_text_color(), dim=True)
        
    else:
        typer.secho("Usage: /prompts [list|use <name>|show <name>]", fg=get_text_color())
        typer.secho("  list        - Show available prompts", fg=get_text_color())
        typer.secho("  use <name>  - Switch to a different prompt", fg=get_text_color())
        typer.secho("  show <name> - Show prompt details", fg=get_text_color())