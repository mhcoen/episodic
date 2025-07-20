"""
Developer commands for Episodic maintenance and debugging.

These commands are not shown in regular help and are intended for developers only.
"""

import typer
from episodic.configuration import get_heading_color, get_text_color, get_warning_color


def dev(subcommand: str = None, *args):
    """Developer commands for maintenance and debugging."""
    if not subcommand:
        typer.secho("\n🔧 Developer Commands", fg=get_heading_color(), bold=True)
        typer.secho("─" * 50, fg=get_heading_color())
        typer.secho("\nAvailable commands:", fg=get_text_color())
        typer.secho("  /dev reindex-help    Reindex help documentation", fg=get_text_color())
        return
    
    if subcommand == "reindex-help":
        # Import and call the help reindex function
        from episodic.commands.help import help_reindex
        help_reindex()
    else:
        typer.secho(f"Unknown developer command: {subcommand}", fg=get_warning_color())
        typer.secho("Use /dev to see available commands", fg=get_text_color())