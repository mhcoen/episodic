"""
Debug command for managing debug categories.
"""

import typer
from typing import Optional, List

from episodic.color_utils import secho_color
from episodic.configuration import get_success_color, get_error_color, get_info_color
from episodic.debug_system import debug_system, format_debug_status


app = typer.Typer(help="Manage debug categories")


@app.command(name="on")
def debug_on(categories: Optional[List[str]] = typer.Argument(None)):
    """
    Enable debug categories.
    
    Examples:
        /debug on           - Enable all debug output
        /debug on memory    - Enable memory debugging
        /debug on memory topic - Enable memory and topic debugging
    """
    if not categories:
        # Enable all
        debug_system.enable('all')
        secho_color("✓ All debug output enabled", fg=get_success_color())
    else:
        enabled = []
        failed = []
        
        for cat in categories:
            if debug_system.enable(cat):
                enabled.append(cat)
            else:
                failed.append(cat)
        
        if enabled:
            secho_color(f"✓ Enabled debug for: {', '.join(enabled)}", fg=get_success_color())
        if failed:
            secho_color(f"✗ Unknown categories: {', '.join(failed)}", fg=get_error_color())
            secho_color("  Available: " + ", ".join(debug_system.CATEGORIES.keys()), fg=get_info_color())


@app.command(name="off")
def debug_off(categories: Optional[List[str]] = typer.Argument(None)):
    """
    Disable debug categories.
    
    Examples:
        /debug off          - Disable all debug output
        /debug off memory   - Disable memory debugging
        /debug off format stream - Disable format and stream debugging
    """
    if not categories:
        # Disable all
        debug_system.disable('all')
        secho_color("✓ All debug output disabled", fg=get_success_color())
    else:
        for cat in categories:
            debug_system.disable(cat)
        secho_color(f"✓ Disabled debug for: {', '.join(categories)}", fg=get_success_color())


@app.command(name="only")
def debug_only(categories: List[str] = typer.Argument(...)):
    """
    Enable only specified debug categories, disabling all others.
    
    Examples:
        /debug only memory      - Only show memory debug output
        /debug only topic drift - Only show topic and drift debug output
    """
    debug_system.set_only(categories)
    enabled = debug_system.get_enabled()
    if enabled:
        secho_color(f"✓ Debug enabled only for: {', '.join(enabled)}", fg=get_success_color())
    else:
        secho_color("✓ All debug output disabled", fg=get_success_color())


@app.command(name="status")
def debug_status():
    """
    Show debug category status.
    
    Example:
        /debug status - Show which categories are enabled
    """
    typer.echo(format_debug_status())


@app.command(name="toggle")
def debug_toggle(category: str):
    """
    Toggle a debug category on/off.
    
    Example:
        /debug toggle memory - Toggle memory debugging
    """
    was_enabled = debug_system.toggle(category)
    if was_enabled:
        secho_color(f"✓ Debug disabled for: {category}", fg=get_success_color())
    else:
        secho_color(f"✓ Debug enabled for: {category}", fg=get_success_color())


@app.callback(invoke_without_command=True)
def debug_main(ctx: typer.Context):
    """
    Manage debug output categories.
    
    Examples:
        /debug              - Show help
        /debug status       - Show current status
        /debug on memory    - Enable memory debugging
        /debug off          - Disable all debugging
        /debug only memory topic - Show only memory and topic debug
    """
    if ctx.invoked_subcommand is None:
        # No subcommand, show status
        typer.echo(format_debug_status())
        typer.echo("\nUse '/debug --help' for usage information")