"""
Unified settings and configuration commands.

This module consolidates all settings-related commands into a single
cohesive interface with subcommands.
"""

import typer
from typing import Optional

# Import existing settings commands
from .settings import (
    set as set_impl,
    verify as verify_impl,
    cost as cost_impl,
    model_params as model_params_impl,
    config_docs as config_docs_impl
)


def settings_command(
    action: str = typer.Argument("show", help="Action: show|set|verify|cost|params|docs"),
    # Set-specific options
    param: Optional[str] = typer.Option(None, "--param", "-p", help="Parameter name"),
    value: Optional[str] = typer.Option(None, "--value", "-v", help="Parameter value"),
    # Params-specific options
    param_set: Optional[str] = typer.Option(None, "--set", "-s", help="Parameter set (main/topic/compression)")
):
    """
    Unified settings management command.
    
    Actions:
      show    - Show all settings (default)
      set     - Set a configuration parameter
      verify  - Verify configuration integrity
      cost    - Show session cost information
      params  - Show/set model parameters
      docs    - Show configuration documentation
      
    Examples:
      /settings                        # Show all settings
      /settings set --param debug --value true
      /settings params --set main      # Show main model params
      /settings docs                   # Show config documentation
    """
    
    if action == "show":
        # Show all settings
        set_impl()
        
    elif action == "set":
        if param is None:
            typer.secho("Error: --param is required for set action", fg="red")
            return
        set_impl(param, value)
        
    elif action == "verify":
        verify_impl()
        
    elif action == "cost":
        cost_impl()
        
    elif action == "params":
        model_params_impl(param_set)
        
    elif action == "docs":
        config_docs_impl()
        
    else:
        typer.secho(f"Unknown action: {action}", fg="red")
        typer.echo("\nAvailable actions: show, set, verify, cost, params, docs")


# Backward compatibility - keep original commands working
def set_config(param: Optional[str] = None, value: Optional[str] = None):
    """Set configuration parameter (backward compatibility)."""
    if param:
        settings_command("set", param=param, value=value)
    else:
        settings_command("show")


def verify_config():
    """Verify configuration (backward compatibility)."""
    settings_command("verify")


def show_cost():
    """Show session cost (backward compatibility)."""
    settings_command("cost")


def show_model_params(param_set: Optional[str] = None):
    """Show model parameters (backward compatibility)."""
    settings_command("params", param_set=param_set)


def show_config_docs():
    """Show configuration documentation (backward compatibility)."""
    settings_command("docs")