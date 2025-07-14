"""
Reset command to restore configuration to defaults.
"""

import json
import typer
from pathlib import Path
from typing import Optional
from episodic.config import config
from episodic.configuration import get_heading_color, get_system_color, get_text_color


def _get_actual_defaults() -> dict:
    """Get the actual default values from config.default.json if it exists, otherwise template defaults."""
    actual_defaults = config.get_template_defaults()
    default_json_path = Path.home() / ".episodic" / "config.default.json"
    if default_json_path.exists():
        try:
            with open(default_json_path, 'r') as f:
                file_defaults = json.load(f)
                # Filter out comment keys that start with "_comment"
                actual_defaults = {k: v for k, v in file_defaults.items() if not k.startswith('_comment')}
        except json.JSONDecodeError:
            # If corrupted, fall back to template defaults
            pass
    return actual_defaults


def reset(param: Optional[str] = None, save: bool = False):
    """
    Reset configuration parameters to default values.
    
    Args:
        param: Specific parameter to reset, or None to reset all
        save: Whether to save the reset values to config.json
    """
    # Get the actual defaults
    actual_defaults = _get_actual_defaults()
    
    if param:
        # Reset specific parameter
        if param in actual_defaults:
            default_value = actual_defaults[param]
            config.config[param] = default_value
            
            if save:
                # Create a method to explicitly save when needed
                with open(config.config_file, 'w') as f:
                    json.dump(config.config, f, indent=2)
                typer.secho(f"Reset {param} to default value (saved to config file)", 
                          fg=get_system_color())
            else:
                typer.secho(f"Reset {param} to default value (this session only)", 
                          fg=get_system_color())
            
            # Show the default value
            if isinstance(default_value, dict):
                typer.secho(f"  Default value: <dict with {len(default_value)} items>", 
                          fg=get_text_color())
            else:
                typer.secho(f"  Default value: {default_value}", fg=get_text_color())
        else:
            typer.secho(f"Unknown parameter: {param}", fg="red")
            typer.secho("Use '/reset' without arguments to see all parameters", 
                      fg=get_text_color())
    else:
        # Show current vs default values for all parameters
        typer.secho("\n🔄 Configuration Reset Options", fg=get_heading_color(), bold=True)
        typer.secho("=" * 60, fg=get_heading_color())
        
        # Check if config.default.json exists
        default_json_path = Path.home() / ".episodic" / "config.default.json"
        if default_json_path.exists():
            typer.secho(f"Default values from: {default_json_path}", fg=get_text_color())
        else:
            typer.secho("Using built-in default values", fg=get_text_color())
        
        typer.secho("\nCommands:", fg=get_heading_color())
        typer.secho("  /reset all", nl=False, fg=get_system_color())
        typer.secho("              # Reset all parameters to defaults (session only)", 
                  fg=get_text_color())
        typer.secho("  /reset all --save", nl=False, fg=get_system_color())
        typer.secho("       # Reset all and save to config file", fg=get_text_color())
        typer.secho("  /reset <param>", nl=False, fg=get_system_color())
        typer.secho("          # Reset specific parameter (session only)", 
                  fg=get_text_color())
        typer.secho("  /reset <param> --save", nl=False, fg=get_system_color())
        typer.secho("   # Reset specific parameter and save", fg=get_text_color())
        
        # Get the actual defaults and show parameters that differ from them
        actual_defaults = _get_actual_defaults()
        different = []
        for key, default_value in actual_defaults.items():
            current_value = config.get(key)
            if current_value != default_value:
                different.append((key, current_value, default_value))
        
        if different:
            typer.secho(f"\nParameters different from defaults ({len(different)}):", 
                      fg=get_heading_color())
            for key, current, default in different:
                # Mask sensitive API keys and other secrets
                if 'api_key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                    if current:
                        current_display = f"{'*' * 8}...{str(current)[-4:]}" if len(str(current)) > 8 else "***masked***"
                    else:
                        current_display = current
                    if default:
                        default_display = f"{'*' * 8}...{str(default)[-4:]}" if len(str(default)) > 8 else "***masked***"
                    else:
                        default_display = default
                else:
                    current_display = current
                    default_display = default
                
                typer.secho(f"  {key}:", fg=get_text_color())
                typer.secho(f"    Current: {current_display}", fg=typer.colors.YELLOW)
                typer.secho(f"    Default: {default_display}", fg=typer.colors.GREEN)
        else:
            typer.secho("\n✅ All parameters are at default values", fg=get_system_color())


def reset_all(save: bool = False):
    """Reset all configuration parameters to defaults."""
    # Get the actual defaults and reset all parameters
    actual_defaults = _get_actual_defaults()
    for key, value in actual_defaults.items():
        config.config[key] = value
    
    if save:
        # Explicitly save when requested
        with open(config.config_file, 'w') as f:
            json.dump(config.config, f, indent=2)
        typer.secho("✅ Reset all parameters to defaults (saved to config file)", 
                  fg=get_system_color())
    else:
        typer.secho("✅ Reset all parameters to defaults (this session only)", 
                  fg=get_system_color())
    
    typer.secho("   Use '/set' to view current values", fg=get_text_color())