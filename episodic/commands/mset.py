"""
Model parameter setting command.

This module provides a streamlined way to set model parameters for different contexts.
"""

import typer
import json
from typing import Optional, Dict, Any
from episodic.config import config
from episodic.configuration import get_heading_color, get_system_color, get_text_color


def mset_command(
    param_spec: Optional[str] = typer.Argument(None, help="Parameter specification (e.g., chat.temperature or just chat)"),
    value: Optional[str] = typer.Argument(None, help="Value to set")
):
    """
    Set or view model parameters for different contexts.
    
    Usage:
        /mset                           # Show all parameters for all contexts
        /mset chat                      # Show all parameters for chat model
        /mset chat.temperature 0.7      # Set temperature for chat model
        /mset detection.temperature 0   # Set temperature for detection model
        
    Contexts:
        - chat: Main conversation model
        - detection: Topic detection model
        - compression: Compression/summarization model
        - synthesis: Web search synthesis model
        
    Parameters:
        - temperature: Randomness (0.0-2.0)
        - max_tokens: Maximum response length
        - top_p: Nucleus sampling (0.0-1.0)
        - presence_penalty: Penalize repeated topics (-2.0-2.0)
        - frequency_penalty: Penalize repeated words (-2.0-2.0)
    """
    
    # No arguments - show all parameters
    if not param_spec:
        show_all_parameters()
        return
    
    # Check if it's just a context name (e.g., "chat")
    valid_contexts = ["chat", "detection", "compression", "synthesis"]
    if param_spec.lower() in valid_contexts and not value:
        show_parameters_for_context(param_spec.lower())
        return
    
    # Parse context.parameter format
    if "." not in param_spec:
        typer.secho("Invalid format. Use: /mset <context>.<parameter> <value>", fg="red")
        typer.secho("Or: /mset <context> to view parameters", fg=get_text_color())
        return
    
    context, param = param_spec.split(".", 1)
    
    # Validate context
    if context.lower() not in valid_contexts:
        typer.secho(f"Unknown context: {context}", fg="red")
        typer.secho(f"Valid contexts: {', '.join(valid_contexts)}", fg=get_text_color())
        return
    
    # Must have a value to set
    if value is None:
        typer.secho("Missing value. Use: /mset <context>.<parameter> <value>", fg="red")
        return
    
    # Set the parameter
    set_parameter(context.lower(), param, value)


def show_all_parameters():
    """Show all parameters for all contexts in a table format."""
    typer.secho("\n⚙️  Model Parameters:", fg=get_heading_color(), bold=True)
    typer.secho("─" * 70, fg=get_heading_color())
    
    contexts = ["chat", "detection", "compression", "synthesis"]
    params_to_show = ["temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"]
    
    # Header
    header = "Parameter".ljust(20)
    for ctx in contexts:
        header += ctx.capitalize().center(12)
    typer.secho(header, fg=get_text_color(), bold=True)
    typer.secho("─" * 70, fg=get_system_color())
    
    # Show each parameter
    for param in params_to_show:
        row = param.ljust(20)
        for ctx in contexts:
            value = get_parameter_value(ctx, param)
            row += format_param_value(value).center(12)
        typer.secho(row, fg=get_text_color())
    
    typer.secho("\nUse '/mset <context>' to see details for a specific context", fg=get_text_color(), dim=True)
    
    # Now show the models
    typer.echo()  # Add blank line
    from episodic.commands.unified_model import show_current_models
    show_current_models()


def show_parameters_for_context(context: str):
    """Show parameters for a specific context."""
    # Get current model for context
    model_key = get_model_key_for_context(context)
    current_model = config.get(model_key, "")
    
    typer.secho(f"\n⚙️  {context.capitalize()} Model Parameters", fg=get_heading_color(), bold=True)
    typer.secho(f"Current model: {current_model}", fg=get_text_color(), dim=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    # Get parameters
    param_key = get_param_key_for_context(context)
    params = config.get(param_key, {})
    
    # Define parameter info
    param_info = {
        "temperature": ("Randomness/creativity", "0.0-2.0", 0.7),
        "max_tokens": ("Maximum response length", "1-∞", None),
        "top_p": ("Nucleus sampling threshold", "0.0-1.0", 1.0),
        "presence_penalty": ("Penalize repeated topics", "-2.0-2.0", 0.0),
        "frequency_penalty": ("Penalize repeated words", "-2.0-2.0", 0.0)
    }
    
    # Check model compatibility
    unsupported = get_unsupported_params(current_model)
    
    for param, (desc, range_str, default) in param_info.items():
        current = params.get(param) if isinstance(params, dict) else None
        
        # Format the display
        typer.secho(f"  {param:<20} ", fg=get_system_color(), nl=False)
        
        if param in unsupported:
            typer.secho("(not supported by model)", fg="red")
        else:
            if current is not None:
                typer.secho(f"{current:<8} ", fg=get_heading_color(), bold=True, nl=False)
            else:
                typer.secho(f"{'default':<8} ", fg=get_text_color(), dim=True, nl=False)
            
            typer.secho(f"{desc} [{range_str}]", fg=get_text_color())
    
    typer.secho(f"\nSet with: /mset {context}.<parameter> <value>", fg=get_text_color(), dim=True)


def set_parameter(context: str, param: str, value: str):
    """Set a parameter for a context."""
    # Validate parameter name
    valid_params = ["temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"]
    if param not in valid_params:
        typer.secho(f"Unknown parameter: {param}", fg="red")
        typer.secho(f"Valid parameters: {', '.join(valid_params)}", fg=get_text_color())
        return
    
    # Parse and validate value
    parsed_value = parse_parameter_value(param, value)
    if parsed_value is None:
        return  # Error already displayed by parse function
    
    # Check if model supports this parameter
    model_key = get_model_key_for_context(context)
    current_model = config.get(model_key, "")
    unsupported = get_unsupported_params(current_model)
    
    if param in unsupported:
        typer.secho(f"⚠️  Warning: {current_model} does not support '{param}'", fg="yellow")
        typer.secho("The parameter will be stored but ignored by this model", fg=get_text_color())
    
    # Get current parameters
    param_key = get_param_key_for_context(context)
    params = config.get(param_key, {})
    if not isinstance(params, dict):
        params = {}
    
    # Set the new value
    params[param] = parsed_value
    config.set(param_key, params)
    
    typer.secho(f"✓ Set {context}.{param} = {parsed_value}", fg="green")


def parse_parameter_value(param: str, value: str) -> Optional[Any]:
    """Parse and validate a parameter value."""
    try:
        if param == "max_tokens":
            val = int(value)
            if val < 1:
                typer.secho("max_tokens must be at least 1", fg="red")
                return None
            return val
        
        elif param in ["temperature", "top_p"]:
            val = float(value)
            if param == "temperature" and not (0.0 <= val <= 2.0):
                typer.secho("temperature must be between 0.0 and 2.0", fg="red")
                return None
            elif param == "top_p" and not (0.0 <= val <= 1.0):
                typer.secho("top_p must be between 0.0 and 1.0", fg="red")
                return None
            return val
        
        elif param in ["presence_penalty", "frequency_penalty"]:
            val = float(value)
            if not (-2.0 <= val <= 2.0):
                typer.secho(f"{param} must be between -2.0 and 2.0", fg="red")
                return None
            return val
        
        else:
            # Shouldn't reach here with validation
            return value
            
    except ValueError:
        typer.secho(f"Invalid value for {param}: {value}", fg="red")
        if param == "max_tokens":
            typer.secho("Expected an integer", fg=get_text_color())
        else:
            typer.secho("Expected a number", fg=get_text_color())
        return None


def get_parameter_value(context: str, param: str) -> Any:
    """Get a parameter value for a context."""
    param_key = get_param_key_for_context(context)
    params = config.get(param_key, {})
    
    if isinstance(params, dict):
        return params.get(param)
    return None


def format_param_value(value: Any) -> str:
    """Format a parameter value for display."""
    if value is None:
        return "default"
    elif isinstance(value, float):
        return f"{value:.1f}"
    else:
        return str(value)


def get_param_key_for_context(context: str) -> str:
    """Get the config key for parameters of a context."""
    if context == "chat":
        return "main_params"
    elif context == "detection":
        return "topic_params"
    elif context == "compression":
        return "compression_params"
    elif context == "synthesis":
        return "synthesis_params"
    else:
        return f"{context}_params"


def get_model_key_for_context(context: str) -> str:
    """Get the config key for model of a context."""
    if context == "chat":
        return "model"
    elif context == "detection":
        return "topic_detection_model"
    elif context == "compression":
        return "compression_model"
    elif context == "synthesis":
        return "synthesis_model"
    else:
        return "model"


def get_unsupported_params(model_name: str) -> list:
    """Get list of parameters not supported by a model."""
    unsupported = []
    
    # Google Gemini models don't support these
    if "gemini" in model_name.lower():
        unsupported.extend(["presence_penalty", "frequency_penalty"])
    
    # Add other model-specific exclusions here as needed
    
    return unsupported