"""
Unified model management command.

This module provides a single command for managing all model selections:
- chat (main conversation)
- detection (topic detection)
- compression (conversation compression)
- synthesis (web search synthesis)
"""

import typer
from typing import Optional, List
from episodic.config import config
from episodic.configuration import get_heading_color, get_system_color, get_text_color
from episodic.llm import get_model_string
from episodic.llm_config import get_available_providers, get_provider_models, LOCAL_PROVIDERS, find_provider_for_model
from episodic.model_utils import get_model_info_string, format_model_display_name
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr

# Constants
LOCAL_PROVIDERS = ["ollama", "lmstudio", "local", "localai"]
PRICING_TOKEN_COUNT = 1000

# Import cost_per_token from litellm if available
try:
    from litellm import cost_per_token
except ImportError:
    cost_per_token = None


def model_command(
    context: Optional[str] = typer.Argument(None, help="Context (chat/detection/compression/synthesis) or 'list'"),
    model_name: Optional[str] = typer.Argument(None, help="Model name to set")
):
    """
    Manage language models for different contexts.
    
    Usage:
        /model                              # Show all four models in use
        /model list                         # Show available models with pricing
        /model chat <number|full-name>      # Set chat model
        /model detection <number|full-name> # Set detection model (instruct recommended)
        /model compression <number|full-name> # Set compression model (instruct recommended)
        /model synthesis <number|full-name>  # Set synthesis model (instruct recommended)
    
    Examples:
        /model chat 3                       # Select by number from list
        /model chat gpt-4o-mini             # Select by full model name
    """
    # No arguments - show all models in use
    if not context:
        show_current_models()
        return
    
    # Handle 'list' command
    if context.lower() == "list":
        show_available_models()
        return
    
    # Validate context
    valid_contexts = ["chat", "detection", "compression", "synthesis"]
    if context.lower() not in valid_contexts:
        # Check if user is trying to set a model directly (old syntax)
        if "/" in context or context in get_all_available_models():
            typer.secho("⚠️  Please specify the context first:", fg="yellow")
            typer.secho("  /model chat <model_name>", fg=get_text_color())
            typer.secho("  /model detection <model_name>", fg=get_text_color())
            typer.secho("  /model compression <model_name>", fg=get_text_color())
            typer.secho("  /model synthesis <model_name>", fg=get_text_color())
            return
        
        typer.secho(f"Unknown context: {context}", fg="red")
        typer.secho(f"Valid contexts: {', '.join(valid_contexts)}", fg=get_text_color())
        return
    
    # If no model specified, show usage
    if not model_name:
        typer.secho(f"Please specify a model name or number.", fg="red")
        typer.secho(f"Use '/model list' to see available models.", fg=get_text_color())
        return
    
    # Set the model for the context
    set_model_for_context(context.lower(), model_name)


def show_current_models():
    """Show all four models currently in use."""
    contexts = [
        ("Chat", "model", "chat"),
        ("Detection", "topic_detection_model", "detection"),
        ("Compression", "compression_model", "compression"),
        ("Synthesis", "synthesis_model", "synthesis")
    ]
    
    typer.secho("\nCurrent models:", fg=get_heading_color(), bold=True)
    seen_types = set()  # Track which model types we've seen
    
    for description, config_key, context_name in contexts:
        current = config.get(config_key, get_default_for_context(context_name))
        model_str = get_model_string(current)
        
        # Get model info
        provider = find_provider_for_model(current)
        if not provider:
            # Try to extract provider from model name
            if "/" in current:
                provider = current.split("/")[0]
            else:
                provider = "unknown"
                
        type_indicator, tech_info = get_model_info_string(current, provider)
        seen_types.add(type_indicator)  # Collect model types for legend
        
        # Display the model info
        typer.secho(f"  {description:<12} ", fg=get_text_color(), nl=False)
        
        # Show type indicator with color
        if type_indicator == '[I]':
            typer.secho(type_indicator, nl=False, fg=typer.colors.GREEN, bold=True)
        elif type_indicator == '[C]':
            typer.secho(type_indicator, nl=False, fg=typer.colors.BLUE, bold=True)
        elif type_indicator == '[CI]':
            typer.secho(type_indicator, nl=False, fg=typer.colors.CYAN, bold=True)
        elif type_indicator == '[B]':
            typer.secho(type_indicator, nl=False, fg=typer.colors.MAGENTA, bold=True)
        else:
            typer.secho(type_indicator, nl=False, fg=get_text_color(), dim=True)
            
        typer.secho(f" {model_str}", fg=get_heading_color(), nl=False)
        
        # Show technical info if available
        if tech_info:
            typer.secho(f" {tech_info}", fg=get_text_color(), dim=True)
        else:
            typer.echo()
    
    # Add legend for model types (only show types that are actually present)
    typer.secho("\nModel Types:", fg=get_heading_color(), bold=True)
    
    type_info = {
        '[C]': ("Chat model (best for conversations)", typer.colors.BLUE),
        '[I]': ("Instruct model (best for detection/compression/synthesis)", typer.colors.GREEN),
        '[CI]': ("Chat & Instruct model (works for both)", typer.colors.CYAN),
        '[B]': ("Base/Completion model", typer.colors.MAGENTA),
        '[?]': ("Unknown type", None)  # No color, will use dim
    }
    
    # Only show types that were actually seen (C first as it's most familiar)
    for type_indicator in ['[C]', '[I]', '[CI]', '[B]', '[?]']:
        if type_indicator in seen_types:
            description, color = type_info[type_indicator]
            typer.secho("  ", nl=False)
            if color:
                typer.secho(type_indicator, fg=color, bold=True, nl=False)
            else:
                typer.secho(type_indicator, fg=get_text_color(), dim=True, nl=False)
            typer.secho(f" = {description}", fg=get_text_color())
    
    typer.secho("\nUse '/model list' to see available models", fg=get_text_color(), dim=True)


def show_available_models():
    """Show all available models with pricing information."""
    try:
        providers = get_available_providers()
        current_idx = 1
        all_models = []  # Store models for number selection
        
        # First pass: collect all model info to calculate max width
        model_info_list = []
        max_width = 0
        
        for provider_name, provider_config in providers.items():
            models = get_provider_models(provider_name)
            if models:
                hf_model_index = 0
                for model in models:
                    if isinstance(model, dict):
                        model_name = model.get("name", "unknown")
                        display_name = model.get("display_name", model_name)
                    else:
                        model_name = model
                        display_name = model
                    
                    # Get model type and tech info
                    type_indicator, tech_info = get_model_info_string(model_name, provider_name)
                    # Adjust display name width based on type indicator length
                    # Base width is 26, but reduce by 1 for 4-char indicators like [CI]
                    display_width = 26 if len(type_indicator) <= 3 else 25
                    formatted_display = format_model_display_name(display_name, display_width)
                    
                    # Calculate width for this model
                    # Base: "  XX. " = 6 chars
                    # Plus type indicator length (3 for [C]/[I]/[B], 4 for [CI], 3 for [?])
                    # Plus space after indicator = 1
                    # Plus actual formatted display length
                    # Plus tech info if present
                    width = 6 + len(type_indicator) + 1 + len(formatted_display)
                    if tech_info:
                        width += len(tech_info) + 1
                    
                    max_width = max(max_width, width)
                    
                    # Store info for second pass
                    model_info_list.append({
                        'provider': provider_name,
                        'model_name': model_name,
                        'display_name': display_name,
                        'type_indicator': type_indicator,
                        'tech_info': tech_info,
                        'formatted_display': formatted_display,
                        'hf_index': hf_model_index if provider_name == "huggingface" else None
                    })
                    
                    if provider_name == "huggingface":
                        hf_model_index += 1
        
        # Add 2 spaces buffer after the longest entry
        target_column = max_width + 2
        
        # Second pass: display with proper alignment
        current_provider = None
        current_idx = 1
        seen_types = set()  # Track which model types we've seen
        
        for info in model_info_list:
            # Collect model types for legend
            seen_types.add(info['type_indicator'])
            # Display provider header if changed
            if info['provider'] != current_provider:
                current_provider = info['provider']
                typer.secho(f"\nAvailable models from ", nl=False, fg=get_heading_color(), bold=True)
                typer.secho(f"{current_provider}:", fg=get_heading_color(), bold=True)
            
            # Store for number selection
            all_models.append(info['model_name'])
            
            # Get pricing info
            pricing = get_pricing_for_model(info['model_name'], info['provider'], info['hf_index'])
            
            # Display the model
            typer.secho(f"  ", nl=False)
            typer.secho(f"{current_idx:2d}", nl=False, fg=typer.colors.BRIGHT_YELLOW, bold=True)
            typer.secho(f". ", nl=False, fg=get_text_color())
            
            # Show type indicator with color
            if info['type_indicator'] == '[I]':
                typer.secho(info['type_indicator'], nl=False, fg=typer.colors.GREEN, bold=True)
            elif info['type_indicator'] == '[C]':
                typer.secho(info['type_indicator'], nl=False, fg=typer.colors.BLUE, bold=True)
            elif info['type_indicator'] == '[CI]':
                typer.secho(info['type_indicator'], nl=False, fg=typer.colors.CYAN, bold=True)
            elif info['type_indicator'] == '[B]':
                typer.secho(info['type_indicator'], nl=False, fg=typer.colors.MAGENTA, bold=True)
            else:
                typer.secho(info['type_indicator'], nl=False, fg=get_text_color(), dim=True)
            
            typer.secho(f" {info['formatted_display']}", nl=False, fg=typer.colors.BRIGHT_CYAN, bold=True)
            
            # Show technical info if available
            if info['tech_info']:
                typer.secho(f" {info['tech_info']}", nl=False, fg=get_text_color(), dim=True)
            
            # Calculate padding for alignment
            # Base: "  XX. " = 6 chars
            # Plus actual type indicator length
            # Plus space after indicator = 1
            # Plus actual formatted display length
            current_length = 6 + len(info['type_indicator']) + 1 + len(info['formatted_display'])
            if info['tech_info']:
                current_length += len(info['tech_info']) + 1
            
            padding_needed = max(1, target_column - current_length)
            padding = " " * padding_needed
            
            # Only show pricing info if not empty
            if pricing:
                typer.secho(f"{padding}(", nl=False, fg=get_text_color())
                if pricing == "Local model":
                    typer.secho(f"{pricing}", nl=False, fg=typer.colors.BRIGHT_GREEN, bold=True)
                elif pricing == "Pricing not available":
                    typer.secho(f"{pricing}", nl=False, fg=typer.colors.YELLOW)
                elif "Free tier" in pricing:
                    typer.secho(f"{pricing}", nl=False, fg=typer.colors.GREEN)
                elif "Pro tier" in pricing:
                    typer.secho(f"{pricing}", nl=False, fg=typer.colors.BRIGHT_BLUE, bold=True)
                else:
                    typer.secho(f"{pricing}", nl=False, fg=typer.colors.BRIGHT_MAGENTA, bold=True)
                typer.secho(")", fg=get_text_color())
            else:
                # No pricing info, just add newline
                typer.echo()
            
            current_idx += 1
    
    except Exception as e:
        typer.echo(f"Error getting model list: {str(e)}")
    
    # Add legend for model types (only show types that are actually present)
    typer.secho("\nModel Types:", fg=get_heading_color(), bold=True)
    
    type_info = {
        '[C]': ("Chat model (best for conversations)", typer.colors.BLUE),
        '[I]': ("Instruct model (best for detection/compression/synthesis)", typer.colors.GREEN),
        '[CI]': ("Chat & Instruct model (works for both)", typer.colors.CYAN),
        '[B]': ("Base/Completion model", typer.colors.MAGENTA),
        '[?]': ("Unknown type", None)  # No color, will use dim
    }
    
    # Only show types that were actually seen (C first as it's most familiar)
    for type_indicator in ['[C]', '[I]', '[CI]', '[B]', '[?]']:
        if type_indicator in seen_types:
            description, color = type_info[type_indicator]
            typer.secho("  ", nl=False)
            if color:
                typer.secho(type_indicator, fg=color, bold=True, nl=False)
            else:
                typer.secho(type_indicator, fg=get_text_color(), dim=True, nl=False)
            typer.secho(f" = {description}", fg=get_text_color())
    
    typer.secho("\nTo change a model:", fg=get_text_color())
    typer.secho("  /model chat <number|full-model-name>", fg=get_system_color())
    typer.secho("  /model detection <number|full-model-name>", fg=get_system_color())
    typer.secho("  /model compression <number|full-model-name>", fg=get_system_color())
    typer.secho("  /model synthesis <number|full-model-name>", fg=get_system_color())
    typer.secho("\nExamples:", fg=get_text_color(), dim=True)
    typer.secho("  /model chat 7                       # Select by number from list", fg=get_text_color(), dim=True)
    typer.secho("  /model chat claude-opus-4-20250514  # Select by full model name", fg=get_text_color(), dim=True)


def get_pricing_for_model(model_name: str, provider_name: str, hf_index: Optional[int] = None) -> str:
    """Get pricing information for a model."""
    # Try to get pricing information using cost_per_token
    if cost_per_token:
        try:
            # Suppress both warnings and stdout/stderr output from LiteLLM during pricing lookup
            with warnings.catch_warnings(), \
                 redirect_stdout(io.StringIO()), \
                 redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                # Calculate cost for 1000 tokens (both input and output separately)
                input_cost_raw = cost_per_token(model=model_name, prompt_tokens=PRICING_TOKEN_COUNT, completion_tokens=0)
                output_cost_raw = cost_per_token(model=model_name, prompt_tokens=0, completion_tokens=PRICING_TOKEN_COUNT)

            # Handle tuple results (sum if tuple, use directly if scalar)
            input_cost = sum(input_cost_raw) if isinstance(input_cost_raw, tuple) else input_cost_raw
            output_cost = sum(output_cost_raw) if isinstance(output_cost_raw, tuple) else output_cost_raw

            if input_cost or output_cost:
                return f"${input_cost:.6f}/1K in, ${output_cost:.6f}/1K out"
        except Exception:
            pass
    
    # Special handling for HuggingFace models
    if provider_name == "huggingface" and hf_index is not None:
        if hf_index == 0:
            return "Free tier: ~30K tokens/month"
        elif hf_index == 1:
            return "Pro tier: $9/month unlimited"
        else:
            return ""  # No pricing for rest
    elif provider_name in LOCAL_PROVIDERS:
        return "Local model"
    else:
        return "Pricing not available"


def show_model_for_context(context: str):
    """Show the current model for a specific context."""
    config_keys = {
        "chat": "model",
        "detection": "topic_detection_model",
        "compression": "compression_model",
        "synthesis": "synthesis_model"
    }
    
    descriptions = {
        "chat": "Chat model",
        "detection": "Topic detection model",
        "compression": "Compression model",
        "synthesis": "Web synthesis model"
    }
    
    config_key = config_keys[context]
    default = get_default_for_context(context)
    current = config.get(config_key, default)
    model_str = get_model_string(current)
    
    typer.secho(f"{descriptions[context]}: {model_str}", fg=get_heading_color())


def set_model_for_context(context: str, model_name: str):
    """Set the model for a specific context."""
    config_keys = {
        "chat": "model",
        "detection": "topic_detection_model",
        "compression": "compression_model",
        "synthesis": "synthesis_model"
    }
    
    descriptions = {
        "chat": "chat",
        "detection": "topic detection",
        "compression": "compression",
        "synthesis": "web synthesis"
    }
    
    # Check if model_name is a number
    try:
        model_index = int(model_name)
        # Build the same model list to map index to model name
        providers = get_available_providers()
        current_idx = 1
        selected_model = None
        
        for provider_name, provider_config in providers.items():
            models = get_provider_models(provider_name)
            if models:
                for model in models:
                    if isinstance(model, dict):
                        model_full_name = model.get("name", "unknown")
                    else:
                        model_full_name = model
                    
                    if current_idx == model_index:
                        selected_model = model_full_name
                        break
                    current_idx += 1
                if selected_model:
                    break
        
        if selected_model:
            model_name = selected_model
        else:
            typer.secho(f"Invalid model number '{model_index}'. Use '/model list' to see available models.", fg="red")
            return
            
    except ValueError:
        # Not a number, use as model name
        pass
    
    # Validate model exists
    if not validate_model_exists(model_name):
        typer.secho(f"Model not found: {model_name}", fg="red")
        typer.secho("Use '/model list' to see available models", fg=get_text_color())
        return
    
    # Set the model
    config_key = config_keys[context]
    config.set(config_key, model_name)
    
    # Clear any cached parameters that might be incompatible
    validate_and_clear_incompatible_params(context, model_name)
    
    model_str = get_model_string(model_name)
    typer.secho(f"✓ {descriptions[context].capitalize()} model set to: {model_str}", fg="green")


def get_default_for_context(context: str) -> str:
    """Get the default model for a context."""
    defaults = {
        "chat": "gpt-3.5-turbo",
        "detection": "ollama/phi3",
        "compression": "ollama/phi3",
        "synthesis": "ollama/phi3"
    }
    return defaults.get(context, "gpt-3.5-turbo")


def validate_model_exists(model_name: str) -> bool:
    """Check if a model exists in available providers."""
    all_models = get_all_available_models()
    return model_name in all_models


def get_all_available_models() -> List[str]:
    """Get all available models from all providers."""
    models = []
    providers = get_available_providers()
    
    for provider_name, provider_config in providers.items():
        provider_models = get_provider_models(provider_name)
        if provider_models:
            for model in provider_models:
                if isinstance(model, dict):
                    model_name = model.get("name", "unknown")
                else:
                    model_name = model
                models.append(model_name)
    
    return models


def validate_and_clear_incompatible_params(context: str, model_name: str):
    """Clear parameters that are incompatible with the new model."""
    # Parameters that some models don't support
    google_unsupported = ["presence_penalty", "frequency_penalty"]
    
    if "gemini" in model_name.lower():
        # Clear unsupported parameters for Google models
        param_key = f"{context}_params"
        current_params = config.get(param_key, {})
        
        if isinstance(current_params, dict):
            modified = False
            for param in google_unsupported:
                if param in current_params:
                    del current_params[param]
                    modified = True
            
            if modified:
                config.set(param_key, current_params)
                typer.secho(
                    f"  ℹ️  Removed unsupported parameters for {model_name}: {', '.join(google_unsupported)}", 
                    fg="yellow"
                )
