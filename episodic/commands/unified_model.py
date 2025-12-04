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
    context: Optional[str] = typer.Argument(None, help="Context (chat/detection/compression/synthesis), 'list', or model number"),
    model_name: Optional[str] = typer.Argument(None, help="Model name to set")
):
    """
    Manage language models for different contexts.

    Usage:
        /model                              # Show all four models in use
        /model list                         # Show available models with pricing
        /model <number>                     # Show detailed info about model #number
        /model chat <number|full-name>      # Set chat model
        /model detection <number|full-name> # Set detection model (instruct recommended)
        /model compression <number|full-name> # Set compression model (instruct recommended)
        /model synthesis <number|full-name>  # Set synthesis model (instruct recommended)

    Examples:
        /model 5                            # Show info about model #5
        /model chat 9                       # Select by number from list
        /model chat claude-opus-4-1-20250805  # Select by full model name
    """
    # No arguments - show all models in use
    if not context:
        show_current_models()
        return

    # Handle 'list' command
    if context.lower() == "list":
        show_available_models()
        return

    # Check if context is a number (show model info)
    try:
        model_number = int(context)
        show_model_info(model_number)
        return
    except ValueError:
        pass  # Not a number, continue with normal flow
    
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
        current = config.get(config_key)
        
        # Handle None values - use defaults
        if current is None:
            current = get_default_for_context(context_name)
        
        model_str = get_model_string(current)
        
        # Get model info
        provider = find_provider_for_model(current)
        if not provider:
            # Try to extract provider from model name
            if current and "/" in current:
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
    typer.secho("  /model chat 9                         # Select by number from list", fg=get_text_color(), dim=True)
    typer.secho("  /model chat claude-opus-4-1-20250805  # Select by full model name", fg=get_text_color(), dim=True)


def show_model_info(model_number: int):
    """Show detailed information about a specific model by its number."""
    from episodic.model_utils import get_models_config

    # Build the numbered model list (same logic as show_available_models)
    providers = get_available_providers()
    current_idx = 1
    target_model = None
    target_provider = None
    target_provider_display = None

    for provider_name, provider_config in providers.items():
        models = get_provider_models(provider_name)
        if models:
            for model in models:
                if current_idx == model_number:
                    target_model = model
                    target_provider = provider_name
                    target_provider_display = provider_config.get("display_name", provider_name)
                    break
                current_idx += 1
            if target_model:
                break

    if not target_model:
        typer.secho(f"Model #{model_number} not found.", fg="red")
        typer.secho(f"Use '/model list' to see available models (1-{current_idx-1}).", fg=get_text_color())
        return

    # Extract model data - for string models, look up full data from models.json
    models_config = get_models_config()

    if isinstance(target_model, dict):
        model_data = target_model
    else:
        # Look up full model data from models.json (string models lose extended fields)
        model_data = None
        for prov_name, prov_data in models_config.get('providers', {}).items():
            for m in prov_data.get('models', []):
                if isinstance(m, dict) and m.get('name') == target_model:
                    model_data = m
                    break
            if model_data:
                break
        if not model_data:
            model_data = {"name": target_model}

    model_name = model_data.get("name", target_model if isinstance(target_model, str) else "unknown")
    display_name = model_data.get("display_name", model_name)
    model_type = model_data.get("type", "unknown")
    parameters = model_data.get("parameters")
    context_window = model_data.get("context_window")
    pricing = model_data.get("pricing")
    creator = model_data.get("creator", target_provider_display)
    released = model_data.get("released")
    description = model_data.get("description")
    strengths = model_data.get("strengths", [])
    weaknesses = model_data.get("weaknesses", [])
    recommended_for = model_data.get("recommended_for", [])

    # Get type indicator
    type_indicator, tech_info = get_model_info_string(model_name, target_provider)

    # Map type indicators to full descriptions
    type_descriptions = {
        '[C]': 'Chat',
        '[I]': 'Instruct',
        '[CI]': 'Chat & Instruct',
        '[B]': 'Base/Completion',
        '[?]': 'Unknown'
    }
    type_desc = type_descriptions.get(type_indicator, model_type.capitalize())

    # Print header
    header_text = f"{display_name}"
    number_text = f"#{model_number}"
    # Calculate padding to right-align the number
    total_width = 60
    padding = total_width - len(header_text) - len(number_text)
    padding = max(1, padding)

    typer.echo()
    typer.secho(f"{header_text}{' ' * padding}{number_text}", fg=get_heading_color(), bold=True)
    typer.secho("━" * total_width, fg=get_text_color(), dim=True)

    # Basic info fields
    typer.secho("Creator:        ", fg=get_text_color(), nl=False)
    typer.secho(creator or "Unknown", fg=get_heading_color())

    typer.secho("Type:           ", fg=get_text_color(), nl=False)
    if type_indicator == '[I]':
        typer.secho(f"{type_desc} ", nl=False, fg=typer.colors.GREEN, bold=True)
        typer.secho(type_indicator, fg=typer.colors.GREEN, bold=True)
    elif type_indicator == '[C]':
        typer.secho(f"{type_desc} ", nl=False, fg=typer.colors.BLUE, bold=True)
        typer.secho(type_indicator, fg=typer.colors.BLUE, bold=True)
    elif type_indicator == '[CI]':
        typer.secho(f"{type_desc} ", nl=False, fg=typer.colors.CYAN, bold=True)
        typer.secho(type_indicator, fg=typer.colors.CYAN, bold=True)
    elif type_indicator == '[B]':
        typer.secho(f"{type_desc} ", nl=False, fg=typer.colors.MAGENTA, bold=True)
        typer.secho(type_indicator, fg=typer.colors.MAGENTA, bold=True)
    else:
        typer.secho(f"{type_desc} {type_indicator}", fg=get_text_color())

    typer.secho("Parameters:     ", fg=get_text_color(), nl=False)
    typer.secho(parameters or "Unknown", fg=get_heading_color())

    typer.secho("Context:        ", fg=get_text_color(), nl=False)
    if context_window:
        typer.secho(f"{context_window:,} tokens", fg=get_heading_color())
    elif target_provider in LOCAL_PROVIDERS:
        typer.secho("(detect on use)", fg=get_text_color(), dim=True)
    else:
        typer.secho("Unknown", fg=get_text_color(), dim=True)

    # Pricing
    typer.secho("Pricing:        ", fg=get_text_color(), nl=False)
    if target_provider in LOCAL_PROVIDERS:
        typer.secho("Free (local)", fg=typer.colors.BRIGHT_GREEN, bold=True)
    elif pricing:
        input_cost = pricing.get('input', 0)
        output_cost = pricing.get('output', 0)
        unit = pricing.get('unit', 'per_1m_tokens')
        if unit == 'per_1m_tokens':
            typer.secho(f"${input_cost:.2f}/1M in, ${output_cost:.2f}/1M out", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
        else:
            typer.secho(f"${input_cost*1000:.2f}/1M in, ${output_cost*1000:.2f}/1M out", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
    else:
        typer.secho("Not available", fg=get_text_color(), dim=True)

    # Released date (if available)
    if released:
        typer.secho("Released:       ", fg=get_text_color(), nl=False)
        typer.secho(released, fg=get_heading_color())

    # API name
    typer.secho("API name:       ", fg=get_text_color(), nl=False)
    typer.secho(model_name, fg=get_text_color(), dim=True)

    # Description
    typer.echo()
    if description:
        # Word wrap the description at ~60 chars
        import textwrap
        wrapped = textwrap.fill(description, width=60)
        typer.secho(wrapped, fg=get_text_color())
    else:
        typer.secho("No detailed description available for this model.", fg=get_text_color(), dim=True)

    # Strengths
    if strengths:
        typer.echo()
        typer.secho("Strengths:", fg=get_heading_color(), bold=True)
        for strength in strengths:
            typer.secho(f"  + {strength}", fg=typer.colors.GREEN)

    # Weaknesses
    if weaknesses:
        typer.echo()
        typer.secho("Weaknesses:", fg=get_heading_color(), bold=True)
        for weakness in weaknesses:
            typer.secho(f"  - {weakness}", fg=typer.colors.RED)

    # Recommended for
    if recommended_for:
        typer.echo()
        typer.secho("Recommended for: ", fg=get_text_color(), nl=False)
        typer.secho(", ".join(recommended_for), fg=typer.colors.CYAN, bold=True)

    typer.echo()


def get_pricing_for_model(model_name: str, provider_name: str, hf_index: Optional[int] = None) -> str:
    """Get pricing information for a model."""
    # First check if we have custom pricing in models.json
    from episodic.model_utils import get_models_config
    models_data = get_models_config()
    
    # Search for the model in models.json
    for provider_key, provider_data in models_data.get('providers', {}).items():
        for model in provider_data.get('models', []):
            if model.get('name') == model_name:
                pricing = model.get('pricing')
                if pricing:
                    input_cost = pricing.get('input', 0)
                    output_cost = pricing.get('output', 0)
                    if input_cost > 0 or output_cost > 0:
                        unit = pricing.get('unit', 'per_1k_tokens')
                        if unit == 'per_1m_tokens':
                            return f"${input_cost:.2f}/1M in, ${output_cost:.2f}/1M out"
                        else:  # per_1k_tokens
                            return f"${input_cost*1000:.2f}/1M in, ${output_cost*1000:.2f}/1M out"
    
    # Check if this is an OpenRouter model
    if model_name.startswith("openrouter/"):
        # Get pricing from OpenRouter API
        from episodic.openrouter_pricing import get_openrouter_pricing
        or_pricing = get_openrouter_pricing()
        
        # Strip the openrouter/ prefix
        model_id = model_name.replace("openrouter/", "")
        pricing_info = or_pricing.get_pricing(model_id)
        
        if pricing_info and (pricing_info[0] > 0 or pricing_info[1] > 0):
            # OpenRouter pricing is per 1K tokens
            return f"${pricing_info[0]*1000:.2f}/1M in, ${pricing_info[1]*1000:.2f}/1M out"
    
    # Try to get pricing information using cost_per_token for non-OpenRouter models
    if cost_per_token and not model_name.startswith("openrouter/"):
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
                # LiteLLM returns cost for PRICING_TOKEN_COUNT tokens
                # Convert to per 1M tokens
                multiplier = 1000000 / PRICING_TOKEN_COUNT
                return f"${input_cost*multiplier:.2f}/1M in, ${output_cost*multiplier:.2f}/1M out"
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
