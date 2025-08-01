"""
Model selection and management commands for the Episodic CLI.
"""

import typer
import warnings
import io
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr
from episodic.config import config
from episodic.configuration import get_system_color, get_text_color, get_heading_color
from episodic.llm_config import (
    get_available_providers, get_provider_models, get_current_provider,
    validate_model_selection
)

# Constants
LOCAL_PROVIDERS = ["ollama", "lmstudio", "local", "localai"]
PRICING_TOKEN_COUNT = 1000

# Import cost_per_token from litellm if available
try:
    from litellm import cost_per_token
except ImportError:
    cost_per_token = None


def handle_model(name: Optional[str] = None):
    """Switch to a different language model or show current model."""
    if name is None:
        # Show current model
        current_model = config.get("model", "gpt-3.5-turbo")
        
        # Find the actual provider for this model
        actual_provider = None
        providers = get_available_providers()
        for provider_name, provider_config in providers.items():
            models = get_provider_models(provider_name)
            if models:
                for model in models:
                    if isinstance(model, dict):
                        model_name = model.get("name", "unknown")
                    else:
                        model_name = model
                    if model_name == current_model:
                        actual_provider = provider_name
                        break
                if actual_provider:
                    break
        
        if not actual_provider:
            actual_provider = get_current_provider()
        
        typer.secho(f"Current model: ", nl=False, fg=get_text_color())
        typer.secho(f"{current_model}", nl=False, fg=get_system_color())
        typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
        typer.secho(f"{actual_provider}", nl=False, fg=get_system_color())
        typer.secho(")", fg=get_text_color())

        # Get provider models using our own configuration
        try:
            providers = get_available_providers()
            current_idx = 1

            for provider_name, provider_config in providers.items():
                models = get_provider_models(provider_name)
                if models:
                    typer.secho(f"\nAvailable models from ", nl=False, fg=get_heading_color(), bold=True)
                    typer.secho(f"{provider_name}:", fg=get_heading_color(), bold=True)

                    for model in models:
                        if isinstance(model, dict):
                            model_name = model.get("name", "unknown")
                        else:
                            model_name = model

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
                                    pricing = f"${input_cost*1000:.2f}/1M input, ${output_cost*1000:.2f}/1M output"
                                else:
                                    # For local providers, show "Local model" instead of "Pricing not available"
                                    if provider_name in LOCAL_PROVIDERS:
                                        pricing = "Local model"
                                    else:
                                        pricing = "Pricing not available"
                            except Exception:
                                # For local providers, show "Local model" instead of "Pricing not available"
                                if provider_name in LOCAL_PROVIDERS:
                                    pricing = "Local model"
                                else:
                                    pricing = "Pricing not available"
                        else:
                            # cost_per_token not available
                            if provider_name in LOCAL_PROVIDERS:
                                pricing = "Local model"
                            else:
                                pricing = "Pricing not available"

                        typer.secho(f"  ", nl=False)
                        typer.secho(f"{current_idx:2d}", nl=False, fg=typer.colors.BRIGHT_YELLOW, bold=True)
                        typer.secho(f". ", nl=False, fg=get_text_color())
                        typer.secho(f"{model_name:20s}", nl=False, fg=typer.colors.BRIGHT_CYAN, bold=True)
                        typer.secho(f"\t(", nl=False, fg=get_text_color())
                        if pricing == "Local model":
                            typer.secho(f"{pricing}", nl=False, fg=typer.colors.BRIGHT_GREEN, bold=True)
                        elif pricing == "Pricing not available":
                            typer.secho(f"{pricing}", nl=False, fg=typer.colors.YELLOW)
                        else:
                            typer.secho(f"{pricing}", nl=False, fg=typer.colors.BRIGHT_MAGENTA, bold=True)
                        typer.secho(")", fg=get_text_color())
                        current_idx += 1

        except Exception as e:
            typer.echo(f"Error getting model list: {str(e)}")
        return

    # Handle model changes - check if input is a number first
    try:
        model_index = int(name)
        # Build the same model list to map index to model name
        providers = get_available_providers()
        current_idx = 1
        selected_model = None
        
        for provider_name, provider_config in providers.items():
            models = get_provider_models(provider_name)
            if models:
                for model in models:
                    if isinstance(model, dict):
                        model_name = model.get("name", "unknown")
                    else:
                        model_name = model
                    
                    if current_idx == model_index:
                        selected_model = model_name
                        break
                    current_idx += 1
                if selected_model:
                    break
        
        if selected_model:
            # Use the selected model name
            name = selected_model
        else:
            typer.secho(f"Error: Invalid model number '{model_index}'. Use '/model' to see available models.", fg=typer.colors.RED, bold=True)
            return
            
    except ValueError:
        # Not a number, treat as model name
        pass
    
    # Now handle model change with the resolved model name
    try:
        # Validate that the model exists and has API key
        validate_model_selection(name)
        
        # Set the model in our config (single source of truth)
        config.set("model", name)
        
        # Get the provider based on the model
        provider = get_current_provider()
        
        # Display model with pricing
        try:
            if cost_per_token:
                # Calculate cost for 1000 tokens (both input and output separately)
                input_cost_raw = cost_per_token(model=name, prompt_tokens=PRICING_TOKEN_COUNT, completion_tokens=0)
                output_cost_raw = cost_per_token(model=name, prompt_tokens=0, completion_tokens=PRICING_TOKEN_COUNT)
                
                # Handle tuple results (sum if tuple, use directly if scalar)
                input_cost = sum(input_cost_raw) if isinstance(input_cost_raw, tuple) else input_cost_raw
                output_cost = sum(output_cost_raw) if isinstance(output_cost_raw, tuple) else output_cost_raw
                
                if input_cost or output_cost:
                    typer.secho(f"Switched to model: ", nl=False, fg=get_text_color(), bold=True)
                    typer.secho(f"{name}", nl=False, fg=typer.colors.BRIGHT_CYAN, bold=True)
                    typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
                    typer.secho(f"{provider}", nl=False, fg=typer.colors.BRIGHT_YELLOW, bold=True)
                    typer.secho(")", fg=get_text_color())
                    typer.secho(f"Pricing: ", nl=False, fg=get_text_color(), bold=True)
                    typer.secho(f"${input_cost*1000:.2f}/1M input, ${output_cost*1000:.2f}/1M output", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
                else:
                    typer.secho(f"Switched to model: ", nl=False, fg=get_text_color(), bold=True)
                    typer.secho(f"{name}", nl=False, fg=typer.colors.BRIGHT_CYAN, bold=True)
                    typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
                    typer.secho(f"{provider}", nl=False, fg=typer.colors.BRIGHT_YELLOW, bold=True)
                    typer.secho(")", fg=get_text_color())
                    typer.secho("Pricing: ", nl=False, fg=get_text_color(), bold=True)
                    # Check if it's a local provider
                    if provider in LOCAL_PROVIDERS:
                        typer.secho("Local model", fg=typer.colors.BRIGHT_GREEN, bold=True)
                    else:
                        typer.secho("Not available", fg=typer.colors.YELLOW)
            else:
                typer.secho(f"Switched to model: ", nl=False, fg=get_text_color(), bold=True)
                typer.secho(f"{name}", nl=False, fg=typer.colors.BRIGHT_CYAN, bold=True)
                typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
                typer.secho(f"{provider}", nl=False, fg=typer.colors.BRIGHT_YELLOW, bold=True)
                typer.secho(")", fg=get_text_color())
                typer.secho("Pricing: ", nl=False, fg=get_text_color(), bold=True)
                # Check if it's a local provider
                if provider in LOCAL_PROVIDERS:
                    typer.secho("Local model", fg=typer.colors.BRIGHT_GREEN, bold=True)
                else:
                    typer.secho("Not available", fg=typer.colors.YELLOW)
        except Exception:
            typer.secho(f"Switched to model: ", nl=False, fg=get_text_color(), bold=True)
            typer.secho(f"{name}", nl=False, fg=typer.colors.BRIGHT_CYAN, bold=True)
            typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
            typer.secho(f"{provider}", nl=False, fg=typer.colors.BRIGHT_YELLOW, bold=True)
            typer.secho(")", fg=get_text_color())
            typer.secho("Pricing: ", nl=False, fg=get_text_color(), bold=True)
            # Check if it's a local provider
            if provider in LOCAL_PROVIDERS:
                typer.secho("Local model", fg=typer.colors.BRIGHT_GREEN, bold=True)
            else:
                typer.secho("Not available", fg=typer.colors.YELLOW)
    except ValueError as e:
        typer.secho(f"Error: {str(e)}", fg=typer.colors.RED, bold=True)