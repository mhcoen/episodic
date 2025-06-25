"""
Model selection and management commands for the Episodic CLI.
"""

import typer
import warnings
import io
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr
from episodic.config import config
from episodic.configuration import get_system_color, get_heading_color, get_text_color
from episodic.llm_config import (
    get_available_providers, get_provider_models, get_current_provider,
    set_default_model, get_default_model
)

# Constants
LOCAL_PROVIDERS = ["ollama", "localai"]
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
        current_model = config.get("model", get_default_model())
        typer.secho(f"Current model: ", nl=False, fg=get_text_color())
        typer.secho(f"{current_model}", nl=False, fg=get_system_color())
        typer.secho(f" (Provider: ", nl=False, fg=get_text_color())
        typer.secho(f"{get_current_provider()}", nl=False, fg=get_system_color())
        typer.secho(")", fg=get_text_color())

        # Get provider models using our own configuration
        try:
            providers = get_available_providers()
            current_idx = 1

            for provider_name, provider_config in providers.items():
                models = get_provider_models(provider_name)
                if models:
                    typer.echo(f"\nAvailable models from {provider_name}:")

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
                                    pricing = f"${input_cost:.6f}/1K input, ${output_cost:.6f}/1K output"
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

                        typer.secho(f"  ", nl=False, fg=get_text_color())
                        typer.secho(f"{current_idx:2d}", nl=False, fg=get_system_color())
                        typer.secho(f". ", nl=False, fg=get_text_color())
                        typer.secho(f"{model_name:20s}", nl=False, fg=get_system_color())
                        typer.secho(f"\t({pricing})", fg=get_text_color())
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
        selected_provider = None
        
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
                        selected_provider = provider_name
                        break
                    current_idx += 1
                if selected_model:
                    break
        
        if selected_model:
            # Use the selected model name
            name = selected_model
        else:
            typer.echo(f"Error: Invalid model number '{model_index}'. Use '/model' to see available models.")
            return
            
    except ValueError:
        # Not a number, treat as model name
        pass
    
    # Now handle model change with the resolved model name
    try:
        set_default_model(name)
        config.set("model", name)
        provider = get_current_provider()
        typer.echo(f"Switched to model: {name} (Provider: {provider})")
    except ValueError as e:
        typer.echo(f"Error: {str(e)}")