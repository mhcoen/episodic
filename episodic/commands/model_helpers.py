"""Model helper functions (pricing, per-context selection, validation).

Split out of unified_model.py to keep it under the size limit. These are leaf
helpers: they do not call the display functions in unified_model.py, so there
is no import cycle.
"""

import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, List

import typer

from episodic.config import config
from episodic.configuration import get_heading_color, get_system_color, get_text_color
from episodic.llm import get_model_string
from episodic.llm_config import (
    get_available_providers, get_provider_models, LOCAL_PROVIDERS, find_provider_for_model,
)
from episodic.model_utils import get_model_info_string, format_model_display_name

# Import cost_per_token from litellm if available
try:
    from litellm import cost_per_token
except Exception:
    cost_per_token = None

PRICING_TOKEN_COUNT = 1000


def get_pricing_for_model(model_name: str, provider_name: str, hf_index: Optional[int] = None) -> str:
    """Get pricing information for a model."""
    # First check if we have pricing in models config (user + template fallback)
    from episodic.model_config import get_model_config
    mc = get_model_config()
    model_info = mc.get_model_info(provider_name, model_name)
    if model_info:
        pricing = model_info.get('pricing')
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
        "synthesis": "synthesis_model",
        "critic": "critic_model"
    }

    descriptions = {
        "chat": "Chat model",
        "detection": "Topic detection model",
        "compression": "Compression model",
        "synthesis": "Web synthesis model",
        "critic": "Critic model"
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
        "synthesis": "synthesis_model",
        "intent": "intent_model",
        "critic": "critic_model",
        "extraction": "extraction_model",
    }

    descriptions = {
        "chat": "chat",
        "detection": "topic detection",
        "compression": "compression",
        "synthesis": "web synthesis",
        "intent": "intent classification",
        "critic": "critic",
        "extraction": "KG extraction",
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

    # Handle custom local models differently - verify file exists instead of API call
    if model_name.startswith("custom/"):
        from episodic.model_config import get_model_config
        mc = get_model_config()
        model_path = mc.get_model_path(model_name)

        if not model_path:
            typer.secho(f"✗ Model '{model_name}' not found in models.json", fg="red")
            typer.secho("Add it to ~/.episodic/models.json under the 'custom' provider", fg=get_text_color())
            return

        if not os.path.exists(model_path):
            typer.secho(f"✗ Model file not found: {model_path}", fg="red")
            return

        typer.secho(f"✓ Found local model at: {model_path}", fg="green", dim=True)
    else:
        # Check if model is in our local list
        in_local_list = validate_model_exists(model_name)

        if not in_local_list:
            typer.secho(f"Model '{model_name}' not in local list, verifying with API...", fg="yellow")

        # Verify model actually works with API
        typer.secho("Verifying model...", fg=get_text_color(), dim=True)
        valid, error_msg = verify_model_with_api(model_name)

        if not valid:
            typer.secho(f"✗ {error_msg}", fg="red")
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
        "chat": "gpt-4o-mini",
        "detection": "custom/topic-boundary-distilbert",
        "compression": "ollama/phi4",
        "synthesis": "ollama/phi4",
        "intent": "gpt-4o-mini",
        "critic": "anthropic/claude-opus-4-5-20251101",
        "extraction": "gpt-4o-mini",
    }
    return defaults.get(context, "gpt-4o-mini")


def validate_model_exists(model_name: str) -> bool:
    """Check if a model exists in available providers."""
    all_models = get_all_available_models()
    return model_name in all_models


def verify_model_with_api(model_name: str) -> tuple[bool, str]:
    """
    Verify a model exists by making a minimal API call.
    Returns (success, error_message).
    """
    from episodic.llm import _execute_llm_query

    try:
        # Minimal request - just enough to verify model exists
        _execute_llm_query(
            messages=[{"role": "user", "content": "hi"}],
            model=model_name,
            stream=False,
            max_tokens=1
        )
        return True, ""
    except Exception as e:
        error_str = str(e).lower()
        if "not_found_error" in error_str or "model:" in error_str or "does not exist" in error_str:
            return False, "Model not found - the model ID may have changed."
        elif "api_key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            return False, "Authentication error - check your API key."
        elif "invalid_request" in error_str:
            # Model exists but request had issues - that's ok for validation
            return True, ""
        else:
            # Unknown error - assume model might exist
            return True, f"Warning: {e}"


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
