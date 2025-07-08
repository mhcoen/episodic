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
from episodic.llm_config import get_available_providers, get_provider_models


def model_command(
    context: Optional[str] = typer.Argument(None, help="Context (chat/detection/compression/synthesis) or 'list'"),
    model_name: Optional[str] = typer.Argument(None, help="Model name to set")
):
    """
    Manage language models for different contexts.
    
    Usage:
        /model                          # Show current chat model
        /model list                     # Show all models for all contexts
        /model chat gpt-4              # Set chat model to gpt-4
        /model detection ollama/llama3  # Set detection model
        /model compression gpt-3.5-turbo # Set compression model
        /model synthesis claude-3-haiku  # Set synthesis model
    """
    # No arguments - show current chat model
    if not context:
        current = config.get("model", "gpt-3.5-turbo")
        model_str = get_model_string(current)
        typer.secho(f"Current chat model: {model_str}", fg=get_heading_color())
        typer.secho("\nTo change models:", fg=get_text_color())
        typer.secho("  /model chat              ", fg=get_system_color(), nl=False)
        typer.secho("# Select a new chat model", fg=get_text_color(), dim=True)
        typer.secho("  /model list              ", fg=get_system_color(), nl=False)
        typer.secho("# See all model contexts", fg=get_text_color(), dim=True)
        return
    
    # Handle 'list' command
    if context.lower() == "list":
        show_all_models()
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
    
    # If no model specified, show available models and allow selection
    if not model_name:
        show_and_select_model_for_context(context.lower())
        return
    
    # Set the model for the context
    set_model_for_context(context.lower(), model_name)


def show_all_models():
    """Show all models for all contexts."""
    typer.secho("\n📊 Model Configuration:", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    contexts = [
        ("chat", "model", "Main conversation"),
        ("detection", "topic_detection_model", "Topic detection"),
        ("compression", "compression_model", "Compression/summarization"),
        ("synthesis", "synthesis_model", "Web search synthesis")
    ]
    
    for context_name, config_key, description in contexts:
        current = config.get(config_key, get_default_for_context(context_name))
        model_str = get_model_string(current)
        
        typer.secho(f"\n{description}:", fg=get_text_color(), bold=True)
        typer.secho(f"  {context_name:<12} ", fg=get_system_color(), nl=False)
        typer.secho(f"{model_str}", fg=get_heading_color())
    
    typer.secho("\n💡 To change a model:", fg=get_text_color())
    typer.secho("  /model chat              ", fg=get_system_color(), nl=False)
    typer.secho("# Interactive selection", fg=get_text_color(), dim=True)
    typer.secho("  /model chat gpt-4        ", fg=get_system_color(), nl=False)
    typer.secho("# Direct selection", fg=get_text_color(), dim=True)
    typer.secho("  /model detection         ", fg=get_system_color(), nl=False)
    typer.secho("# Interactive selection", fg=get_text_color(), dim=True)
    typer.secho("  /model compression       ", fg=get_system_color(), nl=False)
    typer.secho("# Interactive selection", fg=get_text_color(), dim=True)
    typer.secho("  /model synthesis         ", fg=get_system_color(), nl=False)
    typer.secho("# Interactive selection", fg=get_text_color(), dim=True)


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
    
    # Validate model exists
    if not validate_model_exists(model_name):
        typer.secho(f"Model not found: {model_name}", fg="red")
        typer.secho("Use '/model' to see available models", fg=get_text_color())
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
        "detection": "ollama/llama3",
        "compression": "ollama/llama3",
        "synthesis": "gpt-3.5-turbo"
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


def show_and_select_model_for_context(context: str):
    """Show available models and allow selection by number or name."""
    descriptions = {
        "chat": "chat",
        "detection": "topic detection",
        "compression": "compression",
        "synthesis": "web synthesis"
    }
    
    typer.secho(f"\n🤖 Select {descriptions[context]} model:", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    # Get all available models
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
                models.append((model_name, provider_name))
    
    # Display models with numbers
    for idx, (model_name, provider_name) in enumerate(models, 1):
        typer.secho(f"  {idx:2d}. ", fg="yellow", nl=False)
        typer.secho(f"{model_name:30s} ", fg="cyan", nl=False)
        typer.secho(f"({provider_name})", fg="white", dim=True)
    
    typer.secho("\n💡 Enter model number or name (or 'cancel' to abort):", fg=get_text_color())
    
    # Get user input
    user_input = typer.prompt("", prompt_suffix="> ")
    
    if user_input.lower() == "cancel":
        typer.secho("Cancelled.", fg="yellow")
        return
    
    # Check if input is a number
    try:
        model_index = int(user_input)
        if 1 <= model_index <= len(models):
            selected_model, _ = models[model_index - 1]
            set_model_for_context(context, selected_model)
        else:
            typer.secho(f"Invalid number. Please choose between 1 and {len(models)}.", fg="red")
    except ValueError:
        # Try to match by name
        matched = False
        for model_name, _ in models:
            if user_input.lower() in model_name.lower() or model_name.lower() == user_input.lower():
                set_model_for_context(context, model_name)
                matched = True
                break
        
        if not matched:
            typer.secho(f"Model '{user_input}' not found. Use exact model name or number.", fg="red")