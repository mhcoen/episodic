"""
Web provider management command.

This module provides a command for managing web search providers used by muse mode.
"""

import typer
from typing import Optional
from episodic.config import config
from episodic.configuration import get_heading_color, get_system_color, get_text_color


def web_command(
    subcommand: Optional[str] = None,
    provider_name: Optional[str] = None
):
    """
    Manage web search providers for muse mode.
    
    Usage:
        /web                        # Show current web search provider
        /web list                   # List all available providers
        /web provider               # Show current provider details
        /web provider <name>        # Set web search provider
    """
    # No arguments - show current provider
    if not subcommand:
        show_current_provider()
        return
    
    subcommand = subcommand.lower()
    
    if subcommand == "list":
        list_providers()
    elif subcommand == "provider":
        if provider_name:
            set_provider(provider_name)
        else:
            show_provider_details()
    elif subcommand == "reset" or subcommand == "default":
        reset_to_defaults()
    else:
        typer.secho(f"Unknown subcommand: {subcommand}", fg="red")
        typer.secho("Valid subcommands: list, provider, reset", fg=get_text_color())


def show_current_provider():
    """Show the current web search provider and available commands."""
    current = config.get("web_search_provider", "duckduckgo")
    
    typer.secho("\n🌐 Muse Mode Web Search Provider", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    typer.secho("Current provider: ", fg=get_text_color(), nl=False)
    typer.secho(current.title(), fg=get_system_color(), bold=True)
    
    # Show provider list if multiple providers configured
    providers_list = config.get("web_search_providers")
    if providers_list and isinstance(providers_list, list) and len(providers_list) > 1:
        typer.secho("Fallback order: ", fg=get_text_color(), nl=False)
        typer.secho(" → ".join(providers_list), fg=get_system_color())
    
    # Show available providers
    typer.secho("\nAvailable providers:", fg=get_heading_color())
    providers = [
        ("duckduckgo", "DuckDuckGo", "Free, no API key required"),
        ("google", "Google", "Requires API key and CSE ID"),
        ("bing", "Bing", "Requires API key"),
        ("searx", "Searx", "Requires instance URL"),
    ]
    
    for key, name, description in providers:
        # Show current provider
        if key == current:
            typer.secho("  ► ", fg="green", nl=False, bold=True)
        else:
            typer.secho("    ", nl=False)
        
        # Provider name
        typer.secho(f"{name:<15} ", fg=get_system_color(), bold=True, nl=False)
        
        # Description
        typer.secho(description, fg=get_text_color())
    
    typer.secho("\nCommands:", fg=get_heading_color())
    commands = [
        ("/web list", "Show detailed provider information"),
        ("/web provider", "Show current provider details"),
        ("/web provider <name>", "Set primary web search provider"),
        ("/web reset", "Reset to default providers"),
        ("/set web.providers", "Set fallback order (comma-separated)"),
    ]
    
    for cmd, desc in commands:
        typer.secho(f"  {cmd:<28}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Show example for setting provider order
    typer.secho("\nExample:", fg=get_heading_color())
    typer.secho("  /set web.providers google,bing,duckduckgo", fg=get_system_color(), dim=True)
    typer.secho("  Sets Google as primary, falls back to Bing, then DuckDuckGo", fg=get_text_color(), dim=True)


def list_providers():
    """List all available web search providers."""
    typer.secho("\n🌐 Available Web Search Providers:", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    providers = [
        ("duckduckgo", "DuckDuckGo", "Free, no API key required"),
        ("google", "Google", "Requires API key and CSE ID"),
        ("bing", "Bing", "Requires API key"),
        ("searx", "Searx", "Requires instance URL"),
    ]
    
    current = config.get("web_search_provider", "duckduckgo").lower()
    
    for key, name, description in providers:
        # Show current provider
        if key == current:
            typer.secho("  ► ", fg="green", nl=False, bold=True)
        else:
            typer.secho("    ", nl=False)
        
        # Provider name
        typer.secho(f"{name:<15} ", fg=get_system_color(), bold=True, nl=False)
        
        # Description
        typer.secho(description, fg=get_text_color())


def show_provider_details():
    """Show detailed information about the current provider."""
    current = config.get("web_search_provider", "duckduckgo").lower()
    
    typer.secho(f"\n🌐 Web Search Provider: ", fg=get_heading_color(), bold=True, nl=False)
    typer.secho(current.title(), fg=get_system_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    if current == "duckduckgo":
        typer.secho("Type: ", fg=get_text_color(), nl=False)
        typer.secho("Free, Anonymous", fg="green")
        typer.secho("API Key: ", fg=get_text_color(), nl=False)
        typer.secho("Not required", fg=get_system_color())
        typer.secho("Rate Limit: ", fg=get_text_color(), nl=False)
        typer.secho("No official limit", fg=get_system_color())
        typer.secho("Features: ", fg=get_text_color(), nl=False)
        typer.secho("General web search, instant answers", fg=get_system_color())
        
    elif current == "google":
        typer.secho("Type: ", fg=get_text_color(), nl=False)
        typer.secho("Requires API key", fg="yellow")
        typer.secho("API Key: ", fg=get_text_color(), nl=False)
        api_key = config.get("google_api_key") or config.get("GOOGLE_API_KEY")
        if api_key:
            typer.secho("Configured ✓", fg="green")
        else:
            typer.secho("Not configured ✗", fg="red")
        typer.secho("CSE ID: ", fg=get_text_color(), nl=False)
        cse_id = config.get("google_search_engine_id") or config.get("GOOGLE_SEARCH_ENGINE_ID")
        if cse_id:
            typer.secho("Configured ✓", fg="green")
        else:
            typer.secho("Not configured ✗", fg="red")
        typer.secho("Rate Limit: ", fg=get_text_color(), nl=False)
        typer.secho("100 queries/day (free tier)", fg=get_system_color())
        
    elif current == "bing":
        typer.secho("Type: ", fg=get_text_color(), nl=False)
        typer.secho("Requires API key", fg="yellow")
        typer.secho("API Key: ", fg=get_text_color(), nl=False)
        api_key = config.get("bing_search_api_key")
        if api_key:
            typer.secho("Configured ✓", fg="green")
        else:
            typer.secho("Not configured ✗", fg="red")
        typer.secho("Rate Limit: ", fg=get_text_color(), nl=False)
        typer.secho("1000 queries/month (free tier)", fg=get_system_color())
        
    elif current == "searx":
        typer.secho("Type: ", fg=get_text_color(), nl=False)
        typer.secho("Self-hosted or public instance", fg="cyan")
        typer.secho("Instance URL: ", fg=get_text_color(), nl=False)
        instance_url = config.get("searx_instance_url")
        if instance_url:
            typer.secho(instance_url, fg=get_system_color())
        else:
            typer.secho("Not configured ✗", fg="red")


def set_provider(provider_name: str):
    """Set the web search provider."""
    provider_name = provider_name.lower()
    
    valid_providers = ["duckduckgo", "google", "bing", "searx"]
    
    if provider_name not in valid_providers:
        typer.secho(f"Unknown provider: {provider_name}", fg="red")
        typer.secho(f"Valid providers: {', '.join(valid_providers)}", fg=get_text_color())
        return
    
    
    # Check if provider requires configuration
    if provider_name == "google":
        api_key = config.get("google_api_key") or config.get("GOOGLE_API_KEY")
        cse_id = config.get("google_search_engine_id") or config.get("GOOGLE_SEARCH_ENGINE_ID")
        if not api_key or not cse_id:
            typer.secho("⚠️  Google search requires configuration:", fg="yellow")
            if not api_key:
                typer.secho("  - Set GOOGLE_API_KEY environment variable", fg=get_text_color())
            if not cse_id:
                typer.secho("  - Set GOOGLE_SEARCH_ENGINE_ID environment variable", fg=get_text_color())
            typer.secho("\nGet your API key from: https://developers.google.com/custom-search/v1/overview", fg="cyan")
            return
    
    elif provider_name == "bing":
        api_key = config.get("bing_search_api_key")
        if not api_key:
            typer.secho("⚠️  Bing search requires an API key:", fg="yellow")
            typer.secho("  - Set BING_SEARCH_API_KEY environment variable", fg=get_text_color())
            typer.secho("\nGet your API key from: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api", fg="cyan")
            return
    
    elif provider_name == "searx":
        instance_url = config.get("searx_instance_url")
        if not instance_url:
            typer.secho("⚠️  Searx requires an instance URL:", fg="yellow")
            typer.secho("  - Set SEARX_INSTANCE_URL environment variable", fg=get_text_color())
            typer.secho("\nPublic instances: https://searx.space/", fg="cyan")
            return
    
    # Set the provider
    config.set("web_search_provider", provider_name)
    
    # Also update the providers list to include this provider
    current_providers = config.get("web_search_providers", ["duckduckgo"])
    if isinstance(current_providers, str):
        current_providers = [p.strip() for p in current_providers.split(',')]
    
    # Add the new provider to the front of the list if not already there
    if provider_name not in current_providers:
        current_providers.insert(0, provider_name)
    else:
        # Move it to the front
        current_providers.remove(provider_name)
        current_providers.insert(0, provider_name)
    
    config.set("web_search_providers", current_providers)
    
    # Reset the global web search manager to pick up the new configuration
    import episodic.web_search
    episodic.web_search._web_search_manager = None
    
    typer.secho(f"✓ Web search provider set to: {provider_name.title()}", fg="green")
    typer.secho(f"  Provider order: {' → '.join(current_providers)}", fg=get_text_color(), dim=True)
    
    # Show any additional info
    if provider_name == "duckduckgo":
        typer.secho("  No API key required - ready to use!", fg=get_text_color(), dim=True)


def reset_to_defaults():
    """Reset web search configuration to defaults."""
    from episodic.config_defaults import DEFAULT_CONFIG
    
    # Reset to default values
    default_provider = DEFAULT_CONFIG.get("web_search_provider", "duckduckgo")
    default_providers = DEFAULT_CONFIG.get("web_search_providers", ["duckduckgo"])
    
    config.set("web_search_provider", default_provider)
    config.set("web_search_providers", default_providers)
    
    typer.secho("✓ Reset to default web search configuration", fg="green")
    typer.secho(f"  Provider: {default_provider}", fg=get_text_color(), dim=True)
    typer.secho(f"  Provider order: {' → '.join(default_providers)}", fg=get_text_color(), dim=True)