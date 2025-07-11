#!/usr/bin/env python3
"""
Test script for web search provider fallback functionality.

This script demonstrates the automatic fallback between search providers
when errors occur (e.g., quota exceeded, API errors).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.config import config
from episodic.web_search import get_web_search_manager
import typer


def test_fallback():
    """Test the fallback functionality."""
    
    # Configure multiple providers
    typer.secho("\n🧪 Testing Web Search Provider Fallback", fg="cyan", bold=True)
    typer.secho("=" * 50, fg="cyan")
    
    # Set up providers list
    config.set('web_search_providers', ['google', 'bing', 'duckduckgo'])
    config.set('web_search_fallback_enabled', True)
    config.set('web_search_enabled', True)
    config.set('debug', True)  # Enable debug to see fallback messages
    
    # Get the search manager
    manager = get_web_search_manager()
    
    # Show current configuration
    typer.secho("\n📋 Current Configuration:", fg="yellow")
    stats = manager.get_stats()
    typer.secho(f"Providers: {', '.join(stats['providers'])}", fg="white")
    typer.secho(f"Fallback enabled: {config.get('web_search_fallback_enabled')}", fg="white")
    
    # Test search
    typer.secho("\n🔍 Testing search with fallback...", fg="green")
    query = "Python programming best practices 2024"
    
    results = manager.search(query, num_results=3)
    
    if results:
        typer.secho(f"\n✅ Search successful! Found {len(results)} results", fg="green")
        for i, result in enumerate(results, 1):
            typer.secho(f"\n{i}. {result.title}", fg="cyan", bold=True)
            typer.secho(f"   {result.url}", fg="blue")
            typer.secho(f"   {result.snippet[:100]}...", fg="white")
    else:
        typer.secho("\n❌ All providers failed", fg="red")
    
    # Show which provider was used
    stats = manager.get_stats()
    if stats['current_provider']:
        typer.secho(f"\n✨ Provider used: {stats['current_provider']}", fg="green")
    
    # Test single provider (no fallback)
    typer.secho("\n\n🧪 Testing single provider (no fallback)...", fg="cyan")
    config.set('web_search_providers', ['google'])  # Only Google
    
    # Create new manager to pick up config change
    manager = get_web_search_manager()
    
    results = manager.search("AI trends 2024", num_results=3)
    
    if results:
        typer.secho(f"\n✅ Google search successful!", fg="green")
    else:
        typer.secho(f"\n⚠️  Google search failed (no fallback available)", fg="yellow")


if __name__ == "__main__":
    test_fallback()