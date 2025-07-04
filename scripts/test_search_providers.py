#!/usr/bin/env python3
"""Test script for web search providers."""

import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from episodic.config import config
from episodic.web_search import (
    DuckDuckGoProvider, SearxProvider, GoogleProvider, BingProvider
)
import typer


async def test_provider(provider_name: str, provider):
    """Test a single provider."""
    typer.secho(f"\n{'='*60}", fg="cyan")
    typer.secho(f"Testing {provider_name}", fg="cyan", bold=True)
    typer.secho(f"{'='*60}", fg="cyan")
    
    if not provider.is_available():
        typer.secho(f"❌ {provider_name} is not available/configured", fg="red")
        return
    
    query = "Python programming language"
    typer.secho(f"Searching for: '{query}'", fg="yellow")
    
    try:
        results = await provider.search(query, num_results=3)
        
        if results:
            typer.secho(f"✅ Found {len(results)} results:", fg="green")
            for i, result in enumerate(results, 1):
                typer.secho(f"\n[{i}] {result.title}", fg="blue", bold=True)
                typer.secho(f"    URL: {result.url}", fg="cyan")
                typer.secho(f"    Snippet: {result.snippet[:100]}...", fg="white")
        else:
            typer.secho("❌ No results returned", fg="red")
            
    except Exception as e:
        typer.secho(f"❌ Error: {e}", fg="red")


async def main():
    """Test all search providers."""
    typer.secho("🔍 Web Search Provider Test Suite", fg="cyan", bold=True)
    
    # Test DuckDuckGo (should always work)
    await test_provider("DuckDuckGo", DuckDuckGoProvider())
    
    # Test Searx
    await test_provider("Searx", SearxProvider())
    
    # Test Google (requires API key)
    # Set these for testing:
    # export GOOGLE_API_KEY=your_key
    # export GOOGLE_SEARCH_ENGINE_ID=your_id
    await test_provider("Google", GoogleProvider())
    
    # Test Bing (requires API key)
    # Set this for testing:
    # export BING_API_KEY=your_key
    await test_provider("Bing", BingProvider())
    
    typer.secho("\n✅ Test complete!", fg="green", bold=True)


if __name__ == "__main__":
    asyncio.run(main())