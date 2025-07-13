"""
Enhanced async version of web search with proper async/await support.

This module provides truly asynchronous web search functionality to avoid
blocking the event loop during HTTP requests.
"""

import asyncio
import time
from typing import List, Optional
from datetime import datetime

import typer
from episodic.config import config
from episodic.web_search import (
    SearchResult, WebSearchProvider, DuckDuckGoProvider, 
    SearxProvider, GoogleProvider, BingProvider,
    SearchCache, RateLimiter
)


class AsyncWebSearchManager:
    """Manages web search with async support, caching and rate limiting."""
    
    def __init__(self):
        self.providers = self._get_configured_providers()
        self.cache = SearchCache()
        self.rate_limiter = RateLimiter(
            max_per_hour=config.get('web_search_rate_limit', 60)
        )
        self._working_provider_cache = None
        self._working_provider_timestamp = None
    
    def _get_configured_providers(self) -> List[WebSearchProvider]:
        """Get the configured search providers in order of preference."""
        # Check for new providers list first
        providers_list = config.get('web_search_providers')
        
        if not providers_list:
            # Fall back to single provider for backward compatibility
            provider_name = config.get('web_search_provider', 'duckduckgo')
            providers_list = [provider_name]
        elif isinstance(providers_list, str):
            # Handle comma-separated string
            providers_list = [p.strip() for p in providers_list.split(',')]
        
        # Initialize all available provider classes
        provider_classes = {
            'duckduckgo': DuckDuckGoProvider,
            'searx': SearxProvider,
            'google': GoogleProvider,
            'bing': BingProvider
        }
        
        # Create provider instances
        providers = []
        for provider_name in providers_list:
            provider_name = provider_name.lower()
            provider_class = provider_classes.get(provider_name)
            
            if not provider_class:
                continue
            
            try:
                provider = provider_class()
                if provider:
                    providers.append(provider)
            except Exception:
                pass
        
        # Always ensure DuckDuckGo is available as last resort
        if not any(isinstance(p, DuckDuckGoProvider) for p in providers):
            providers.append(DuckDuckGoProvider())
        
        return providers
    
    def _get_working_provider(self) -> Optional[WebSearchProvider]:
        """Get the cached working provider if still valid."""
        if not config.get('web_search_fallback_enabled', True):
            return None
            
        if self._working_provider_cache and self._working_provider_timestamp:
            cache_minutes = config.get('web_search_fallback_cache_minutes', 5)
            age = datetime.now() - self._working_provider_timestamp
            if age.total_seconds() < cache_minutes * 60:
                return self._working_provider_cache
        
        return None
    
    async def search_async(self, query: str, num_results: int = None, 
                          use_cache: bool = True) -> List[SearchResult]:
        """
        Perform a web search asynchronously with caching, rate limiting, and provider fallback.
        
        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use cached results
            
        Returns:
            List of search results
        """
        if num_results is None:
            num_results = config.get('web_search_max_results', 5)
        
        # Check cache first
        if use_cache:
            cache_duration = config.get('web_search_cache_duration', 3600)
            cached = self.cache.get(query, cache_duration)
            if cached:
                if config.get('debug'):
                    typer.secho(f"Using cached results for: {query}", fg="cyan")
                return cached[:num_results]
        
        # Check rate limit
        if not self.rate_limiter.can_search():
            remaining = self.rate_limiter.remaining()
            typer.secho(
                f"⚠️  Rate limit reached. {remaining} searches remaining this hour.",
                fg="yellow"
            )
            return []
        
        # Get providers to try
        working_provider = self._get_working_provider()
        if working_provider and self.providers and working_provider != self.providers[0]:
            self._working_provider_cache = None
            self._working_provider_timestamp = None
            working_provider = None
        
        providers_to_try = [working_provider] if working_provider else self.providers
        
        # Try each provider in order
        for i, provider in enumerate(providers_to_try):
            if provider is None:
                continue
                
            provider_name = provider.__class__.__name__.replace('Provider', '')
            
            # Skip unavailable providers
            if not provider.is_available():
                if i == 0 or config.get('debug'):
                    self._log_provider_skip(provider_name)
                continue
            
            try:
                # Always show which provider we're using
                typer.secho(f"🔍 Searching with {provider_name}...", fg="cyan")
                
                # Perform async search
                results = await provider.search(query, num_results)
                
                # If we got results, cache and return
                if results:
                    self.rate_limiter.record_search()
                    self.cache.set(query, results)
                    
                    # Cache this working provider
                    if config.get('web_search_fallback_enabled', True):
                        self._working_provider_cache = provider
                        self._working_provider_timestamp = datetime.now()
                    
                    if i > 0:  # We used a fallback
                        typer.secho(f"✅ {provider_name} search successful", fg="green")
                    
                    return results
                
                # Empty results might be valid, but try next provider
                if config.get('debug'):
                    typer.secho(f"{provider_name} returned no results", fg="yellow")
                    
            except Exception as e:
                # Log errors for primary provider or in debug mode
                if i == 0 or config.get('debug'):
                    typer.secho(
                        f"⚠️  {provider_name} failed: {str(e)[:100]}",
                        fg="yellow"
                    )
                
                # Continue to next provider
                if i < len(providers_to_try) - 1:
                    if config.get('web_search_fallback_enabled', True):
                        typer.secho(
                            f"    Trying next provider...",
                            fg="yellow"
                        )
                    continue
        
        # All providers failed
        typer.secho("❌ All search providers failed", fg="red")
        return []
    
    def search(self, query: str, num_results: int = None, 
               use_cache: bool = True) -> List[SearchResult]:
        """
        Synchronous wrapper for async search.
        
        Creates a new event loop if needed to run the async search.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're here, we're already in an async context
            # We can't use asyncio.run() here, so we need to schedule the coroutine
            future = asyncio.create_task(self.search_async(query, num_results, use_cache))
            # This is a bit tricky - we're in an async context but need to return sync
            # In practice, this should be called from async code
            raise RuntimeError("Cannot call synchronous search from async context. Use search_async instead.")
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(self.search_async(query, num_results, use_cache))
    
    def _log_provider_skip(self, provider_name: str):
        """Log why a provider was skipped."""
        if provider_name == 'Google':
            typer.secho(
                f"⚠️  Skipping Google (requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID)",
                fg="yellow"
            )
        elif provider_name == 'Bing':
            typer.secho(
                f"⚠️  Skipping Bing (requires BING_API_KEY)",
                fg="yellow"
            )
        else:
            typer.secho(
                f"⚠️  Skipping {provider_name} (not configured)",
                fg="yellow"
            )
    
    def clear_cache(self):
        """Clear the search cache."""
        self.cache.clear()
    
    def get_stats(self) -> dict:
        """Get search statistics."""
        provider_names = [p.__class__.__name__.replace('Provider', '') 
                         for p in self.providers if p]
        
        current_provider = None
        if self._working_provider_cache:
            current_provider = self._working_provider_cache.__class__.__name__.replace('Provider', '')
        
        return {
            'providers': provider_names,
            'current_provider': current_provider,
            'cache': self.cache.stats(),
            'rate_limit_remaining': self.rate_limiter.remaining(),
            'rate_limit_max': self.rate_limiter.max_per_hour
        }


# Global instance
_async_web_search_manager: Optional[AsyncWebSearchManager] = None


def get_async_web_search_manager() -> AsyncWebSearchManager:
    """Get or create the global async web search manager."""
    global _async_web_search_manager
    if _async_web_search_manager is None:
        _async_web_search_manager = AsyncWebSearchManager()
    return _async_web_search_manager