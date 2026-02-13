"""
Web search functionality for Episodic RAG.

This module provides configurable web search capabilities to enhance
the RAG system with current information from the internet.

Search providers are implemented in the web_search_providers package.
This module re-exports them for backward compatibility.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import typer
from episodic.config import config
from episodic.debug_system import debug_enabled
from episodic.configuration import (
    get_error_color, get_warning_color, get_success_color,
    get_info_color
)

# Re-export provider classes and base types for backward compatibility.
# External callers can continue to import from episodic.web_search.
from episodic.web_search_providers import (  # noqa: F401
    SearchResult,
    WebSearchProvider,
    DuckDuckGoProvider,
    SearxProvider,
    GoogleProvider,
    BingProvider,
    BraveProvider,
)


class SearchCache:
    """Simple in-memory cache for search results."""

    def __init__(self):
        self._cache: Dict[str, tuple[List[SearchResult], datetime]] = {}

    def get(self, query: str, max_age_seconds: int = 3600) -> Optional[List[SearchResult]]:
        """Get cached results if they exist and aren't too old."""
        if query not in self._cache:
            return None

        results, timestamp = self._cache[query]
        age = datetime.now() - timestamp

        if age.total_seconds() > max_age_seconds:
            del self._cache[query]
            return None

        return results

    def set(self, query: str, results: List[SearchResult]):
        """Cache search results."""
        self._cache[query] = (results, datetime.now())

    def clear(self):
        """Clear all cached results."""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'entries': len(self._cache),
            'queries': list(self._cache.keys())
        }


class RateLimiter:
    """Simple rate limiter for web searches."""

    def __init__(self, max_per_hour: int = 60):
        self.max_per_hour = max_per_hour
        self.searches: List[datetime] = []

    def can_search(self) -> bool:
        """Check if we can perform another search."""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Remove old searches
        self.searches = [s for s in self.searches if s > cutoff]

        return len(self.searches) < self.max_per_hour

    def record_search(self):
        """Record that a search was performed."""
        self.searches.append(datetime.now())

    def remaining(self) -> int:
        """Get number of searches remaining in current hour."""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        self.searches = [s for s in self.searches if s > cutoff]
        return max(0, self.max_per_hour - len(self.searches))


class WebSearchManager:
    """Manages web search with caching and rate limiting."""

    def __init__(self):
        self.providers = self._get_configured_providers()
        self.cache = SearchCache()
        self.rate_limiter = RateLimiter(
            max_per_hour=config.get('web_search_rate_limit', 60)
        )
        self._working_provider_cache = None
        self._working_provider_timestamp = None
        self._last_search_diagnostics: Dict[str, Any] = {}

    def get_last_search_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent search attempt."""
        return dict(self._last_search_diagnostics)

    def _get_configured_providers(self) -> List[WebSearchProvider]:
        """Get the configured search providers in order of preference."""
        # Check for new providers list first
        providers_list = config.get('web_search_providers')

        if config.get('debug'):
            typer.secho(f"[DEBUG] web_search_providers from config: {providers_list}", fg=get_warning_color())

        if not providers_list:
            # Default to duckduckgo
            providers_list = ['duckduckgo']
        elif isinstance(providers_list, str):
            # Handle comma-separated string
            providers_list = [p.strip() for p in providers_list.split(',')]

        if config.get('debug'):
            typer.secho(f"[DEBUG] Final providers list: {providers_list}", fg=get_warning_color())

        # Initialize all available provider classes
        provider_classes = {
            'duckduckgo': DuckDuckGoProvider,
            'searx': SearxProvider,
            'google': GoogleProvider,
            'bing': BingProvider,
            'brave': BraveProvider
        }

        # Create provider instances
        providers = []
        for provider_name in providers_list:
            provider_name = provider_name.lower()
            provider_class = provider_classes.get(provider_name)

            if not provider_class:
                if config.get('debug'):
                    typer.secho(
                        f"Unknown provider '{provider_name}', skipping.",
                        fg="yellow"
                    )
                continue

            try:
                provider = provider_class()
                if provider is None:
                    if config.get('debug'):
                        typer.secho(f"\u26a0\ufe0f  {provider_name} class returned None instance", fg=get_warning_color())
                    continue
                providers.append(provider)
                if config.get('debug'):
                    typer.secho(f"\u2713 Successfully created {provider_name} provider", fg=get_info_color())
            except Exception as e:
                if config.get('debug'):
                    typer.secho(f"\u26a0\ufe0f  Failed to create {provider_name} provider: {e}", fg=get_error_color())

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

    def _is_quota_or_auth_error(self, response_status: int) -> bool:
        """Check if error is due to quota/auth issues."""
        return response_status in [401, 403, 429]

    async def _search_provider_async(self, query: str, num_results: int,
                                   provider: WebSearchProvider, provider_name: str) -> Optional[List[SearchResult]]:
        """Execute async search for a single provider."""
        verbose = config.get('debug') or debug_enabled('web') or debug_enabled('muse')
        try:
            # Show provider attempts only in debug mode to reduce UI noise.
            if verbose:
                typer.secho(f"\U0001f50d Searching with {provider_name}...", fg=get_info_color())

            results = await provider.search(query, num_results)

            if results:
                return results

            # Empty results might be valid
            if verbose:
                typer.secho(f"{provider_name} returned no results", fg=get_warning_color())

        except Exception as e:
            # Show errors for debugging
            if verbose:
                error_msg = str(e)
                if provider_name == 'Google' and 'API_KEY_SERVICE_BLOCKED' in error_msg:
                    typer.secho(
                        f"\u26a0\ufe0f  Google Search API is not enabled for your project",
                        fg="yellow"
                    )
                else:
                    typer.secho(f"\u26a0\ufe0f  {provider_name} search failed: {error_msg}", fg=get_error_color())

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
        verbose = config.get('debug') or debug_enabled('web') or debug_enabled('muse')

        self._last_search_diagnostics = {
            "query": query,
            "providers_attempted": [],
            "success_provider": None,
        }

        # Check cache first
        if use_cache:
            cache_duration = config.get('web_search_cache_duration', 3600)
            cached = self.cache.get(query, cache_duration)
            if cached:
                if verbose:
                    typer.secho(f"Using cached results for: {query}", fg=get_info_color())
                return cached[:num_results]

        # Check rate limit
        if not self.rate_limiter.can_search():
            remaining = self.rate_limiter.remaining()
            typer.secho(
                f"\u26a0\ufe0f  Rate limit reached. {remaining} searches remaining this hour.",
                fg=get_warning_color()
            )
            return []

        # Check for cached working provider
        working_provider = self._get_working_provider()

        # If we have a cached provider, but it's not the first in our list,
        # clear the cache to respect the user's configuration
        if working_provider and self.providers and working_provider != self.providers[0]:
            self._working_provider_cache = None
            self._working_provider_timestamp = None
            working_provider = None

        if working_provider:
            providers_to_try = [working_provider]
        else:
            providers_to_try = self.providers

        # Try each provider in order
        for i, provider in enumerate(providers_to_try):
            if provider is None or provider.__class__ is None:
                if verbose:
                    typer.secho(f"\u26a0\ufe0f  Provider at index {i} is None, skipping", fg=get_warning_color())
                continue
            provider_name = provider.__class__.__name__.replace('Provider', '')
            attempt = {"provider": provider_name, "status": "unknown"}

            # Skip providers that aren't available (missing credentials)
            if not provider.is_available():
                attempt["status"] = "skipped"
                attempt["reason"] = "provider unavailable (missing dependencies or credentials)"
                self._last_search_diagnostics["providers_attempted"].append(attempt)
                # Always show when skipping a provider that was explicitly configured
                if i == 0 or verbose:
                    # More specific message for different providers
                    if provider_name == 'Google':
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping Google (requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID)",
                            fg="yellow"
                        )
                    elif provider_name == 'Bing':
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping Bing (requires BING_API_KEY)",
                            fg="yellow"
                        )
                    elif provider_name == 'DuckDuckGo':
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping DuckDuckGo (requires: pip install ddgs)",
                            fg="yellow"
                        )
                    else:
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping {provider_name} (not configured or missing dependencies)",
                            fg="yellow"
                        )
                continue

            # Try async search
            try:
                results = await self._search_provider_async(query, num_results, provider, provider_name)
            except Exception as e:
                attempt["status"] = "error"
                attempt["reason"] = str(e)
                self._last_search_diagnostics["providers_attempted"].append(attempt)
                continue

            if results:
                attempt["status"] = "ok"
                attempt["result_count"] = len(results)
                self._last_search_diagnostics["providers_attempted"].append(attempt)
                self._last_search_diagnostics["success_provider"] = provider_name
                # Success - cache and return
                self.rate_limiter.record_search()
                self.cache.set(query, results)

                # Cache this working provider
                if config.get('web_search_fallback_enabled', True):
                    self._working_provider_cache = provider
                    self._working_provider_timestamp = datetime.now()

                if i > 0:  # We used a fallback
                    typer.secho(f"\u2705 {provider_name} search successful", fg=get_success_color())

                return results

            attempt["status"] = "empty"
            attempt["reason"] = "provider returned no results"
            self._last_search_diagnostics["providers_attempted"].append(attempt)

        # No provider succeeded (details are in diagnostics)
        return []

    def search(self, query: str, num_results: int = None,
               use_cache: bool = True) -> List[SearchResult]:
        """
        Perform a web search with caching, rate limiting, and provider fallback.

        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use cached results

        Returns:
            List of search results
        """
        if num_results is None:
            num_results = config.get('web_search_max_results', 5)

        self._last_search_diagnostics = {
            "query": query,
            "providers_attempted": [],
            "success_provider": None,
        }

        # Check cache first
        if use_cache:
            cache_duration = config.get('web_search_cache_duration', 3600)
            cached = self.cache.get(query, cache_duration)
            if cached:
                if config.get('debug'):
                    typer.secho(f"Using cached results for: {query}", fg=get_info_color())
                return cached[:num_results]

        # Check rate limit
        if not self.rate_limiter.can_search():
            remaining = self.rate_limiter.remaining()
            typer.secho(
                f"\u26a0\ufe0f  Rate limit reached. {remaining} searches remaining this hour.",
                fg=get_warning_color()
            )
            return []

        # Check for cached working provider
        working_provider = self._get_working_provider()

        # If we have a cached provider, but it's not the first in our list,
        # clear the cache to respect the user's configuration
        if working_provider and self.providers and working_provider != self.providers[0]:
            self._working_provider_cache = None
            self._working_provider_timestamp = None
            working_provider = None

        if working_provider:
            providers_to_try = [working_provider]
        else:
            providers_to_try = self.providers

        # Try each provider in order
        for i, provider in enumerate(providers_to_try):
            if provider is None or provider.__class__ is None:
                if config.get('debug'):
                    typer.secho(f"\u26a0\ufe0f  Provider at index {i} is None, skipping", fg=get_warning_color())
                continue
            provider_name = provider.__class__.__name__.replace('Provider', '')
            attempt = {"provider": provider_name, "status": "unknown"}

            # Skip providers that aren't available (missing credentials)
            if not provider.is_available():
                attempt["status"] = "skipped"
                attempt["reason"] = "provider unavailable (missing dependencies or credentials)"
                self._last_search_diagnostics["providers_attempted"].append(attempt)
                # Always show when skipping a provider that was explicitly configured
                if i == 0 or config.get('debug'):
                    # More specific message for different providers
                    if provider_name == 'Google':
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping Google (requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID)",
                            fg="yellow"
                        )
                    elif provider_name == 'Bing':
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping Bing (requires BING_API_KEY)",
                            fg="yellow"
                        )
                    elif provider_name == 'DuckDuckGo':
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping DuckDuckGo (requires: pip install ddgs)",
                            fg="yellow"
                        )
                    else:
                        typer.secho(
                            f"\u26a0\ufe0f  Skipping {provider_name} (not configured or missing dependencies)",
                            fg="yellow"
                        )
                continue

            try:
                # Show provider attempts only in debug mode to reduce UI noise.
                if config.get('debug'):
                    typer.secho(f"\U0001f50d Searching with {provider_name}...", fg=get_info_color())

                # Run async search in sync context
                previous_loop = None
                had_previous_loop = False
                try:
                    previous_loop = asyncio.get_event_loop()
                    had_previous_loop = True
                except RuntimeError:
                    pass

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        provider.search(query, num_results)
                    )
                finally:
                    loop.close()
                    if had_previous_loop and previous_loop and not previous_loop.is_closed():
                        asyncio.set_event_loop(previous_loop)
                    else:
                        asyncio.set_event_loop(None)

                # If we got results, cache the provider and return
                if results:
                    attempt["status"] = "ok"
                    attempt["result_count"] = len(results)
                    self._last_search_diagnostics["providers_attempted"].append(attempt)
                    self._last_search_diagnostics["success_provider"] = provider_name
                    self.rate_limiter.record_search()
                    self.cache.set(query, results)

                    # Cache this working provider
                    if config.get('web_search_fallback_enabled', True):
                        self._working_provider_cache = provider
                        self._working_provider_timestamp = datetime.now()

                    if i > 0:  # We used a fallback
                        typer.secho(f"\u2705 {provider_name} search successful", fg=get_success_color())

                    return results

                # Empty results might be valid, but try next provider
                attempt["status"] = "empty"
                attempt["reason"] = "provider returned no results"
                self._last_search_diagnostics["providers_attempted"].append(attempt)
                if config.get('debug'):
                    typer.secho(f"{provider_name} returned no results", fg=get_warning_color())

            except Exception as e:
                attempt["status"] = "error"
                attempt["reason"] = str(e)
                self._last_search_diagnostics["providers_attempted"].append(attempt)
                # Always show errors for the primary provider
                if i == 0 or config.get('debug'):
                    error_msg = str(e)
                    # Check for specific Google API errors
                    if provider_name == 'Google' and 'API_KEY_SERVICE_BLOCKED' in error_msg:
                        typer.secho(
                            f"\u26a0\ufe0f  Google Search API is not enabled for your project",
                            fg="yellow"
                        )
                        typer.secho(
                            f"    Enable it at: https://console.cloud.google.com/apis/library/customsearch.googleapis.com",
                            fg="cyan"
                        )
                    elif 'permission' in error_msg.lower() or '403' in error_msg:
                        typer.secho(
                            f"\u26a0\ufe0f  {provider_name} access denied: {error_msg[:100]}",
                            fg="yellow"
                        )
                    elif 'api' in error_msg.lower() or 'key' in error_msg.lower():
                        typer.secho(
                            f"\u26a0\ufe0f  {provider_name} configuration error: {error_msg[:100]}",
                            fg="yellow"
                        )
                    else:
                        typer.secho(
                            f"\u26a0\ufe0f  {provider_name} failed: {error_msg[:100]}",
                            fg="yellow"
                        )

                # For quota/auth errors, immediately try next provider
                if i < len(providers_to_try) - 1:
                    if config.get('web_search_fallback_enabled', True):
                        pass  # Just continue to next provider
                        typer.secho(
                            f"    Trying next provider...",
                            fg="yellow"
                        )
                    continue

        # All providers failed (details are in diagnostics)
        return []

    def clear_cache(self):
        """Clear the search cache."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        provider_names = [p.__class__.__name__.replace('Provider', '')
                         for p in self.providers if p is not None and p.__class__ is not None]

        current_provider = None
        if self._working_provider_cache and self._working_provider_cache.__class__ is not None:
            current_provider = self._working_provider_cache.__class__.__name__.replace('Provider', '')

        return {
            'providers': provider_names,
            'current_provider': current_provider,
            'cache': self.cache.stats(),
            'rate_limit_remaining': self.rate_limiter.remaining(),
            'rate_limit_max': self.rate_limiter.max_per_hour
        }


# Global instance
_web_search_manager: Optional[WebSearchManager] = None


def get_web_search_manager() -> WebSearchManager:
    """Get or create the global web search manager."""
    global _web_search_manager
    if _web_search_manager is None:
        _web_search_manager = WebSearchManager()
    return _web_search_manager
