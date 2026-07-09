"""WebSearchManager helper methods: query heuristics, quality controls,
provider selection.

Mixin split out of web_search.py; WebSearchManager inherits it, so these run on
the instance (self._last_search_diagnostics, self._providers, ... resolve via
inheritance).
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import typer

from episodic.config import config
from episodic.debug_system import debug_enabled
from episodic.configuration import (
    get_error_color, get_warning_color, get_success_color, get_info_color,
)
from episodic.web_search_providers import (
    SearchResult, WebSearchProvider,
    DuckDuckGoProvider, SearxProvider, GoogleProvider, BingProvider, BraveProvider,
)


class _WebSearchHelpersMixin:
    """Query heuristics, result quality controls, and provider selection."""

    @staticmethod
    def _is_time_sensitive_query(query: str) -> bool:
        """Detect queries that should prefer fresh event-oriented results."""
        q = (query or "").lower()
        if not q:
            return False

        time_markers = (
            "this weekend", "weekend", "today", "tonight", "tomorrow",
            "this week", "latest", "current", "happening now", "upcoming",
        )
        activity_markers = (
            "things to do", "what to do", "events", "activities",
            "concerts", "shows", "festival",
        )
        has_time = any(marker in q for marker in time_markers)
        has_activity = any(marker in q for marker in activity_markers)
        return has_time or has_activity

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract normalized hostname from URL."""
        try:
            return (urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _contains_any(text: str, needles: List[str]) -> bool:
        t = (text or "").lower()
        return any(n in t for n in needles)

    def _apply_result_quality_controls(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Filter and rerank results for better Muse grounding quality."""
        if not results:
            return []

        # Apply globally excluded domains first.
        excluded = config.get("web_search_excluded_domains", []) or []
        excluded = [d.lower() for d in excluded if isinstance(d, str) and d]
        filtered = [
            r for r in results
            if not any(d in self._extract_domain(r.url) for d in excluded)
        ]

        if not filtered:
            return []

        if not self._is_time_sensitive_query(query):
            return filtered

        # Exclude known low-signal social/discovery pages for event/time-sensitive queries.
        low_signal = config.get("web_search_time_sensitive_excluded_domains", []) or []
        low_signal = [d.lower() for d in low_signal if isinstance(d, str) and d]
        event_filtered = [
            r for r in filtered
            if not any(d in self._extract_domain(r.url) for d in low_signal)
        ]
        if event_filtered:
            filtered = event_filtered

        # Rerank toward event directories and explicitly time-sensitive pages.
        url_terms = [
            "eventbrite.com", "allevents.", "funcheap.com",
            "/events", "this-weekend", "weekend", "calendar",
        ]
        text_terms = [
            "this weekend", "today", "tonight", "tomorrow",
            "events", "tickets", "festival", "concert",
        ]

        def score(result: SearchResult) -> int:
            url = (result.url or "").lower()
            title = (result.title or "").lower()
            snippet = (result.snippet or "").lower()
            score_val = 0
            if self._contains_any(url, url_terms):
                score_val += 3
            if self._contains_any(title, text_terms):
                score_val += 2
            if self._contains_any(snippet, text_terms):
                score_val += 1
            return score_val

        return sorted(filtered, key=score, reverse=True)

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
        time_sensitive = self._is_time_sensitive_query(query)

        self._last_search_diagnostics = {
            "query": query,
            "providers_attempted": [],
            "success_provider": None,
        }

        # Check cache first
        bypass_cache = (
            time_sensitive
            and config.get("web_search_bypass_cache_for_time_sensitive", True)
        )
        if use_cache and not bypass_cache:
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
                results = self._apply_result_quality_controls(query, results)
                if not results:
                    attempt["status"] = "filtered_empty"
                    attempt["reason"] = "all results filtered by quality controls"
                    self._last_search_diagnostics["providers_attempted"].append(attempt)
                    continue
                attempt["status"] = "ok"
                attempt["result_count"] = len(results)
                self._last_search_diagnostics["providers_attempted"].append(attempt)
                self._last_search_diagnostics["success_provider"] = provider_name
                # Success - cache and return
                self.rate_limiter.record_search()
                if not bypass_cache:
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

