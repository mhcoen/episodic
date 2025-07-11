"""
Web search functionality for Episodic RAG.

This module provides configurable web search capabilities to enhance
the RAG system with current information from the internet.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus

import typer
from episodic.config import config


@dataclass
class SearchResult:
    """Represents a single web search result."""
    title: str
    url: str
    snippet: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class WebSearchProvider(ABC):
    """Base class for web search providers."""
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Perform a web search and return results."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available for use."""


class DuckDuckGoProvider(WebSearchProvider):
    """
    Free web search provider using DuckDuckGo.
    No API key required, uses web scraping with rate limiting.
    """
    
    def __init__(self):
        self.last_search_time = 0
        self.min_delay = 1.0  # Minimum seconds between searches
    
    def is_available(self) -> bool:
        """DuckDuckGo is always available."""
        return True
    
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search DuckDuckGo and parse results."""
        # Rate limiting
        elapsed = time.time() - self.last_search_time
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)
        
        try:
            import aiohttp
            from bs4 import BeautifulSoup
        except ImportError:
            typer.secho(
                "⚠️  Web search requires additional dependencies. Install with:\n"
                "    pip install aiohttp beautifulsoup4",
                fg="yellow"
            )
            return []
        
        results = []
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Parse DuckDuckGo results
                    for result_div in soup.find_all('div', class_='result__body')[:num_results]:
                        title_elem = result_div.find('a', class_='result__a')
                        snippet_elem = result_div.find('a', class_='result__snippet')
                        
                        if title_elem and snippet_elem:
                            title = title_elem.get_text(strip=True)
                            url = title_elem.get('href', '')
                            snippet = snippet_elem.get_text(strip=True)
                            
                            if title and url:
                                results.append(SearchResult(
                                    title=title,
                                    url=url,
                                    snippet=snippet
                                ))
                    
                    self.last_search_time = time.time()
                    
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"DuckDuckGo search error: {e}", fg="red")
        
        return results


class SearxProvider(WebSearchProvider):
    """
    Open source metasearch engine provider.
    Requires Searx/SearxNG instance URL, no API key needed.
    """
    
    def __init__(self):
        self.instance_url = config.get('searx_instance_url', 'https://searx.be')
        self.last_search_time = 0
        self.min_delay = 0.5  # Searx is usually self-hosted
    
    def is_available(self) -> bool:
        """Check if Searx instance is configured."""
        return bool(self.instance_url)
    
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Searx/SearxNG API."""
        # Rate limiting
        elapsed = time.time() - self.last_search_time
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)
        
        try:
            import aiohttp
        except ImportError:
            typer.secho(
                "⚠️  Web search requires aiohttp. Install with:\n"
                "    pip install aiohttp",
                fg="yellow"
            )
            return []
        
        results = []
        # Use JSON format for easier parsing
        url = f"{self.instance_url}/search"
        params = {
            'q': query,
            'format': 'json',
            'categories': 'general',
            'engines': 'google,bing,duckduckgo',
            'pageno': 1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        if config.get('debug'):
                            typer.secho(f"Searx returned status {response.status}", fg="yellow")
                        return []
                    
                    data = await response.json()
                    
                    # Parse Searx results
                    for result in data.get('results', [])[:num_results]:
                        title = result.get('title', '')
                        url = result.get('url', '')
                        content = result.get('content', '')
                        
                        if title and url:
                            results.append(SearchResult(
                                title=title,
                                url=url,
                                snippet=content
                            ))
                    
                    self.last_search_time = time.time()
                    
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"Searx search error: {e}", fg="red")
        
        return results


class GoogleProvider(WebSearchProvider):
    """
    Google Custom Search API provider.
    Requires API key and search engine ID.
    """
    
    def __init__(self):
        self.api_key = config.get('google_api_key') or config.get('GOOGLE_API_KEY')
        self.search_engine_id = config.get('google_search_engine_id') or config.get('GOOGLE_SEARCH_ENGINE_ID')
    
    def is_available(self) -> bool:
        """Check if Google Search is configured."""
        return bool(self.api_key and self.search_engine_id)
    
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        if not self.is_available():
            if config.get('debug'):
                typer.secho(
                    "Google Search requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID",
                    fg="yellow"
                )
            return []
        
        try:
            import aiohttp
        except ImportError:
            typer.secho(
                "⚠️  Web search requires aiohttp. Install with:\n"
                "    pip install aiohttp",
                fg="yellow"
            )
            return []
        
        results = []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)  # Google limits to 10 per request
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        if config.get('debug'):
                            error_data = await response.text()
                            typer.secho(f"Google API error: {error_data}", fg="red")
                        return []
                    
                    data = await response.json()
                    
                    # Parse Google results
                    for item in data.get('items', []):
                        title = item.get('title', '')
                        link = item.get('link', '')
                        snippet = item.get('snippet', '')
                        
                        if title and link:
                            results.append(SearchResult(
                                title=title,
                                url=link,
                                snippet=snippet
                            ))
                    
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"Google search error: {e}", fg="red")
        
        return results


class BingProvider(WebSearchProvider):
    """
    Bing Search API provider.
    Requires API key from Azure Cognitive Services.
    """
    
    def __init__(self):
        self.api_key = config.get('bing_api_key') or config.get('BING_API_KEY')
        self.endpoint = config.get('bing_endpoint', 'https://api.bing.microsoft.com/v7.0/search')
    
    def is_available(self) -> bool:
        """Check if Bing Search is configured."""
        return bool(self.api_key)
    
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Bing Search API."""
        if not self.is_available():
            if config.get('debug'):
                typer.secho(
                    "Bing Search requires BING_API_KEY from Azure Cognitive Services",
                    fg="yellow"
                )
            return []
        
        try:
            import aiohttp
        except ImportError:
            typer.secho(
                "⚠️  Web search requires aiohttp. Install with:\n"
                "    pip install aiohttp",
                fg="yellow"
            )
            return []
        
        results = []
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key
        }
        params = {
            'q': query,
            'count': num_results,
            'textDecorations': False,
            'textFormat': 'Raw'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoint, headers=headers, params=params) as response:
                    if response.status != 200:
                        if config.get('debug'):
                            error_data = await response.text()
                            typer.secho(f"Bing API error: {error_data}", fg="red")
                        return []
                    
                    data = await response.json()
                    
                    # Parse Bing results
                    for result in data.get('webPages', {}).get('value', []):
                        name = result.get('name', '')
                        url = result.get('url', '')
                        snippet = result.get('snippet', '')
                        
                        if name and url:
                            results.append(SearchResult(
                                title=name,
                                url=url,
                                snippet=snippet
                            ))
                    
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"Bing search error: {e}", fg="red")
        
        return results


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
                if config.get('debug'):
                    typer.secho(
                        f"Unknown provider '{provider_name}', skipping.",
                        fg="yellow"
                    )
                continue
            
            provider = provider_class()
            providers.append(provider)
        
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
        
        # Check for cached working provider
        working_provider = self._get_working_provider()
        if working_provider:
            providers_to_try = [working_provider]
        else:
            providers_to_try = self.providers
        
        # Try each provider in order
        for i, provider in enumerate(providers_to_try):
            provider_name = provider.__class__.__name__.replace('Provider', '')
            
            # Skip providers that aren't available (missing credentials)
            if not provider.is_available():
                if config.get('debug'):
                    typer.secho(
                        f"Skipping {provider_name} (not configured)",
                        fg="yellow"
                    )
                continue
            
            try:
                if config.get('debug') or (i > 0 and config.get('web_search_fallback_enabled', True)):
                    typer.secho(f"🔍 Searching with {provider_name}...", fg="cyan")
                
                # Run async search in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    provider.search(query, num_results)
                )
                loop.close()
                
                # If we got results, cache the provider and return
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
                if config.get('debug'):
                    typer.secho(f"{provider_name} error: {e}", fg="red")
                
                # For quota/auth errors, immediately try next provider
                if i < len(providers_to_try) - 1:
                    if config.get('web_search_fallback_enabled', True):
                        typer.secho(
                            f"⚠️  {provider_name} failed, trying next provider...",
                            fg="yellow"
                        )
                    continue
        
        # All providers failed
        typer.secho("❌ All search providers failed", fg="red")
        return []
    
    def clear_cache(self):
        """Clear the search cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        provider_names = [p.__class__.__name__.replace('Provider', '') 
                         for p in self.providers]
        
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
_web_search_manager: Optional[WebSearchManager] = None


def get_web_search_manager() -> WebSearchManager:
    """Get or create the global web search manager."""
    global _web_search_manager
    if _web_search_manager is None:
        _web_search_manager = WebSearchManager()
    return _web_search_manager