"""
Web search functionality for Episodic RAG.

This module provides configurable web search capabilities to enhance
the RAG system with current information from the internet.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus

import typer
from episodic.config import config
from episodic.configuration import get_text_color, get_system_color


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
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available for use."""
        pass


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
        self.provider = self._get_configured_provider()
        self.cache = SearchCache()
        self.rate_limiter = RateLimiter(
            max_per_hour=config.get('web_search_rate_limit', 60)
        )
    
    def _get_configured_provider(self) -> WebSearchProvider:
        """Get the configured search provider."""
        provider_name = config.get('web_search_provider', 'duckduckgo')
        
        if provider_name == 'duckduckgo':
            return DuckDuckGoProvider()
        else:
            # Default to DuckDuckGo for now
            return DuckDuckGoProvider()
    
    def search(self, query: str, num_results: int = None, 
               use_cache: bool = True) -> List[SearchResult]:
        """
        Perform a web search with caching and rate limiting.
        
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
        
        # Perform search
        try:
            # Run async search in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.provider.search(query, num_results)
            )
            loop.close()
            
            # Record search and cache results
            self.rate_limiter.record_search()
            if results:
                self.cache.set(query, results)
            
            return results
            
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"Web search error: {e}", fg="red")
            return []
    
    def clear_cache(self):
        """Clear the search cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            'provider': self.provider.__class__.__name__,
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