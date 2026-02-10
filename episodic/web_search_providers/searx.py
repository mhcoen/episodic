"""
Searx/SearxNG web search provider.

Open source metasearch engine provider.
Requires Searx/SearxNG instance URL, no API key needed.
"""

import asyncio
import time
from typing import List

import typer
from episodic.config import config
from episodic.configuration import get_error_color, get_warning_color
from episodic.web_search_providers.base import SearchResult, WebSearchProvider


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
        """Check if Searx instance is configured and dependencies are installed."""
        try:
            import aiohttp  # noqa: F401
            return bool(self.instance_url)
        except ImportError:
            return False

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Searx/SearxNG API."""
        # Rate limiting
        elapsed = time.time() - self.last_search_time
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)

        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "Web search requires aiohttp. Install with: pip install aiohttp"
            )

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
                            typer.secho(f"Searx returned status {response.status}", fg=get_warning_color())
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
                typer.secho(f"Searx search error: {e}", fg=get_error_color())

        return results
