"""
DuckDuckGo web search provider.

Free web search provider using DuckDuckGo.
No API key required, uses official ddgs library.
"""

import asyncio
import time
from typing import List

import typer
from episodic.config import config
from episodic.configuration import get_error_color
from episodic.web_search_providers.base import SearchResult, WebSearchProvider


class DuckDuckGoProvider(WebSearchProvider):
    """
    Free web search provider using DuckDuckGo.
    No API key required, uses official ddgs library.
    """

    def __init__(self):
        self.last_search_time = 0
        self.min_delay = 1.0  # Minimum seconds between searches

    def is_available(self) -> bool:
        """Check if DuckDuckGo library is installed."""
        try:
            from ddgs import DDGS  # noqa: F401
            return True
        except ImportError:
            return False

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search DuckDuckGo using ddgs library."""
        # Rate limiting
        elapsed = time.time() - self.last_search_time
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)

        try:
            from ddgs import DDGS
        except ImportError:
            raise ImportError(
                "Web search requires ddgs library. Install with: pip install ddgs"
            )

        results = []

        try:
            # Run synchronous DDGS search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ddgs_results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=num_results))
            )

            for result in ddgs_results:
                if isinstance(result, dict):
                    title = result.get('title', '')
                    url = result.get('href', '')
                    snippet = result.get('body', '')

                    if title and url:
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet
                        ))

            self.last_search_time = time.time()

        except Exception as e:
            if config.get('debug'):
                typer.secho(f"DuckDuckGo search error: {e}", fg=get_error_color())

        return results
