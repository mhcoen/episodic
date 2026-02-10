"""
Google Custom Search API provider.

Requires API key and search engine ID.
"""

from typing import List

import typer
from episodic.config import config
from episodic.configuration import get_error_color
from episodic.web_search_providers.base import SearchResult, WebSearchProvider


class GoogleProvider(WebSearchProvider):
    """
    Google Custom Search API provider.
    Requires API key and search engine ID.
    """

    def __init__(self):
        self.api_key = config.get('google_api_key') or config.get('GOOGLE_API_KEY')
        self.search_engine_id = config.get('google_search_engine_id') or config.get('GOOGLE_SEARCH_ENGINE_ID')

    def is_available(self) -> bool:
        """Check if Google Search is configured and dependencies are installed."""
        try:
            import aiohttp  # noqa: F401
            return bool(self.api_key and self.search_engine_id)
        except ImportError:
            return False

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
            raise ImportError(
                "Web search requires aiohttp. Install with: pip install aiohttp"
            )

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
                        error_data = await response.text()
                        if config.get('debug'):
                            typer.secho(f"Google API error: {error_data}", fg=get_error_color())
                        # Parse common Google API errors
                        if response.status == 403:
                            if "custom search api has not been used" in error_data.lower():
                                raise Exception("Google Custom Search API not enabled. Enable it at: https://console.cloud.google.com/apis/library/customsearch.googleapis.com")
                            else:
                                raise Exception("Invalid API key or insufficient permissions")
                        elif response.status == 400:
                            if "cx" in error_data.lower() or "invalid value" in error_data.lower():
                                raise Exception("Invalid search engine ID")
                            else:
                                raise Exception(f"Bad request: {error_data[:100]}")
                        else:
                            raise Exception(f"API error (status {response.status})")
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
                typer.secho(f"Google search error: {e}", fg=get_error_color())

        return results
