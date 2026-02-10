"""
Bing Search API provider.

Requires API key from Azure Cognitive Services.
"""

from typing import List

import typer
from episodic.config import config
from episodic.configuration import get_error_color
from episodic.web_search_providers.base import SearchResult, WebSearchProvider


class BingProvider(WebSearchProvider):
    """
    Bing Search API provider.
    Requires API key from Azure Cognitive Services.
    """

    def __init__(self):
        self.api_key = config.get('bing_api_key') or config.get('BING_API_KEY')
        self.endpoint = config.get('bing_endpoint', 'https://api.bing.microsoft.com/v7.0/search')

    def is_available(self) -> bool:
        """Check if Bing Search is configured and dependencies are installed."""
        try:
            import aiohttp  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

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
            raise ImportError(
                "Web search requires aiohttp. Install with: pip install aiohttp"
            )

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
                            typer.secho(f"Bing API error: {error_data}", fg=get_error_color())
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
                typer.secho(f"Bing search error: {e}", fg=get_error_color())

        return results
