"""
Brave Search API provider using official client.

Requires API key from https://api.search.brave.com/
"""

from typing import List

import typer
from episodic.config import config
from episodic.configuration import get_error_color, get_warning_color
from episodic.web_search_providers.base import SearchResult, WebSearchProvider


class BraveProvider(WebSearchProvider):
    """
    Brave Search API provider using official client.
    Requires API key from https://api.search.brave.com/
    """

    def __init__(self):
        self.api_key = config.get('brave_api_key') or config.get('BRAVE_API_KEY')

    def is_available(self) -> bool:
        """Check if Brave Search is configured."""
        return bool(self.api_key)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Brave Search API."""
        if not self.is_available():
            if config.get('debug'):
                typer.secho(
                    "Brave Search requires BRAVE_API_KEY from https://api.search.brave.com/",
                    fg="yellow"
                )
            return []

        try:
            from brave_search_python_client import BraveSearch, WebSearchRequest
        except ImportError:
            typer.secho(
                "\u26a0\ufe0f  Brave Search requires the official client. Install with:\n"
                "    pip install brave-search-python-client",
                fg=get_warning_color()
            )
            return []

        results = []

        try:
            # Initialize Brave Search client
            brave = BraveSearch(api_key=self.api_key)

            # Create search request
            request = WebSearchRequest(
                q=query,
                count=num_results,
                search_lang='en',
                result_filter='web'
            )

            # Perform search
            response = await brave.web(request)

            # Parse results from response
            if response and hasattr(response, 'web') and hasattr(response.web, 'results'):
                for result in response.web.results[:num_results]:
                    if hasattr(result, 'title') and hasattr(result, 'url'):
                        snippet = getattr(result, 'description', '')
                        results.append(SearchResult(
                            title=result.title,
                            url=result.url,
                            snippet=snippet
                        ))

        except Exception as e:
            if config.get('debug'):
                typer.secho(f"Brave search error: {e}", fg=get_error_color())
            # Check for common errors
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise Exception("Invalid Brave API key")
            elif "429" in str(e):
                raise Exception("Brave API rate limit exceeded")
            else:
                raise e

        return results
