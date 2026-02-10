"""
Web search providers package.

Contains the base classes and individual provider implementations
for web search functionality in Episodic.
"""

from episodic.web_search_providers.base import SearchResult, WebSearchProvider
from episodic.web_search_providers.duckduckgo import DuckDuckGoProvider
from episodic.web_search_providers.searx import SearxProvider
from episodic.web_search_providers.google import GoogleProvider
from episodic.web_search_providers.bing import BingProvider
from episodic.web_search_providers.brave import BraveProvider

__all__ = [
    'SearchResult',
    'WebSearchProvider',
    'DuckDuckGoProvider',
    'SearxProvider',
    'GoogleProvider',
    'BingProvider',
    'BraveProvider',
]
