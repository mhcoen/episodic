"""
Base classes for web search providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class SearchResult:
    """Represents a single web search result."""
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0  # Added relevance score field
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
