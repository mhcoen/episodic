"""
Base classes for Data Providers.

Defines the DataProvider protocol and supporting types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Optional, Dict, Any, List
from abc import ABC, abstractmethod


@dataclass
class ProviderResult:
    """Result from a provider query."""

    status: str  # "ok" | "error" | "stale"
    payload: Dict[str, Any]
    speech_text: str
    display_text: str
    fetched_at: datetime
    expires_at: datetime
    source: str
    cache_key: str

    @classmethod
    def ok(
        cls,
        payload: Dict[str, Any],
        speech_text: str,
        display_text: str,
        source: str,
        cache_key: str,
        fetched_at: Optional[datetime] = None,
        ttl_seconds: int = 1800,
    ) -> "ProviderResult":
        """Create a successful result."""
        now = fetched_at or datetime.now()
        from datetime import timedelta

        return cls(
            status="ok",
            payload=payload,
            speech_text=speech_text,
            display_text=display_text,
            fetched_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            source=source,
            cache_key=cache_key,
        )

    @classmethod
    def error(
        cls,
        error_message: str,
        source: str,
        cache_key: str = "",
    ) -> "ProviderResult":
        """Create an error result."""
        now = datetime.now()
        return cls(
            status="error",
            payload={"error": error_message},
            speech_text=error_message,
            display_text=f"Error: {error_message}",
            fetched_at=now,
            expires_at=now,
            source=source,
            cache_key=cache_key,
        )

    @classmethod
    def stale(
        cls,
        payload: Dict[str, Any],
        speech_text: str,
        display_text: str,
        source: str,
        cache_key: str,
        fetched_at: datetime,
    ) -> "ProviderResult":
        """Create a stale result (cached data past expiration)."""
        return cls(
            status="stale",
            payload=payload,
            speech_text=speech_text,
            display_text=display_text + " (cached)",
            fetched_at=fetched_at,
            expires_at=fetched_at,  # Already expired
            source=source,
            cache_key=cache_key,
        )


@dataclass
class RefreshResult:
    """Result from a refresh operation."""

    success: bool
    cache_key: str
    payload: Optional[Dict[str, Any]]
    error: Optional[str]
    next_refresh_s: int  # Seconds until next refresh


@dataclass
class CacheEntry:
    """In-memory cache entry."""

    key: str
    payload: Dict[str, Any]
    speech_text: str
    display_text: str
    fetched_at: datetime
    expires_at: datetime
    hit_count: int = 0


class ProviderError(Exception):
    """Base class for provider errors."""

    pass


class NotConfigured(ProviderError):
    """Provider missing required configuration (API key, etc.)."""

    pass


class RateLimited(ProviderError):
    """API rate limit exceeded."""

    def __init__(self, message: str, retry_after_s: int = 60):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class SourceUnavailable(ProviderError):
    """External source not responding."""

    pass


class DataProvider(ABC):
    """Abstract base class for data providers."""

    name: str
    refresh_interval_s: int
    queries: List[str]

    @abstractmethod
    def get(self, command: str, args: Dict[str, Any]) -> ProviderResult:
        """
        Get cached data. Returns immediately.
        If cache is empty/stale, returns status="stale" with best-effort data.
        """
        ...

    @abstractmethod
    def refresh(self, args: Dict[str, Any]) -> RefreshResult:
        """
        Fetch fresh data from source. Called by scheduler.
        Updates internal cache. Returns next refresh interval.
        """
        ...

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Apply provider-specific configuration.
        Called on startup and when preferences change.
        """
        ...

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """
        Return provider health: last refresh, cache age, error count, etc.
        """
        ...


class ProviderRegistry:
    """Central registry for all data providers."""

    def __init__(self):
        self._providers: Dict[str, DataProvider] = {}
        self._query_map: Dict[str, str] = {}  # command -> provider name

    def register(self, provider: DataProvider) -> None:
        """Register a provider. Called at startup."""
        self._providers[provider.name] = provider
        for query in provider.queries:
            self._query_map[query] = provider.name

    def unregister(self, name: str) -> None:
        """Unregister a provider."""
        if name in self._providers:
            provider = self._providers.pop(name)
            for query in provider.queries:
                self._query_map.pop(query, None)

    def get_provider(self, name: str) -> Optional[DataProvider]:
        """Get provider by name."""
        return self._providers.get(name)

    def route_query(self, command: str) -> Optional[DataProvider]:
        """Find provider that handles this command."""
        provider_name = self._query_map.get(command)
        if provider_name:
            return self._providers.get(provider_name)
        return None

    def all_providers(self) -> List[DataProvider]:
        """List all registered providers."""
        return list(self._providers.values())
