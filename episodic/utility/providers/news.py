"""
News Provider.

Fetches news headlines from NewsAPI with caching.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from .base import (
    DataProvider,
    ProviderResult,
    RefreshResult,
    CacheEntry,
)

logger = logging.getLogger(__name__)

# Category aliases
CATEGORY_ALIASES = {
    "tech": "technology",
    "sci": "science",
    "biz": "business",
    "sports": "sports",
    "health": "health",
    "entertainment": "entertainment",
    "ent": "entertainment",
}


class NewsProvider(DataProvider):
    """
    News provider using NewsAPI.

    Supports headlines and topic-based queries.
    """

    name = "news"
    refresh_interval_s = 3600  # 1 hour
    queries = ["news_headlines", "news_topic"]

    def __init__(self):
        self._api_key: Optional[str] = None
        self._country: str = "us"
        self._default_count: int = 5
        self._cache: Dict[str, CacheEntry] = {}
        self._last_refresh: Optional[datetime] = None
        self._error_count: int = 0

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration."""
        self._api_key = config.get("api_key") or os.environ.get("NEWSAPI_KEY")
        self._country = config.get("country", "us")
        self._default_count = config.get("default_count", 5)

    def get(self, command: str, args: Dict[str, Any]) -> ProviderResult:
        """Get news data from cache or fetch fresh."""
        category = args.get("category", "general")
        count = args.get("count", self._default_count)

        # Resolve category aliases
        category = CATEGORY_ALIASES.get(category.lower(), category.lower())

        cache_key = f"news:{command}:{category}"

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            entry.hit_count += 1

            if datetime.now() < entry.expires_at:
                # Return cached, possibly truncated to requested count
                payload = entry.payload.copy()
                if "headlines" in payload:
                    payload["headlines"] = payload["headlines"][:count]

                return ProviderResult.ok(
                    payload=payload,
                    speech_text=self._build_speech(payload["headlines"]),
                    display_text=self._build_display(payload["headlines"]),
                    source=self.name,
                    cache_key=cache_key,
                    fetched_at=entry.fetched_at,
                    ttl_seconds=int((entry.expires_at - datetime.now()).total_seconds()),
                )
            else:
                # Stale cache
                payload = entry.payload.copy()
                if "headlines" in payload:
                    payload["headlines"] = payload["headlines"][:count]

                return ProviderResult.stale(
                    payload=payload,
                    speech_text=self._build_speech(payload["headlines"]),
                    display_text=self._build_display(payload["headlines"]),
                    source=self.name,
                    cache_key=cache_key,
                    fetched_at=entry.fetched_at,
                )

        # No cache - fetch fresh
        return self._fetch_news(category, count, cache_key)

    def refresh(self, args: Dict[str, Any]) -> RefreshResult:
        """Refresh news data."""
        category = args.get("category", "general")
        cache_key = f"news:news_headlines:{category}"

        result = self._fetch_news(category, 10, cache_key)

        if result.status == "ok":
            self._last_refresh = datetime.now()
            self._error_count = 0
            return RefreshResult(
                success=True,
                cache_key=cache_key,
                payload=result.payload,
                error=None,
                next_refresh_s=self.refresh_interval_s,
            )
        else:
            self._error_count += 1
            backoff = min(self.refresh_interval_s * (2 ** self._error_count), 7200)
            return RefreshResult(
                success=False,
                cache_key=cache_key,
                payload=None,
                error=result.payload.get("error", "Unknown error"),
                next_refresh_s=backoff,
            )

    def status(self) -> Dict[str, Any]:
        """Return provider status."""
        return {
            "name": self.name,
            "configured": self._api_key is not None,
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "cache_entries": len(self._cache),
            "error_count": self._error_count,
            "country": self._country,
        }

    def _fetch_news(
        self, category: str, count: int, cache_key: str
    ) -> ProviderResult:
        """Fetch news from NewsAPI."""
        if not self._api_key:
            return ProviderResult.error(
                "News requires NEWSAPI_KEY environment variable",
                self.name,
            )

        try:
            import urllib.request
            import urllib.parse
            import json

            # Build API URL
            params = urllib.parse.urlencode({
                "country": self._country,
                "category": category,
                "pageSize": min(count * 2, 20),  # Fetch extra for filtering
                "apiKey": self._api_key,
            })

            url = f"https://newsapi.org/v2/top-headlines?{params}"

            # Make request
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Episodic/1.0")

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if data.get("status") != "ok":
                error_msg = data.get("message", "Unknown API error")
                return ProviderResult.error(error_msg, self.name, cache_key)

            return self._parse_headlines(data, category, count, cache_key)

        except urllib.error.HTTPError as e:
            if e.code == 401:
                return ProviderResult.error("Invalid API key", self.name, cache_key)
            elif e.code == 429:
                return ProviderResult.error("Rate limit exceeded", self.name, cache_key)
            else:
                return ProviderResult.error(f"API error: {e.code}", self.name, cache_key)
        except urllib.error.URLError as e:
            return ProviderResult.error(f"Network error: {e.reason}", self.name, cache_key)
        except Exception as e:
            logger.exception("News fetch error")
            return ProviderResult.error(str(e), self.name, cache_key)

    def _parse_headlines(
        self, data: Dict[str, Any], category: str, count: int, cache_key: str
    ) -> ProviderResult:
        """Parse headlines response."""
        try:
            articles = data.get("articles", [])

            headlines = []
            for article in articles[:count]:
                source = article.get("source", {}).get("name", "Unknown")
                title = article.get("title", "")
                description = article.get("description", "")
                url = article.get("url", "")

                # Skip articles with no title or "[Removed]" content
                if not title or title == "[Removed]":
                    continue

                headlines.append({
                    "title": title,
                    "source": source,
                    "description": description or "",
                    "url": url,
                })

            if not headlines:
                return ProviderResult.error(
                    f"No headlines found for category: {category}",
                    self.name,
                    cache_key,
                )

            payload = {
                "category": category,
                "headlines": headlines,
                "count": len(headlines),
            }

            speech_text = self._build_speech(headlines)
            display_text = self._build_display(headlines)

            # Cache the result (cache more than requested for future queries)
            now = datetime.now()
            cache_payload = payload.copy()
            cache_payload["headlines"] = headlines  # Full list

            self._cache[cache_key] = CacheEntry(
                key=cache_key,
                payload=cache_payload,
                speech_text=speech_text,
                display_text=display_text,
                fetched_at=now,
                expires_at=now + timedelta(seconds=self.refresh_interval_s),
            )

            return ProviderResult.ok(
                payload=payload,
                speech_text=speech_text,
                display_text=display_text,
                source=self.name,
                cache_key=cache_key,
                ttl_seconds=self.refresh_interval_s,
            )

        except Exception as e:
            logger.exception("Error parsing news data")
            return ProviderResult.error(f"Parse error: {e}", self.name, cache_key)

    def _build_speech(self, headlines: List[Dict[str, Any]]) -> str:
        """Build speech text from headlines."""
        if not headlines:
            return "No headlines available."

        parts = ["Here are today's headlines."]

        ordinals = ["First", "Second", "Third", "Fourth", "Fifth"]

        for i, h in enumerate(headlines[:5]):
            ordinal = ordinals[i] if i < len(ordinals) else f"Number {i + 1}"
            source = h.get("source", "")
            title = h.get("title", "")
            desc = h.get("description", "")

            if source:
                parts.append(f"{ordinal}, from {source}: {title}.")
            else:
                parts.append(f"{ordinal}: {title}.")

            # Add description for first 2 headlines only (brevity)
            if i < 2 and desc:
                parts.append(desc)

        return " ".join(parts)

    def _build_display(self, headlines: List[Dict[str, Any]]) -> str:
        """Build display text from headlines."""
        if not headlines:
            return "📰 No headlines available."

        lines = ["📰 Top Headlines"]

        for i, h in enumerate(headlines, 1):
            source = h.get("source", "")
            title = h.get("title", "")
            desc = h.get("description", "")

            if source:
                lines.append(f"{i}. {title} — {source}")
            else:
                lines.append(f"{i}. {title}")

            if desc:
                # Truncate long descriptions
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                lines.append(f"   {desc}")

        return "\n".join(lines)
