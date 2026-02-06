"""
News Provider.

Fetches news headlines from NPR RSS feeds with caching.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from email.utils import parsedate_to_datetime

from .base import (
    DataProvider,
    ProviderResult,
    RefreshResult,
    CacheEntry,
)

logger = logging.getLogger(__name__)

# NPR RSS feed URLs by category
NPR_FEEDS = {
    "general": "https://feeds.npr.org/1001/rss.xml",
    "technology": "https://feeds.npr.org/1019/rss.xml",
    "tech": "https://feeds.npr.org/1019/rss.xml",  # alias
    "politics": "https://feeds.npr.org/1014/rss.xml",
    "business": "https://feeds.npr.org/1006/rss.xml",
    "science": "https://feeds.npr.org/1007/rss.xml",
    "health": "https://feeds.npr.org/1128/rss.xml",
    "world": "https://feeds.npr.org/1004/rss.xml",
}

# Category aliases
CATEGORY_ALIASES = {
    "tech": "technology",
    "sci": "science",
    "biz": "business",
}


class NewsProvider(DataProvider):
    """
    News provider using NPR RSS feeds.

    Supports headlines by category with background refresh for general news.
    """

    name = "news"
    refresh_interval_s = 1800  # 30 minutes cache TTL
    queries = ["news_headlines", "news_detail"]

    def __init__(self):
        self._default_count: int = 5
        self._voice_count: int = 3  # Headlines to read aloud
        self._cache: Dict[str, CacheEntry] = {}
        self._last_refresh: Optional[datetime] = None
        self._error_count: int = 0

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration."""
        self._default_count = config.get("default_count", 5)
        self._voice_count = config.get("voice_count", 3)

    def get(self, command: str, args: Dict[str, Any]) -> ProviderResult:
        """Get news data from cache or fetch fresh."""
        category = args.get("category", "general").lower()
        count = args.get("count", self._default_count)

        # Resolve category aliases
        category = CATEGORY_ALIASES.get(category, category)

        # Check if category is supported
        if category not in NPR_FEEDS:
            return ProviderResult.error(
                f"Unknown news category: {category}. "
                f"Available: {', '.join(sorted(set(NPR_FEEDS.keys()) - {'tech'}))}",
                self.name,
            )

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
                    payload["count"] = len(payload["headlines"])

                return ProviderResult.ok(
                    payload=payload,
                    speech_text=self._build_speech(payload["headlines"]),
                    display_text=self._build_display(payload["headlines"], category),
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
                    payload["count"] = len(payload["headlines"])

                return ProviderResult.stale(
                    payload=payload,
                    speech_text=self._build_speech(payload["headlines"]),
                    display_text=self._build_display(payload["headlines"], category),
                    source=self.name,
                    cache_key=cache_key,
                    fetched_at=entry.fetched_at,
                )

        # No cache - fetch fresh
        return self._fetch_news(category, count, cache_key)

    def refresh(self, args: Dict[str, Any]) -> RefreshResult:
        """Refresh news data (for background scheduler)."""
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
                error=result.payload.get("error", "Unknown error") if result.payload else "Unknown error",
                next_refresh_s=backoff,
            )

    def status(self) -> Dict[str, Any]:
        """Return provider status."""
        return {
            "name": self.name,
            "configured": True,  # No API key needed for RSS
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "cache_entries": len(self._cache),
            "error_count": self._error_count,
            "categories": list(set(NPR_FEEDS.keys()) - {"tech"}),
        }

    def _fetch_news(
        self, category: str, count: int, cache_key: str
    ) -> ProviderResult:
        """Fetch news from NPR RSS feed."""
        feed_url = NPR_FEEDS.get(category)
        if not feed_url:
            return ProviderResult.error(
                f"No feed URL for category: {category}",
                self.name,
                cache_key,
            )

        try:
            import urllib.request

            # Fetch RSS feed
            req = urllib.request.Request(feed_url)
            req.add_header("User-Agent", "Episodic/1.0")

            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read().decode("utf-8")

            return self._parse_rss(xml_data, category, count, cache_key)

        except urllib.error.HTTPError as e:
            return ProviderResult.error(f"HTTP error: {e.code}", self.name, cache_key)
        except urllib.error.URLError as e:
            return ProviderResult.error(f"Network error: {e.reason}", self.name, cache_key)
        except Exception as e:
            logger.exception("News fetch error")
            return ProviderResult.error(str(e), self.name, cache_key)

    def _parse_rss(
        self, xml_data: str, category: str, count: int, cache_key: str
    ) -> ProviderResult:
        """Parse RSS feed XML."""
        try:
            root = ET.fromstring(xml_data)

            # Find all items in the feed
            items = root.findall(".//item")

            headlines = []
            for item in items[:count * 2]:  # Fetch extra for filtering
                title = self._get_text(item, "title")
                description = self._get_text(item, "description")
                link = self._get_text(item, "link")
                pub_date = self._get_text(item, "pubDate")

                # Author is in dc:creator namespace
                author = self._get_text(item, "{http://purl.org/dc/elements/1.1/}creator")

                if not title:
                    continue

                # Clean up description (remove HTML tags if any)
                if description:
                    description = self._strip_html(description)

                # Parse publication date
                published_at = None
                if pub_date:
                    try:
                        published_at = parsedate_to_datetime(pub_date).isoformat()
                    except Exception:
                        pass

                headlines.append({
                    "title": title.strip(),
                    "description": description.strip() if description else "",
                    "author": author.strip() if author else "",
                    "url": link.strip() if link else "",
                    "published_at": published_at,
                })

                if len(headlines) >= count:
                    break

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
            display_text = self._build_display(headlines, category)

            # Cache the result
            now = datetime.now()
            cache_payload = payload.copy()

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

        except ET.ParseError as e:
            logger.exception("RSS parse error")
            return ProviderResult.error(f"RSS parse error: {e}", self.name, cache_key)
        except Exception as e:
            logger.exception("Error parsing news data")
            return ProviderResult.error(f"Parse error: {e}", self.name, cache_key)

    def _get_text(self, element: ET.Element, tag: str) -> Optional[str]:
        """Get text content of a child element."""
        child = element.find(tag)
        return child.text if child is not None and child.text else None

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", text)
        # Decode common entities
        clean = clean.replace("&amp;", "&")
        clean = clean.replace("&lt;", "<")
        clean = clean.replace("&gt;", ">")
        clean = clean.replace("&quot;", '"')
        clean = clean.replace("&#39;", "'")
        clean = clean.replace("&nbsp;", " ")
        # Normalize whitespace
        clean = " ".join(clean.split())
        return clean

    def _build_speech(self, headlines: List[Dict[str, Any]]) -> str:
        """Build speech text from headlines."""
        if not headlines:
            return "No headlines available."

        parts = ["Here are today's headlines."]

        ordinals = ["First", "Second", "Third", "Fourth", "Fifth"]

        # Use voice_count for speech (default 3)
        for i, h in enumerate(headlines[:self._voice_count]):
            ordinal = ordinals[i] if i < len(ordinals) else f"Number {i + 1}"
            title = h.get("title", "")
            desc = h.get("description", "")

            # Combine title and description for richer speech
            if desc:
                parts.append(f"{ordinal}: {title}. {desc}")
            else:
                parts.append(f"{ordinal}: {title}.")

        parts.append("Say a number for more details.")

        return " ".join(parts)

    def _build_display(self, headlines: List[Dict[str, Any]], category: str = "general") -> str:
        """Build display text from headlines."""
        if not headlines:
            return "📰\u00a0\u00a0No headlines available."

        category_title = category.title() if category != "general" else ""
        header = f"📰\u00a0\u00a0{category_title} Headlines" if category_title else "📰\u00a0\u00a0Headlines"
        lines = [header, ""]

        for i, h in enumerate(headlines, 1):
            title = h.get("title", "")
            lines.append(f"{i}. {title}")
            lines.append("")

        return "\n".join(lines).rstrip()
