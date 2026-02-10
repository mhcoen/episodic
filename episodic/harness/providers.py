"""
Stub providers for testing.

These implement the same interface as real providers but return scripted responses.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class WeatherResult:
    """Weather data for stub responses."""
    temp: int = 70
    condition: str = "Clear"
    high: int = 75
    low: int = 65
    humidity: int = 50
    location: str = "Test City"
    emoji: str = "☀️"


@dataclass
class NewsItem:
    """News item for stub responses."""
    title: str = "Test Headline"
    description: str = "Test description"
    author: str = "Test Author"
    url: str = "https://example.com/news"
    published_at: Optional[str] = None


@dataclass
class ProviderResult:
    """Result from a provider query (matches real ProviderResult)."""
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
        cache_key: str = "",
        ttl_seconds: int = 1800,
    ) -> "ProviderResult":
        now = datetime.now()
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
    def error(cls, message: str, source: str, cache_key: str = "") -> "ProviderResult":
        now = datetime.now()
        return cls(
            status="error",
            payload={"error": message},
            speech_text=message,
            display_text=f"Error: {message}",
            fetched_at=now,
            expires_at=now,
            source=source,
            cache_key=cache_key,
        )


@dataclass
class RefreshResult:
    """Result from refresh operation."""
    success: bool
    cache_key: str
    payload: Optional[Dict[str, Any]]
    error: Optional[str]
    next_refresh_s: int


class StubWeatherProvider:
    """
    Stub weather provider for testing.

    Returns scripted responses based on location.
    """

    name = "weather"
    refresh_interval_s = 1800
    queries = ["weather_now", "weather_forecast"]

    def __init__(self, responses: Optional[Dict[str, WeatherResult]] = None):
        self._responses = responses or {}
        self._default = WeatherResult()
        self.calls: List[tuple] = []  # Track (command, args) for assertions
        self._configured = False

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration."""
        self._configured = True

    def get(self, command: str, args: Dict[str, Any]) -> ProviderResult:
        """Get weather data."""
        self.calls.append((command, args))

        location = args.get("place", "current")
        weather = self._responses.get(location, self._default)

        payload = {
            "location": weather.location,
            "temp": weather.temp,
            "temp_unit": "F",
            "condition": weather.condition,
            "high": weather.high,
            "low": weather.low,
            "humidity": weather.humidity,
            "emoji": weather.emoji,
        }

        speech_text = (
            f"{weather.temp} degrees and {weather.condition.lower()} in {weather.location}. "
            f"High of {weather.high}, low of {weather.low}."
        )
        display_text = (
            f"{weather.emoji} {weather.location}: {weather.temp}°F {weather.condition}\n"
            f"   High: {weather.high}° Low: {weather.low}° Humidity: {weather.humidity}%"
        )

        return ProviderResult.ok(
            payload=payload,
            speech_text=speech_text,
            display_text=display_text,
            source=self.name,
            cache_key=f"weather:{command}:{location}",
        )

    def refresh(self, args: Dict[str, Any]) -> RefreshResult:
        """Refresh weather data."""
        return RefreshResult(
            success=True,
            cache_key="weather:refresh",
            payload={"refreshed": True},
            error=None,
            next_refresh_s=self.refresh_interval_s,
        )

    def status(self) -> Dict[str, Any]:
        """Return provider status."""
        return {
            "name": self.name,
            "configured": self._configured,
            "call_count": len(self.calls),
        }


class StubNewsProvider:
    """
    Stub news provider for testing.

    Returns scripted news headlines.
    """

    name = "news"
    refresh_interval_s = 1800
    queries = ["news_headlines", "news_detail"]

    def __init__(self, headlines: Optional[List[NewsItem]] = None):
        self._headlines = headlines or [
            NewsItem(title="Test Headline 1", description="Description 1"),
            NewsItem(title="Test Headline 2", description="Description 2"),
            NewsItem(title="Test Headline 3", description="Description 3"),
        ]
        self.calls: List[tuple] = []

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration."""
        pass

    def get(self, command: str, args: Dict[str, Any]) -> ProviderResult:
        """Get news data."""
        self.calls.append((command, args))

        category = args.get("category", "general")
        count = args.get("count", 5)

        headlines = [
            {
                "title": h.title,
                "description": h.description,
                "author": h.author,
                "url": h.url,
                "published_at": h.published_at,
            }
            for h in self._headlines[:count]
        ]

        payload = {
            "category": category,
            "headlines": headlines,
            "count": len(headlines),
        }

        speech_parts = ["Here are today's headlines."]
        ordinals = ["First", "Second", "Third", "Fourth", "Fifth"]
        for i, h in enumerate(headlines[:3]):
            ordinal = ordinals[i] if i < len(ordinals) else f"Number {i + 1}"
            speech_parts.append(f"{ordinal}: {h['title']}.")

        display_lines = [f"📰 {category.title()} Headlines", ""]
        for i, h in enumerate(headlines, 1):
            display_lines.append(f"{i}. {h['title']}")

        return ProviderResult.ok(
            payload=payload,
            speech_text=" ".join(speech_parts),
            display_text="\n".join(display_lines),
            source=self.name,
            cache_key=f"news:{command}:{category}",
        )

    def refresh(self, args: Dict[str, Any]) -> RefreshResult:
        """Refresh news data."""
        return RefreshResult(
            success=True,
            cache_key="news:refresh",
            payload={"refreshed": True},
            error=None,
            next_refresh_s=self.refresh_interval_s,
        )

    def status(self) -> Dict[str, Any]:
        """Return provider status."""
        return {
            "name": self.name,
            "call_count": len(self.calls),
        }


class StubTimerProvider:
    """
    Stub timer provider for testing.

    Tracks timer operations without actual timing.
    """

    name = "timer"

    def __init__(self):
        self.timers: List[Dict[str, Any]] = []
        self.calls: List[tuple] = []

    def set_timer(self, duration_s: int, label: Optional[str] = None) -> Dict[str, Any]:
        """Set a timer."""
        self.calls.append(("set", {"duration_s": duration_s, "label": label}))

        timer = {
            "id": f"timer_{len(self.timers) + 1}",
            "duration_s": duration_s,
            "label": label,
            "status": "running",
        }
        self.timers.append(timer)
        return timer

    def cancel_timer(self, timer_id: Optional[str] = None) -> bool:
        """Cancel a timer."""
        self.calls.append(("cancel", {"timer_id": timer_id}))

        if not self.timers:
            return False

        if timer_id:
            for t in self.timers:
                if t["id"] == timer_id:
                    t["status"] = "cancelled"
                    return True
            return False
        else:
            # Cancel most recent
            self.timers[-1]["status"] = "cancelled"
            return True

    def list_timers(self) -> List[Dict[str, Any]]:
        """List active timers."""
        self.calls.append(("list", {}))
        return [t for t in self.timers if t["status"] == "running"]


def create_default_stub_providers() -> Dict[str, Any]:
    """Create default stub providers for testing."""
    return {
        "weather": StubWeatherProvider({
            "current": WeatherResult(temp=72, condition="Sunny", location="Test City"),
            "Madison, WI": WeatherResult(temp=33, condition="Cloudy", location="Madison", high=36, low=28),
        }),
        "news": StubNewsProvider(),
        "timer": StubTimerProvider(),
    }
