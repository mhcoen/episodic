"""
Weather Provider.

Fetches weather data from OpenWeatherMap API with caching.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from .base import (
    DataProvider,
    ProviderResult,
    RefreshResult,
    CacheEntry,
    NotConfigured,
    RateLimited,
    SourceUnavailable,
)

logger = logging.getLogger(__name__)

# OpenWeatherMap condition code to description and emoji
CONDITION_MAP = {
    # Thunderstorm
    (200, 232): ("thunderstorm", "⛈️"),
    # Drizzle
    (300, 321): ("drizzle", "🌧️"),
    # Rain
    (500, 504): ("rain", "🌧️"),
    (511, 511): ("freezing rain", "🌨️"),
    (520, 531): ("showers", "🌦️"),
    # Snow
    (600, 622): ("snow", "❄️"),
    # Atmosphere (fog, mist, etc.)
    (701, 781): ("foggy", "🌫️"),
    # Clear
    (800, 800): ("clear", "☀️"),
    # Clouds
    (801, 801): ("partly cloudy", "⛅"),
    (802, 802): ("cloudy", "🌤️"),
    (803, 804): ("overcast", "☁️"),
}


def _get_condition(code: int) -> tuple:
    """Get condition description and emoji from weather code."""
    for (low, high), (desc, emoji) in CONDITION_MAP.items():
        if low <= code <= high:
            return desc, emoji
    return "unknown", "🌡️"


class WeatherProvider(DataProvider):
    """
    Weather provider using OpenWeatherMap API.

    Supports current weather and forecasts.
    """

    name = "weather"
    refresh_interval_s = 1800  # 30 minutes
    queries = ["weather_now", "weather_day", "weather_forecast"]

    def __init__(self):
        self._api_key: Optional[str] = None
        self._temp_unit: str = "F"
        self._default_location: Optional[str] = None
        self._home_location: Optional[str] = None
        self._work_location: Optional[str] = None
        self._detected_location: Optional[str] = None
        self._cache: Dict[str, CacheEntry] = {}
        self._last_refresh: Optional[datetime] = None
        self._error_count: int = 0
        self._ip_location_cache: Optional[str] = None

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration."""
        self._api_key = config.get("api_key") or os.environ.get("OPENWEATHERMAP_API_KEY")
        self._temp_unit = config.get("temp_unit", "F")
        self._default_location = config.get("default_location")
        self._home_location = config.get("location_home")
        self._work_location = config.get("location_work")
        self._detected_location = config.get("location_detected")

    def get(self, command: str, args: Dict[str, Any]) -> ProviderResult:
        """Get weather data from cache or fetch fresh."""
        place = args.get("place", "current")
        location = self._resolve_location(place)

        if not location:
            return ProviderResult.error(
                "No location specified and no default configured",
                self.name,
            )

        cache_key = f"weather:{command}:{location.lower()}"

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            entry.hit_count += 1

            if datetime.now() < entry.expires_at:
                return ProviderResult.ok(
                    payload=entry.payload,
                    speech_text=entry.speech_text,
                    display_text=entry.display_text,
                    source=self.name,
                    cache_key=cache_key,
                    fetched_at=entry.fetched_at,
                    ttl_seconds=int((entry.expires_at - datetime.now()).total_seconds()),
                )
            else:
                # Stale cache - return but note it's stale
                return ProviderResult.stale(
                    payload=entry.payload,
                    speech_text=entry.speech_text,
                    display_text=entry.display_text,
                    source=self.name,
                    cache_key=cache_key,
                    fetched_at=entry.fetched_at,
                )

        # No cache - fetch fresh
        result = self._fetch_weather(location, command, args)
        return result

    def refresh(self, args: Dict[str, Any]) -> RefreshResult:
        """Refresh weather data."""
        location = args.get("location") or self._resolve_location("current")

        if not location:
            return RefreshResult(
                success=False,
                cache_key="",
                payload=None,
                error="No location configured",
                next_refresh_s=self.refresh_interval_s,
            )

        result = self._fetch_weather(location, "weather_now", {})

        if result.status == "ok":
            self._last_refresh = datetime.now()
            self._error_count = 0
            return RefreshResult(
                success=True,
                cache_key=result.cache_key,
                payload=result.payload,
                error=None,
                next_refresh_s=self.refresh_interval_s,
            )
        else:
            self._error_count += 1
            # Exponential backoff on errors
            backoff = min(self.refresh_interval_s * (2 ** self._error_count), 3600)
            return RefreshResult(
                success=False,
                cache_key=result.cache_key,
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
            "default_location": self._default_location,
            "temp_unit": self._temp_unit,
        }

    def _resolve_location(self, place: str) -> Optional[str]:
        """
        Resolve location placeholder to actual location.

        Resolution order:
        1. Explicit location (not "current"/"home"/"work")
        2. For "home" -> location_home preference
        3. For "work" -> location_work preference
        4. For "current" -> default_location (config file)
        5. For "current"/"home" -> location_home preference
        6. Detected location (cached IP lookup)
        7. Fresh IP geolocation (then cache)
        """
        place_lower = place.lower()

        if place_lower == "work":
            return self._work_location
        elif place_lower == "home":
            return self._home_location or self._get_fallback_location()
        elif place_lower == "current":
            # Try default_location first (from config file)
            if self._default_location:
                return self._default_location
            # Then home location
            if self._home_location:
                return self._home_location
            # Fall back to detected/IP location
            return self._get_fallback_location()
        else:
            # Explicit location
            return place

    def _get_fallback_location(self) -> Optional[str]:
        """Get fallback location from detected or IP geolocation."""
        # Use cached detected location first
        if self._detected_location:
            return self._detected_location

        # Try IP geolocation (with in-memory cache)
        if self._ip_location_cache:
            return self._ip_location_cache

        # Fetch from IP
        ip_location = self._get_location_from_ip()
        if ip_location:
            self._ip_location_cache = ip_location
        return ip_location

    def _get_location_from_ip(self) -> Optional[str]:
        """Get location from IP address using ip-api.com."""
        import urllib.request
        import json

        try:
            url = "http://ip-api.com/json/?fields=city,regionCode,country"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("city"):
                    region = data.get("regionCode", "")
                    if region:
                        return f"{data['city']}, {region}"
                    return data["city"]
        except Exception as e:
            logger.debug(f"IP geolocation failed: {e}")
        return None

    def get_new_ip_location(self) -> Optional[str]:
        """
        Get newly detected IP location that should be saved to DB.

        Returns the IP-detected location if it was just fetched and differs
        from what was loaded from the database (detected_location).
        """
        if self._ip_location_cache and self._ip_location_cache != self._detected_location:
            return self._ip_location_cache
        return None

    def _fetch_weather(
        self, location: str, command: str, args: Dict[str, Any]
    ) -> ProviderResult:
        """Fetch weather from OpenWeatherMap API."""
        if not self._api_key:
            return ProviderResult.error(
                "Weather requires OPENWEATHERMAP_API_KEY environment variable",
                self.name,
            )

        cache_key = f"weather:{command}:{location.lower()}"

        try:
            import urllib.request
            import urllib.parse
            import json

            # Build API URL
            units = "imperial" if self._temp_unit == "F" else "metric"
            params = urllib.parse.urlencode({
                "q": location,
                "appid": self._api_key,
                "units": units,
            })

            if command == "weather_forecast":
                url = f"https://api.openweathermap.org/data/2.5/forecast?{params}"
            else:
                url = f"https://api.openweathermap.org/data/2.5/weather?{params}"

            # Make request
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if command == "weather_forecast":
                return self._parse_forecast(data, location, cache_key)
            else:
                return self._parse_current(data, location, cache_key)

        except urllib.error.HTTPError as e:
            if e.code == 401:
                return ProviderResult.error("Invalid API key", self.name, cache_key)
            elif e.code == 404:
                return ProviderResult.error(f"Location not found: {location}", self.name, cache_key)
            elif e.code == 429:
                return ProviderResult.error("Rate limit exceeded", self.name, cache_key)
            else:
                return ProviderResult.error(f"API error: {e.code}", self.name, cache_key)
        except urllib.error.URLError as e:
            return ProviderResult.error(f"Network error: {e.reason}", self.name, cache_key)
        except Exception as e:
            logger.exception("Weather fetch error")
            return ProviderResult.error(str(e), self.name, cache_key)

    def _parse_current(
        self, data: Dict[str, Any], location: str, cache_key: str
    ) -> ProviderResult:
        """Parse current weather response."""
        try:
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            wind = data.get("wind", {})

            temp = round(main.get("temp", 0))
            feels_like = round(main.get("feels_like", 0))
            humidity = main.get("humidity", 0)
            high = round(main.get("temp_max", temp))
            low = round(main.get("temp_min", temp))

            condition_code = weather.get("id", 800)
            condition, emoji = _get_condition(condition_code)

            wind_speed = round(wind.get("speed", 0))

            city = data.get("name", location)

            unit_symbol = "°F" if self._temp_unit == "F" else "°C"
            wind_unit = "mph" if self._temp_unit == "F" else "m/s"

            payload = {
                "location": city,
                "temp": temp,
                "temp_unit": self._temp_unit,
                "feels_like": feels_like,
                "condition": condition,
                "condition_code": condition_code,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "high": high,
                "low": low,
            }

            speech_text = f"{temp} degrees and {condition} in {city}. High of {high}, low of {low}."
            display_text = (
                f"{emoji}\u00a0\u00a0{city}: {temp}{unit_symbol} {condition.title()}\n"
                f"   High: {high}° Low: {low}° Humidity: {humidity}%"
            )

            # Cache the result
            now = datetime.now()
            self._cache[cache_key] = CacheEntry(
                key=cache_key,
                payload=payload,
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
            logger.exception("Error parsing weather data")
            return ProviderResult.error(f"Parse error: {e}", self.name, cache_key)

    def _parse_forecast(
        self, data: Dict[str, Any], location: str, cache_key: str
    ) -> ProviderResult:
        """Parse forecast response."""
        try:
            city = data.get("city", {}).get("name", location)
            forecasts = data.get("list", [])

            # Group by day
            daily = {}
            for item in forecasts:
                dt = datetime.fromtimestamp(item["dt"])
                day_key = dt.strftime("%Y-%m-%d")

                if day_key not in daily:
                    daily[day_key] = {
                        "date": dt,
                        "temps": [],
                        "conditions": [],
                    }

                daily[day_key]["temps"].append(item["main"]["temp"])
                daily[day_key]["conditions"].append(item["weather"][0]["id"])

            # Build forecast list
            forecast_list = []
            unit_symbol = "°F" if self._temp_unit == "F" else "°C"

            for day_key in sorted(daily.keys())[:5]:  # Next 5 days
                day_data = daily[day_key]
                high = round(max(day_data["temps"]))
                low = round(min(day_data["temps"]))

                # Most common condition
                from collections import Counter
                most_common = Counter(day_data["conditions"]).most_common(1)[0][0]
                condition, emoji = _get_condition(most_common)

                forecast_list.append({
                    "date": day_data["date"].strftime("%A"),
                    "high": high,
                    "low": low,
                    "condition": condition,
                    "emoji": emoji,
                })

            payload = {
                "location": city,
                "temp_unit": self._temp_unit,
                "forecast": forecast_list,
            }

            # Build speech and display text
            speech_parts = [f"Forecast for {city}."]
            display_lines = [f"📅\u00a0\u00a0{city} Forecast"]

            for f in forecast_list[:3]:  # Speak first 3 days
                speech_parts.append(f"{f['date']}: {f['condition']}, high of {f['high']}, low of {f['low']}.")

            for f in forecast_list:
                display_lines.append(
                    f"   {f['emoji']}\u00a0\u00a0{f['date']}: {f['high']}{unit_symbol}/{f['low']}{unit_symbol} {f['condition'].title()}"
                )

            speech_text = " ".join(speech_parts)
            display_text = "\n".join(display_lines)

            # Cache the result
            now = datetime.now()
            self._cache[cache_key] = CacheEntry(
                key=cache_key,
                payload=payload,
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
            logger.exception("Error parsing forecast data")
            return ProviderResult.error(f"Parse error: {e}", self.name, cache_key)
