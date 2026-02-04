"""
Weather Command Handlers.

Handles weather-related utility commands.
"""

import sqlite3
from typing import Optional

from ..types import UtilityQuery, UtilityResult
from ..providers.weather import WeatherProvider


# Global provider instance (initialized on first use)
_weather_provider: Optional[WeatherProvider] = None


def get_weather_provider() -> WeatherProvider:
    """Get or create the weather provider."""
    global _weather_provider
    if _weather_provider is None:
        _weather_provider = WeatherProvider()
    return _weather_provider


def handle_weather_now(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle weather_now command.

    Gets current weather for a location.

    Args in query:
        place: Location (optional, defaults to "current")
    """
    provider = get_weather_provider()

    # Get user preferences for configuration
    if conn is not None:
        from ..db import get_preference

        temp_unit = get_preference(conn, "temp_unit") or "F"
        location_home = get_preference(conn, "location_home")
        location_work = get_preference(conn, "location_work")

        provider.configure({
            "temp_unit": temp_unit,
            "location_home": location_home,
            "location_work": location_work,
            "default_location": location_home,  # Use home as default
        })

    place = query.args.get("place", "current")
    result = provider.get("weather_now", {"place": place})

    if result.status == "error":
        return UtilityResult.error(
            result.payload.get("error", "weather_error"),
            result.speech_text,
        )

    return UtilityResult.ok(
        display=result.display_text,
        speech=result.speech_text,
        **result.payload,
    )


def handle_weather_forecast(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle weather_forecast command.

    Gets weather forecast for a location.

    Args in query:
        place: Location (optional, defaults to "current")
        days: Number of days (optional, defaults to 5)
    """
    provider = get_weather_provider()

    # Get user preferences for configuration
    if conn is not None:
        from ..db import get_preference

        temp_unit = get_preference(conn, "temp_unit") or "F"
        location_home = get_preference(conn, "location_home")
        location_work = get_preference(conn, "location_work")

        provider.configure({
            "temp_unit": temp_unit,
            "location_home": location_home,
            "location_work": location_work,
            "default_location": location_home,
        })

    place = query.args.get("place", "current")
    result = provider.get("weather_forecast", {"place": place})

    if result.status == "error":
        return UtilityResult.error(
            result.payload.get("error", "weather_error"),
            result.speech_text,
        )

    return UtilityResult.ok(
        display=result.display_text,
        speech=result.speech_text,
        **result.payload,
    )


# Command routing
WEATHER_HANDLERS = {
    "weather_now": handle_weather_now,
    "weather_forecast": handle_weather_forecast,
}


def dispatch_weather_command(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """Dispatch a weather category command to the appropriate handler."""
    handler = WEATHER_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown weather command: {query.command}"
        )

    return handler(query, conn, user_tz)
