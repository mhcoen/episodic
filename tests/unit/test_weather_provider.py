"""
Tests for Weather Provider.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import json

from episodic.utility.providers.weather import WeatherProvider, _get_condition
from episodic.utility.providers.base import ProviderResult


class TestWeatherConditions:
    """Test condition code mapping."""

    def test_clear_sky(self):
        desc, emoji = _get_condition(800)
        assert desc == "clear"
        assert emoji == "☀️"

    def test_thunderstorm(self):
        desc, emoji = _get_condition(211)
        assert desc == "thunderstorm"
        assert emoji == "⛈️"

    def test_rain(self):
        desc, emoji = _get_condition(501)
        assert desc == "rain"
        assert emoji == "🌧️"

    def test_snow(self):
        desc, emoji = _get_condition(601)
        assert desc == "snow"
        assert emoji == "❄️"

    def test_cloudy(self):
        desc, emoji = _get_condition(803)
        assert desc == "overcast"
        assert emoji == "☁️"

    def test_unknown_code(self):
        desc, emoji = _get_condition(9999)
        assert desc == "unknown"


class TestWeatherProvider:
    """Test WeatherProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a configured provider."""
        p = WeatherProvider()
        p.configure({
            "api_key": "test_api_key",
            "temp_unit": "F",
            "default_location": "Chicago",
            "location_home": "Chicago",
            "location_work": "Evanston",
        })
        return p

    def test_configure(self, provider):
        """Test configuration."""
        assert provider._api_key == "test_api_key"
        assert provider._temp_unit == "F"
        assert provider._default_location == "Chicago"

    def test_status(self, provider):
        """Test status method."""
        status = provider.status()
        assert status["name"] == "weather"
        assert status["configured"] is True
        assert status["temp_unit"] == "F"

    def test_resolve_location_current(self, provider):
        """Test location resolution for 'current'."""
        location = provider._resolve_location("current")
        assert location == "Chicago"

    def test_resolve_location_home(self, provider):
        """Test location resolution for 'home'."""
        location = provider._resolve_location("home")
        assert location == "Chicago"

    def test_resolve_location_work(self, provider):
        """Test location resolution for 'work'."""
        location = provider._resolve_location("work")
        assert location == "Evanston"

    def test_resolve_location_explicit(self, provider):
        """Test location resolution for explicit city."""
        location = provider._resolve_location("Boston")
        assert location == "Boston"

    def test_no_api_key(self):
        """Test error when no API key configured."""
        provider = WeatherProvider()
        result = provider.get("weather_now", {"place": "Chicago"})
        assert result.status == "error"
        assert "OPENWEATHERMAP_API_KEY" in result.speech_text

    def test_no_location(self, provider):
        """Test error when no location available."""
        provider._default_location = None
        result = provider.get("weather_now", {"place": "current"})
        assert result.status == "error"
        assert "No location" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_fetch_current_weather(self, mock_urlopen, provider):
        """Test fetching current weather."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "name": "Chicago",
            "main": {
                "temp": 72,
                "feels_like": 70,
                "humidity": 45,
                "temp_max": 78,
                "temp_min": 65,
            },
            "weather": [{"id": 800, "description": "clear sky"}],
            "wind": {"speed": 8},
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("weather_now", {"place": "Chicago"})

        assert result.status == "ok"
        assert result.payload["temp"] == 72
        assert result.payload["condition"] == "clear"
        assert result.payload["location"] == "Chicago"
        assert "72" in result.speech_text
        assert "☀️" in result.display_text

    @patch("urllib.request.urlopen")
    def test_cache_hit(self, mock_urlopen, provider):
        """Test that cached results are returned."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "name": "Chicago",
            "main": {"temp": 72, "humidity": 45, "temp_max": 78, "temp_min": 65},
            "weather": [{"id": 800}],
            "wind": {"speed": 8},
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # First call - should fetch
        result1 = provider.get("weather_now", {"place": "Chicago"})
        assert mock_urlopen.call_count == 1

        # Second call - should use cache
        result2 = provider.get("weather_now", {"place": "Chicago"})
        assert mock_urlopen.call_count == 1  # Not called again
        assert result2.status == "ok"

    @patch("urllib.request.urlopen")
    def test_fetch_forecast(self, mock_urlopen, provider):
        """Test fetching weather forecast."""
        now = datetime.now()
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "city": {"name": "Chicago"},
            "list": [
                {
                    "dt": int(now.timestamp()),
                    "main": {"temp": 70},
                    "weather": [{"id": 800}],
                },
                {
                    "dt": int((now + timedelta(hours=3)).timestamp()),
                    "main": {"temp": 75},
                    "weather": [{"id": 800}],
                },
                {
                    "dt": int((now + timedelta(days=1)).timestamp()),
                    "main": {"temp": 68},
                    "weather": [{"id": 500}],
                },
            ],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("weather_forecast", {"place": "Chicago"})

        assert result.status == "ok"
        assert "forecast" in result.payload
        assert "📅" in result.display_text


class TestWeatherProviderErrors:
    """Test error handling."""

    @pytest.fixture
    def provider(self):
        p = WeatherProvider()
        p.configure({"api_key": "test_key"})
        return p

    @patch("urllib.request.urlopen")
    def test_invalid_api_key(self, mock_urlopen, provider):
        """Test handling of invalid API key."""
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(None, 401, "Unauthorized", {}, None)

        result = provider.get("weather_now", {"place": "Chicago"})
        assert result.status == "error"
        assert "Invalid API key" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_location_not_found(self, mock_urlopen, provider):
        """Test handling of unknown location."""
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(None, 404, "Not Found", {}, None)

        result = provider.get("weather_now", {"place": "Nowhere"})
        assert result.status == "error"
        assert "not found" in result.speech_text.lower()

    @patch("urllib.request.urlopen")
    def test_rate_limited(self, mock_urlopen, provider):
        """Test handling of rate limit."""
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(None, 429, "Too Many Requests", {}, None)

        result = provider.get("weather_now", {"place": "Chicago"})
        assert result.status == "error"
        assert "Rate limit" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_network_error(self, mock_urlopen, provider):
        """Test handling of network error."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        result = provider.get("weather_now", {"place": "Chicago"})
        assert result.status == "error"
        assert "Network error" in result.speech_text
