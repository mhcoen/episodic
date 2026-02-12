"""
Tests for Weather Provider.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import json

from episodic.utility.providers.weather import WeatherProvider, _get_condition, _normalize_owm_location
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


class TestOWMLocationNormalization:
    """Test OpenWeatherMap location normalization."""

    def test_us_state_appends_country(self):
        assert _normalize_owm_location("Madison, WI") == "Madison,WI,US"

    def test_us_state_no_space(self):
        assert _normalize_owm_location("Madison,WI") == "Madison,WI,US"

    def test_us_state_case_insensitive(self):
        assert _normalize_owm_location("austin, tx") == "austin,tx,US"

    def test_country_code_not_modified(self):
        # "UK" is not a US state code, so leave as-is
        assert _normalize_owm_location("London, UK") == "London, UK"

    def test_city_only_not_modified(self):
        assert _normalize_owm_location("Chicago") == "Chicago"

    def test_full_format_not_modified(self):
        # Already has country code
        assert _normalize_owm_location("Madison,WI,US") == "Madison,WI,US"

    def test_dc_recognized(self):
        assert _normalize_owm_location("Washington, DC") == "Washington,DC,US"


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

    @patch("urllib.request.urlopen")
    def test_no_location(self, mock_urlopen, provider):
        """Test error when no location available and IP geolocation fails."""
        from urllib.error import URLError
        # IP geolocation fails
        mock_urlopen.side_effect = URLError("Connection refused")

        # Clear all location sources
        provider._default_location = None
        provider._home_location = None
        provider._detected_location = None
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


class TestIPGeolocation:
    """Test IP geolocation fallback."""

    @pytest.fixture
    def provider(self):
        """Create provider without location configured."""
        p = WeatherProvider()
        p.configure({"api_key": "test_key"})
        return p

    @patch("urllib.request.urlopen")
    def test_ip_geolocation_with_region(self, mock_urlopen, provider):
        """Test IP geolocation returns city with region code."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "city": "Chicago",
            "region": "IL",
            "country": "United States"
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        location = provider._get_location_from_ip()
        assert location == "Chicago, IL"

    @patch("urllib.request.urlopen")
    def test_ip_geolocation_without_region(self, mock_urlopen, provider):
        """Test IP geolocation returns city only if no region."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "city": "London",
            "region": "",
            "country": "United Kingdom"
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        location = provider._get_location_from_ip()
        assert location == "London"

    @patch("urllib.request.urlopen")
    def test_ip_geolocation_failure_returns_none(self, mock_urlopen, provider):
        """Test graceful failure when IP geolocation fails."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        location = provider._get_location_from_ip()
        assert location is None

    @patch("urllib.request.urlopen")
    def test_resolve_location_uses_ip_fallback(self, mock_urlopen, provider):
        """Test that _resolve_location uses IP geolocation as fallback."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "city": "Seattle",
            "region": "WA",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # No location configured, should fall back to IP
        location = provider._resolve_location("current")
        assert location == "Seattle, WA"

    @patch("urllib.request.urlopen")
    def test_ip_location_cached_in_memory(self, mock_urlopen, provider):
        """Test that IP location is cached after first lookup."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "city": "Denver",
            "region": "CO",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # First call
        location1 = provider._resolve_location("current")
        assert location1 == "Denver, CO"
        assert mock_urlopen.call_count == 1

        # Second call should use cache
        location2 = provider._resolve_location("current")
        assert location2 == "Denver, CO"
        assert mock_urlopen.call_count == 1  # No additional call

    def test_detected_location_used_before_ip(self, provider):
        """Test that location_detected preference is used before IP lookup."""
        provider.configure({
            "api_key": "test_key",
            "location_detected": "Portland, OR"
        })

        location = provider._resolve_location("current")
        assert location == "Portland, OR"

    def test_default_location_takes_priority(self, provider):
        """Test that config default_location takes priority over detected."""
        provider.configure({
            "api_key": "test_key",
            "default_location": "New York, NY",
            "location_detected": "Portland, OR"
        })

        location = provider._resolve_location("current")
        assert location == "New York, NY"

    def test_home_takes_priority_over_detected(self, provider):
        """Test that location_home takes priority for 'home' place."""
        provider.configure({
            "api_key": "test_key",
            "location_home": "Boston, MA",
            "location_detected": "Portland, OR"
        })

        location = provider._resolve_location("home")
        assert location == "Boston, MA"

    @patch("urllib.request.urlopen")
    def test_get_new_ip_location(self, mock_urlopen, provider):
        """Test get_new_ip_location returns newly detected location."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "city": "Austin",
            "region": "TX",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Trigger IP lookup
        provider._resolve_location("current")

        # Should return the new location
        new_loc = provider.get_new_ip_location()
        assert new_loc == "Austin, TX"

    def test_get_new_ip_location_none_when_not_fetched(self, provider):
        """Test get_new_ip_location returns None when no IP lookup done."""
        new_loc = provider.get_new_ip_location()
        assert new_loc is None

    def test_get_new_ip_location_none_when_same_as_detected(self, provider):
        """Test get_new_ip_location returns None when same as detected."""
        provider.configure({
            "api_key": "test_key",
            "location_detected": "Austin, TX"
        })
        # Manually set cache to same value
        provider._ip_location_cache = "Austin, TX"

        new_loc = provider.get_new_ip_location()
        assert new_loc is None


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
