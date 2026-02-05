"""
Provider integration tests.

Tests that providers are correctly called and return expected results.
"""

import pytest
from episodic.harness import EventKind


class TestWeatherProvider:
    """Tests for weather provider integration."""

    def test_weather_command_calls_provider(self, session_with_providers):
        """Weather command should call the weather provider."""
        result = session_with_providers.send("/weather")

        # Check provider was called
        weather_provider = session_with_providers.runtime.providers["weather"]
        assert len(weather_provider.calls) == 1
        assert weather_provider.calls[0][0] == "weather_now"

    def test_weather_command_returns_result(self, session_with_providers):
        """Weather command should return provider result."""
        result = session_with_providers.send("/weather")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert len(utility_results) == 1
        assert utility_results[0].fields["status"] == "ok"
        assert "temp" in utility_results[0].fields["payload"]

    def test_weather_with_location(self, session_with_providers):
        """Weather command should pass location to provider."""
        result = session_with_providers.send("/weather Madison, WI")

        weather_provider = session_with_providers.runtime.providers["weather"]
        assert weather_provider.calls[0][1]["place"] == "Madison, WI"

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert utility_results[0].fields["payload"]["temp"] == 33

    def test_forecast_command(self, session_with_providers):
        """Forecast command should call weather provider with forecast method."""
        result = session_with_providers.send("/forecast")

        weather_provider = session_with_providers.runtime.providers["weather"]
        assert len(weather_provider.calls) == 1
        assert weather_provider.calls[0][0] == "weather_forecast"

    def test_provider_debug_event(self, session_with_providers):
        """Provider call should emit debug event."""
        result = session_with_providers.send("/weather")

        provider_calls = [
            e for e in result.debug_events
            if e.kind == EventKind.PROVIDER_CALL.value
        ]
        assert len(provider_calls) == 1
        assert provider_calls[0].fields["provider"] == "weather"
        assert provider_calls[0].fields["has_provider"] is True


class TestNewsProvider:
    """Tests for news provider integration."""

    def test_news_command_calls_provider(self, session_with_providers):
        """News command should call the news provider."""
        result = session_with_providers.send("/news")

        news_provider = session_with_providers.runtime.providers["news"]
        assert len(news_provider.calls) == 1
        assert news_provider.calls[0][0] == "news_headlines"

    def test_news_with_category(self, session_with_providers):
        """News command should pass category to provider."""
        result = session_with_providers.send("/news tech")

        news_provider = session_with_providers.runtime.providers["news"]
        assert news_provider.calls[0][1]["category"] == "tech"

    def test_news_returns_headlines(self, session_with_providers):
        """News command should return headlines from provider."""
        result = session_with_providers.send("/news")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert len(utility_results) == 1
        assert "headlines" in utility_results[0].fields["payload"]
        assert len(utility_results[0].fields["payload"]["headlines"]) > 0


class TestProviderNotConfigured:
    """Tests for commands when provider is not configured."""

    def test_timer_without_provider(self, test_session):
        """Timer command without provider should still return result."""
        result = test_session.send("/timer 5m")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert len(utility_results) == 1
        # Should be a placeholder result
        assert "command" in utility_results[0].fields


class TestProviderCallTracking:
    """Tests for tracking provider calls."""

    def test_multiple_calls_tracked(self, session_with_providers):
        """Multiple provider calls should all be tracked."""
        session_with_providers.send("/weather")
        session_with_providers.send("/weather New York")
        session_with_providers.send("/news")

        weather_provider = session_with_providers.runtime.providers["weather"]
        news_provider = session_with_providers.runtime.providers["news"]

        assert len(weather_provider.calls) == 2
        assert len(news_provider.calls) == 1

    def test_provider_call_args_recorded(self, session_with_providers):
        """Provider call arguments should be recorded correctly."""
        session_with_providers.send("/weather Madison, WI")

        weather_provider = session_with_providers.runtime.providers["weather"]
        _, args = weather_provider.calls[0]
        assert args["place"] == "Madison, WI"


class TestProviderPayload:
    """Tests for provider payload handling."""

    def test_weather_payload_structure(self, session_with_providers):
        """Weather payload should have expected structure."""
        result = session_with_providers.send("/weather")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        payload = utility_results[0].fields["payload"]

        assert "temp" in payload
        assert "condition" in payload
        assert "high" in payload
        assert "low" in payload
        assert "location" in payload

    def test_weather_display_text(self, session_with_providers):
        """Weather result should include display text."""
        result = session_with_providers.send("/weather")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        display = utility_results[0].fields["display"]

        assert "72" in display  # Temperature
        assert "Test City" in display  # Location

    def test_weather_speech_text(self, session_with_providers):
        """Weather result should include speech text."""
        result = session_with_providers.send("/weather")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        speech = utility_results[0].fields["speech"]

        assert "72 degrees" in speech
        assert "Test City" in speech


class TestWeatherLocationVariants:
    """Tests for weather with various location formats."""

    @pytest.mark.parametrize("location", [
        "Madison, WI",
        "New York",
        "current",
    ])
    def test_weather_locations(self, session_with_providers, location):
        """Weather should work with various location formats."""
        result = session_with_providers.send(f"/weather {location}")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert len(utility_results) == 1
        assert utility_results[0].fields["status"] == "ok"

    def test_weather_default_location(self, session_with_providers):
        """Weather without location should use default."""
        result = session_with_providers.send("/weather")

        weather_provider = session_with_providers.runtime.providers["weather"]
        _, args = weather_provider.calls[0]
        assert args["place"] == "current"


class TestNewsCategories:
    """Tests for news with various categories."""

    @pytest.mark.parametrize("category", [
        "general",
        "tech",
        "politics",
        "business",
        "science",
    ])
    def test_news_categories(self, session_with_providers, category):
        """News should work with various categories."""
        result = session_with_providers.send(f"/news {category}")

        news_provider = session_with_providers.runtime.providers["news"]
        _, args = news_provider.calls[-1]
        assert args["category"] == category

    def test_news_default_category(self, session_with_providers):
        """News without category should use general."""
        result = session_with_providers.send("/news")

        news_provider = session_with_providers.runtime.providers["news"]
        _, args = news_provider.calls[0]
        assert args["category"] == "general"

    def test_news_headline_count(self, session_with_providers):
        """News should return requested headline count."""
        result = session_with_providers.send("/news")

        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        headlines = utility_results[0].fields["payload"]["headlines"]
        assert len(headlines) >= 1
        assert len(headlines) <= 5


class TestProviderStatus:
    """Tests for provider status reporting."""

    def test_weather_provider_status(self, session_with_providers):
        """Weather provider should report status."""
        weather_provider = session_with_providers.runtime.providers["weather"]
        status = weather_provider.status()

        assert "name" in status
        assert status["name"] == "weather"
        assert "call_count" in status

    def test_news_provider_status(self, session_with_providers):
        """News provider should report status."""
        news_provider = session_with_providers.runtime.providers["news"]
        status = news_provider.status()

        assert "name" in status
        assert status["name"] == "news"

    def test_provider_call_count_in_status(self, session_with_providers):
        """Provider status should reflect call count."""
        weather_provider = session_with_providers.runtime.providers["weather"]

        status1 = weather_provider.status()
        assert status1["call_count"] == 0

        session_with_providers.send("/weather")
        session_with_providers.send("/weather")

        status2 = weather_provider.status()
        assert status2["call_count"] == 2


class TestProviderRefresh:
    """Tests for provider refresh behavior."""

    def test_weather_provider_refresh(self, session_with_providers):
        """Weather provider should support refresh."""
        weather_provider = session_with_providers.runtime.providers["weather"]

        result = weather_provider.refresh({})

        assert result.success is True
        assert result.next_refresh_s > 0

    def test_news_provider_refresh(self, session_with_providers):
        """News provider should support refresh."""
        news_provider = session_with_providers.runtime.providers["news"]

        result = news_provider.refresh({})

        assert result.success is True
        assert result.next_refresh_s > 0


class TestTimerProvider:
    """Tests for timer provider integration."""

    def test_timer_set(self, session_with_providers):
        """Timer provider should track set timers."""
        timer_provider = session_with_providers.runtime.providers.get("timer")
        if timer_provider is None:
            pytest.skip("Timer provider not configured")

        timer = timer_provider.set_timer(300, label="test")

        assert timer["id"] is not None
        assert timer["duration_s"] == 300
        assert timer["label"] == "test"
        assert timer["status"] == "running"

    def test_timer_cancel(self, session_with_providers):
        """Timer provider should allow canceling timers."""
        timer_provider = session_with_providers.runtime.providers.get("timer")
        if timer_provider is None:
            pytest.skip("Timer provider not configured")

        timer_provider.set_timer(300)
        result = timer_provider.cancel_timer()

        assert result is True

    def test_timer_list(self, session_with_providers):
        """Timer provider should list active timers."""
        timer_provider = session_with_providers.runtime.providers.get("timer")
        if timer_provider is None:
            pytest.skip("Timer provider not configured")

        timer_provider.set_timer(300, label="timer1")
        timer_provider.set_timer(600, label="timer2")

        active = timer_provider.list_timers()
        assert len(active) == 2


class TestProviderIsolation:
    """Tests for provider isolation between sessions."""

    def test_providers_isolated_between_sessions(self):
        """Each session should have its own providers."""
        from tests.integration.conftest import HarnessSession
        from episodic.harness import create_default_stub_providers

        providers1 = create_default_stub_providers()
        providers2 = create_default_stub_providers()

        session1 = HarnessSession(providers=providers1)
        session2 = HarnessSession(providers=providers2)

        session1.send("/weather")

        assert len(providers1["weather"].calls) == 1
        assert len(providers2["weather"].calls) == 0

    def test_provider_state_not_shared(self):
        """Provider state should not be shared."""
        from tests.integration.conftest import HarnessSession
        from episodic.harness import create_default_stub_providers

        providers1 = create_default_stub_providers()
        session1 = HarnessSession(providers=providers1)

        session1.send("/weather Madison, WI")
        session1.send("/weather New York")

        # New session with new providers
        providers2 = create_default_stub_providers()
        session2 = HarnessSession(providers=providers2)

        # Should start fresh
        assert len(providers2["weather"].calls) == 0
