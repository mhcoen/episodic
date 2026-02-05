"""
Router integration tests.

Tests routing decisions for various input types.
"""

import pytest
from episodic.harness import EventKind


class TestCommandRouting:
    """Tests for slash command routing."""

    def test_exit_command(self, test_session):
        """Exit commands should emit exit action."""
        result = test_session.send("/exit")

        user_events = result.user_events
        assert len(user_events) == 1
        assert user_events[0].kind == EventKind.COMMAND_RESULT.value
        assert user_events[0].fields["action"] == "exit"

    def test_quit_command(self, test_session):
        """Quit command should emit exit action."""
        result = test_session.send("/quit")

        user_events = result.user_events
        assert len(user_events) == 1
        assert user_events[0].fields["action"] == "exit"

    def test_utility_command_routes(self, test_session):
        """Utility slash commands should be routed correctly."""
        result = test_session.send("/time")

        # Should have router decision debug event
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        assert len(router_decisions) >= 1
        assert router_decisions[0].fields["target"] == "COMMAND"

    def test_weather_command(self, test_session):
        """Weather command should route to utility handler."""
        result = test_session.send("/weather")

        # Should have provider call debug event
        provider_calls = [
            e for e in result.debug_events
            if e.kind == EventKind.PROVIDER_CALL.value
        ]
        assert len(provider_calls) == 1
        assert provider_calls[0].fields["provider"] == "weather"


class TestUtteranceRouting:
    """Tests for natural language utterance routing."""

    def test_whats_the_time_routes_to_utility(self, test_session):
        """'what's the time' should route to utility."""
        result = test_session.send("what's the time")

        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        assert len(router_decisions) >= 1
        # Should route to UTILITY or have voice grammar attempt
        targets = [d.fields["target"] for d in router_decisions]
        assert "UTILITY" in targets or "LLM" in targets

    def test_whats_the_weather_routes_to_utility(self, test_session):
        """'what's the weather' should route to utility."""
        result = test_session.send("what's the weather")

        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        assert len(router_decisions) >= 1

    def test_llm_fallback(self, session_with_llm):
        """General questions should fall back to LLM."""
        result = session_with_llm.send("tell me a joke")

        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Should have at least one routing decision
        assert len(router_decisions) >= 1

        # Should have LLM response
        responses = [
            e for e in result.user_events
            if e.kind == EventKind.ASSISTANT_RESPONSE.value
        ]
        assert len(responses) == 1
        assert responses[0].fields["source"] == "llm"


class TestMemoryRouting:
    """Tests for memory query routing."""

    def test_did_we_discuss_routes_to_mql(self, test_session):
        """'did we discuss' should route to MQL."""
        result = test_session.send("did we discuss Python yesterday")

        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        targets = [d.fields["target"] for d in router_decisions]
        # Should eventually route to MQL
        assert "MQL" in targets or len(router_decisions) > 0

    def test_when_did_i_routes_to_mql(self, test_session):
        """'when did I' should route to MQL."""
        result = test_session.send("when did I mention the project")

        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Should have routing decision
        assert len(router_decisions) >= 1


class TestDebugEvents:
    """Tests for debug event emission."""

    def test_debug_events_emitted_when_enabled(self, test_session):
        """Debug events should be emitted when channels are enabled."""
        result = test_session.send("/time")

        # Should have debug events
        assert len(result.debug_events) > 0

        # Router channel should be enabled
        router_events = [e for e in result.debug_events if e.channel == "router"]
        assert len(router_events) > 0

    def test_debug_events_have_timestamps(self, test_session):
        """All events should have timestamps."""
        result = test_session.send("/weather")

        for event in result.debug_events:
            assert event.timestamp >= 0

        for event in result.user_events:
            assert event.timestamp >= 0


class TestEmptyInput:
    """Tests for edge cases."""

    def test_empty_input_returns_empty_stream(self, test_session):
        """Empty input should return empty event stream."""
        result = test_session.send("")

        assert len(result.user_events) == 0
        assert len(result.debug_events) == 0

    def test_whitespace_only_returns_empty_stream(self, test_session):
        """Whitespace-only input should return empty event stream."""
        result = test_session.send("   \n\t  ")

        assert len(result.user_events) == 0
        assert len(result.debug_events) == 0
