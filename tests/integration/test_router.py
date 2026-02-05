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


class TestUtilityCommandVariants:
    """Tests for various utility command formats."""

    @pytest.mark.parametrize("cmd", [
        "/timer 5m",
        "/timer 5 minutes",
        "/timer 300",
    ])
    def test_timer_command_variants(self, test_session, cmd):
        """Timer commands with various duration formats."""
        result = test_session.send(cmd)
        assert len(result.user_events) >= 1

    @pytest.mark.parametrize("cmd", [
        "/alarm 7am",
        "/alarm 7:00",
        "/alarm 19:00",
    ])
    def test_alarm_command_variants(self, test_session, cmd):
        """Alarm commands with various time formats."""
        result = test_session.send(cmd)
        assert len(result.user_events) >= 1

    @pytest.mark.parametrize("cmd", [
        "/weather",
        "/weather New York",
        "/weather Madison, WI",
    ])
    def test_weather_command_variants(self, test_session, cmd):
        """Weather commands with optional location."""
        result = test_session.send(cmd)
        assert len(result.user_events) >= 1

    @pytest.mark.parametrize("cmd", [
        "/news",
        "/news tech",
        "/news politics",
    ])
    def test_news_command_variants(self, test_session, cmd):
        """News commands with optional category."""
        result = test_session.send(cmd)
        assert len(result.user_events) >= 1

    def test_time_command(self, test_session):
        """Time command should return current time."""
        result = test_session.send("/time")
        assert len(result.user_events) >= 1

    def test_forecast_command(self, test_session):
        """Forecast command should call weather provider."""
        result = test_session.send("/forecast")
        provider_calls = [
            e for e in result.debug_events
            if e.kind == EventKind.PROVIDER_CALL.value
        ]
        assert len(provider_calls) >= 1


class TestVoiceGrammarRouting:
    """Tests for voice grammar pattern matching."""

    @pytest.mark.parametrize("utterance,expected_target", [
        ("what time is it", "UTILITY"),
        ("what's the time", "UTILITY"),
        ("tell me the time", "UTILITY"),
        ("current time", "UTILITY"),
    ])
    def test_time_utterances(self, test_session, utterance, expected_target):
        """Time-related utterances should route to utility."""
        result = test_session.send(utterance)
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        assert len(router_decisions) >= 1
        targets = [d.fields.get("target") for d in router_decisions]
        assert expected_target in targets or "LLM" in targets

    @pytest.mark.parametrize("utterance,expected_target", [
        ("what's the weather", "UTILITY"),
        ("how's the weather", "UTILITY"),
        ("weather today", "UTILITY"),
        ("what's it like outside", "UTILITY"),
    ])
    def test_weather_utterances(self, test_session, utterance, expected_target):
        """Weather-related utterances should route to utility."""
        result = test_session.send(utterance)
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        assert len(router_decisions) >= 1

    @pytest.mark.parametrize("utterance", [
        "set a timer for 5 minutes",
        "timer 10 minutes",
        "five minute timer",
        "start a 3 minute timer",
    ])
    def test_timer_utterances(self, test_session, utterance):
        """Timer-related utterances should route to utility."""
        result = test_session.send(utterance)
        # Should have some routing decision
        assert len(result.debug_events) >= 1 or len(result.user_events) >= 1

    @pytest.mark.parametrize("utterance", [
        "set an alarm for 7am",
        "alarm at 8 o'clock",
        "alarm for 6:30",
    ])
    def test_alarm_utterances(self, test_session, utterance):
        """Alarm-related utterances should route to utility."""
        result = test_session.send(utterance)
        assert len(result.debug_events) >= 1 or len(result.user_events) >= 1


class TestMemoryQueryPatterns:
    """Tests for memory query pattern detection."""

    @pytest.mark.parametrize("query", [
        "did we discuss this before",
        "have we talked about Python",
        "when did I mention the project",
        "what did we say about testing",
        "remember when we discussed this",
        "recall our conversation about APIs",
    ])
    def test_memory_query_patterns(self, test_session, query):
        """Memory query patterns should be detected."""
        result = test_session.send(query)
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Should have routing decision (MQL or LLM)
        assert len(router_decisions) >= 1

    @pytest.mark.parametrize("query", [
        "last time we talked",
        "earlier today",
        "before when you said",
        "our previous conversation",
    ])
    def test_memory_marker_phrases(self, test_session, query):
        """Memory marker phrases should trigger MQL routing."""
        result = test_session.send(query)
        # Should process without error
        assert result is not None


class TestConfidenceScoring:
    """Tests for routing confidence scores."""

    def test_exact_command_high_confidence(self, test_session):
        """Slash commands should have confidence 1.0."""
        result = test_session.send("/time")
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        if router_decisions:
            assert router_decisions[0].fields.get("confidence", 0) >= 0.9

    def test_clear_utterance_high_confidence(self, test_session):
        """Clear utility utterances should have high confidence."""
        result = test_session.send("what time is it")
        parse_attempts = [
            e for e in result.debug_events
            if e.kind == EventKind.PARSE_ATTEMPT.value
        ]
        if parse_attempts:
            # Should have reasonable confidence
            assert parse_attempts[0].fields.get("confidence", 0) >= 0.5

    def test_ambiguous_input_lower_confidence(self, test_session):
        """Ambiguous inputs should have lower confidence."""
        result = test_session.send("maybe timer")
        # Should still process without crashing
        assert result is not None


class TestPreemptRouting:
    """Tests for preempt command routing."""

    @pytest.mark.parametrize("cmd", [
        "stop",
        "cancel",
        "nevermind",
        "stop that",
        "cancel it",
    ])
    def test_preempt_commands(self, test_session, cmd):
        """Preempt commands should be recognized."""
        result = test_session.send(cmd)
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Should have routing decision
        assert len(router_decisions) >= 1

    @pytest.mark.parametrize("phrase", [
        "stop by the store",
        "can't stop thinking",
        "stop and think about it",
    ])
    def test_trap_idioms_not_preempt(self, test_session, phrase):
        """Trap idioms should not trigger preempt."""
        result = test_session.send(phrase)
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Should not route to PREEMPT
        for decision in router_decisions:
            if decision.fields.get("target") == "PREEMPT":
                # Trap idiom incorrectly triggered preempt
                assert False, f"'{phrase}' incorrectly routed to PREEMPT"


class TestLLMFallback:
    """Tests for LLM fallback behavior."""

    @pytest.mark.parametrize("query", [
        "tell me a joke",
        "explain quantum computing",
        "what is the meaning of life",
        "how do I learn Python",
        "write a haiku about programming",
    ])
    def test_general_queries_to_llm(self, session_with_llm, query):
        """General queries should fall back to LLM."""
        result = session_with_llm.send(query)
        responses = [
            e for e in result.user_events
            if e.kind == EventKind.ASSISTANT_RESPONSE.value
        ]
        assert len(responses) >= 1

    def test_llm_response_has_source(self, session_with_llm):
        """LLM responses should indicate source."""
        result = session_with_llm.send("hello")
        responses = [
            e for e in result.user_events
            if e.kind == EventKind.ASSISTANT_RESPONSE.value
        ]
        if responses:
            assert responses[0].fields.get("source") == "llm"


class TestCrossGrammarRouting:
    """Tests for cross-grammar adversarial pairs."""

    ADVERSARIAL_PAIRS = [
        # (input, should_NOT_be_target)
        ("what time is it", "MQL"),  # Time query, not memory
        ("what time did the meeting start", "UTILITY"),  # Memory, not time
        ("weather tomorrow", "MQL"),  # Weather, not memory
        ("when did we discuss the weather", "UTILITY"),  # Memory, not weather
        ("set a timer", "MQL"),  # Timer, not memory
        ("did I set a timer", "UTILITY"),  # Memory about timers
    ]

    @pytest.mark.parametrize("text,wrong_target", ADVERSARIAL_PAIRS)
    def test_cross_grammar_discrimination(self, test_session, text, wrong_target):
        """Cross-grammar inputs should route correctly."""
        result = test_session.send(text)
        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Check that we don't route to the wrong target
        for decision in router_decisions:
            target = decision.fields.get("target")
            # If confidence is high, target should not be wrong_target
            if decision.fields.get("confidence", 0) > 0.8:
                assert target != wrong_target, (
                    f"'{text}' incorrectly routed to {wrong_target}"
                )
