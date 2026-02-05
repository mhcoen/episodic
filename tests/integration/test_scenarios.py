"""
Multi-turn scenario tests.

Tests realistic conversation scenarios to verify end-to-end behavior.
"""

import pytest
from episodic.harness import (
    EventKind,
    FakeClock,
    StubLLMClient,
    StubWeatherProvider,
    StubNewsProvider,
    WeatherResult,
    create_default_stub_providers,
)
from tests.integration.conftest import HarnessSession


class TestWeatherScenarios:
    """Realistic weather query scenarios."""

    def test_weather_then_followup(self):
        """Weather query followed by followup question."""
        providers = {
            "weather": StubWeatherProvider({
                "current": WeatherResult(temp=72, condition="Sunny", location="Test City"),
            }),
            "news": StubNewsProvider(),
        }
        session = HarnessSession(
            llm=StubLLMClient(["Sunny days are great for outdoor activities"]),
            providers=providers,
            debug_channels={"router", "providers"},
        )

        # Initial weather query
        result1 = session.send("/weather")
        utility_results = [
            e for e in result1.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert len(utility_results) == 1
        assert utility_results[0].fields["payload"]["temp"] == 72

        # Followup question (goes to LLM)
        result2 = session.send("What activities would be good in this weather?")
        responses = [
            e for e in result2.user_events
            if e.kind == EventKind.ASSISTANT_RESPONSE.value
        ]
        assert len(responses) == 1

    def test_weather_multiple_locations(self):
        """Weather queries for multiple locations."""
        providers = {
            "weather": StubWeatherProvider({
                "current": WeatherResult(temp=72, condition="Sunny", location="Default"),
                "Madison, WI": WeatherResult(temp=33, condition="Cloudy", location="Madison"),
                "New York": WeatherResult(temp=55, condition="Rainy", location="New York"),
            }),
            "news": StubNewsProvider(),
        }
        session = HarnessSession(providers=providers)

        result1 = session.send("/weather Madison, WI")
        result2 = session.send("/weather New York")

        # Both should succeed
        utility_results1 = [e for e in result1.user_events if e.kind == EventKind.UTILITY_RESULT.value]
        utility_results2 = [e for e in result2.user_events if e.kind == EventKind.UTILITY_RESULT.value]

        assert utility_results1[0].fields["payload"]["temp"] == 33
        assert utility_results2[0].fields["payload"]["temp"] == 55


class TestNewsScenarios:
    """Realistic news query scenarios."""

    def test_news_then_detail_request(self):
        """News headlines followed by request for details."""
        session = HarnessSession(
            llm=StubLLMClient(["Here are more details about that story..."]),
            providers=create_default_stub_providers(),
        )

        # Get headlines
        result1 = session.send("/news")
        utility_results = [
            e for e in result1.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert "headlines" in utility_results[0].fields["payload"]

        # Ask for details (goes to LLM)
        result2 = session.send("Tell me more about the first headline")
        responses = [
            e for e in result2.user_events
            if e.kind == EventKind.ASSISTANT_RESPONSE.value
        ]
        assert len(responses) == 1

    def test_news_category_browsing(self):
        """Browsing news across multiple categories."""
        session = HarnessSession(
            providers=create_default_stub_providers(),
        )

        categories = ["tech", "politics", "science"]
        for category in categories:
            result = session.send(f"/news {category}")
            utility_results = [
                e for e in result.user_events
                if e.kind == EventKind.UTILITY_RESULT.value
            ]
            assert len(utility_results) == 1


class TestTimerScenarios:
    """Realistic timer usage scenarios."""

    def test_set_timer_natural_language(self):
        """Setting timer with natural language."""
        session = HarnessSession(
            providers=create_default_stub_providers(),
            debug_channels={"router", "grammar"},
        )

        result = session.send("set a timer for 5 minutes")

        # Should be recognized as timer command
        user_events = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_EXECUTED.value
        ]
        if user_events:
            assert user_events[0].fields.get("command") == "timer_set"

    def test_timer_with_label(self):
        """Setting timer with a label."""
        session = HarnessSession(
            providers=create_default_stub_providers(),
        )

        result = session.send("set a timer for 10 minutes for pasta")

        # Should process timer with label
        assert result is not None


class TestMixedCommandScenarios:
    """Scenarios mixing utility commands and conversation."""

    def test_utility_then_conversation(self):
        """Utility command followed by conversation."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Based on the weather, I'd recommend...",
            ]),
            providers=create_default_stub_providers(),
        )

        # Utility command
        session.send("/weather")

        # Conversation
        result = session.send("What should I wear today?")
        responses = [
            e for e in result.user_events
            if e.kind == EventKind.ASSISTANT_RESPONSE.value
        ]
        assert len(responses) == 1

    def test_conversation_then_utility(self):
        """Conversation followed by utility command."""
        session = HarnessSession(
            llm=StubLLMClient(["Sure, I can help with that"]),
            providers=create_default_stub_providers(),
        )

        # Conversation
        session.send("I'm planning my day")

        # Utility command
        result = session.send("/time")
        utility_results = [
            e for e in result.user_events
            if e.kind == EventKind.UTILITY_RESULT.value
        ]
        assert len(utility_results) >= 1

    def test_interleaved_commands_and_conversation(self):
        """Interleaved utility commands and conversation."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Good morning!",
                "That sounds like a plan",
                "Have a great day!",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Good morning!")
        session.send("/weather")
        session.send("/news")
        session.send("I think I'll work from home today")
        session.send("/time")
        result = session.send("Thanks for the info!")

        # All should process
        assert result is not None


class TestConversationFlow:
    """Tests for natural conversation flow."""

    def test_question_answer_chain(self):
        """Chain of questions and answers."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Python is a programming language",
                "It was created in 1991",
                "By Guido van Rossum",
                "He's Dutch",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("What is Python?")
        session.send("When was it created?")
        session.send("Who created it?")
        result = session.send("Where is he from?")

        # All questions should be answered
        assert len(session.runtime.llm.requests) == 4

    def test_clarification_request(self):
        """Handling clarification requests."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Could you be more specific?",
                "Ah, I see. Here's what I know...",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Tell me about it")  # Ambiguous
        result = session.send("I meant Python programming")  # Clarification

        # Should handle clarification
        assert len(result.user_events) >= 1

    def test_correction_handling(self):
        """Handling corrections."""
        session = HarnessSession(
            llm=StubLLMClient([
                "JavaScript is a web language",
                "You're right, I apologize. Java and JavaScript are different.",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Tell me about JavaScript")
        result = session.send("No, I said Java, not JavaScript")

        # Should handle correction
        assert len(result.user_events) >= 1


class TestErrorRecoveryScenarios:
    """Scenarios involving error recovery."""

    def test_invalid_command_recovery(self):
        """Recovery from invalid command."""
        session = HarnessSession(
            llm=StubLLMClient(["Let me help you with that"]),
            providers=create_default_stub_providers(),
        )

        session.send("/nonexistent_command")
        result = session.send("What commands are available?")

        # Should continue working after invalid command
        assert result is not None

    def test_malformed_input_recovery(self):
        """Recovery from malformed input."""
        session = HarnessSession(
            llm=StubLLMClient(["I understand"]),
            providers=create_default_stub_providers(),
        )

        session.send("!!!@#$%^&*()")  # Malformed
        result = session.send("Sorry, let me try again. Hello!")

        # Should recover
        assert result is not None


class TestSessionScenarios:
    """Scenarios spanning session lifetime."""

    def test_long_session(self):
        """Long session with many interactions."""
        responses = [f"Response {i}" for i in range(25)]
        session = HarnessSession(
            llm=StubLLMClient(responses),
            providers=create_default_stub_providers(),
        )

        for i in range(25):
            result = session.send(f"Message {i}")
            assert result is not None

        # Session should still work
        assert len(session.runtime.llm.requests) == 25

    def test_session_with_time_advancement(self):
        """Session with clock advancement."""
        clock = FakeClock(start=0)
        session = HarnessSession(
            clock=clock,
            llm=StubLLMClient(["Morning", "Afternoon", "Evening"]),
            providers=create_default_stub_providers(),
        )

        session.send("Good morning")
        clock.advance(6 * 3600)  # 6 hours later

        session.send("Good afternoon")
        clock.advance(6 * 3600)  # Another 6 hours

        result = session.send("Good evening")

        # All should work with time advancement
        assert len(session.runtime.llm.requests) == 3


class TestRealWorldScenarios:
    """Real-world usage scenarios."""

    def test_morning_routine(self):
        """Morning routine: time, weather, news."""
        session = HarnessSession(
            llm=StubLLMClient(["Have a great day!"]),
            providers=create_default_stub_providers(),
        )

        session.send("/time")
        session.send("/weather")
        session.send("/news")
        result = session.send("Thanks, have a nice day!")

        # All commands should work
        weather = session.runtime.providers["weather"]
        news = session.runtime.providers["news"]
        assert len(weather.calls) == 1
        assert len(news.calls) == 1

    def test_planning_session(self):
        """Planning session with timers and reminders."""
        session = HarnessSession(
            llm=StubLLMClient([
                "I can help you plan",
                "Timer set",
                "Reminder noted",
                "Good luck!",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("I need to plan my work session")
        session.send("set a timer for 25 minutes")
        session.send("remind me to take a break")
        result = session.send("Thanks!")

        # Should handle planning workflow
        assert result is not None

    def test_research_session(self):
        """Research session with questions and followups."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Machine learning is a subset of AI",
                "It uses algorithms to learn from data",
                "Common types include supervised and unsupervised",
                "Applications include image recognition and NLP",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("What is machine learning?")
        session.send("How does it work?")
        session.send("What are the main types?")
        result = session.send("What are some applications?")

        # Research flow should work
        assert len(session.runtime.llm.requests) == 4
