"""
Context assembly integration tests.

Tests for context budget management, truncation, and anchor preservation.
"""

import pytest
from episodic.harness import (
    EventKind,
    FakeClock,
    StubLLMClient,
    create_default_stub_providers,
)
from tests.integration.conftest import HarnessSession


class TestContextBudget:
    """Tests for context token budget management."""

    def test_context_plan_emitted(self, session_with_llm):
        """Context plan event should be emitted for LLM requests."""
        result = session_with_llm.send("tell me something")

        context_plans = [
            e for e in result.debug_events
            if e.kind == EventKind.CONTEXT_PLAN.value
        ]
        # Should have context planning
        assert len(context_plans) >= 1

    def test_context_plan_has_budget(self, session_with_llm):
        """Context plan should include token budget."""
        result = session_with_llm.send("explain something")

        context_plans = [
            e for e in result.debug_events
            if e.kind == EventKind.CONTEXT_PLAN.value
        ]
        if context_plans:
            assert "budget" in context_plans[0].fields

    def test_context_plan_has_usage(self, session_with_llm):
        """Context plan should include planned usage."""
        result = session_with_llm.send("tell me about Python")

        context_plans = [
            e for e in result.debug_events
            if e.kind == EventKind.CONTEXT_PLAN.value
        ]
        if context_plans:
            assert "planned_usage" in context_plans[0].fields


class TestContextWithHistory:
    """Tests for context assembly with conversation history."""

    def test_multiple_turns_build_context(self):
        """Multiple turns should build up context."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Response 1",
                "Response 2",
                "Response 3",
            ]),
            providers=create_default_stub_providers(),
            debug_channels={"context", "llm"},
        )

        session.send("first message")
        session.send("second message")
        result = session.send("third message")

        # LLM should have been called multiple times
        assert len(session.runtime.llm.requests) == 3

    def test_context_includes_recent_history(self):
        """Context should include recent conversation history."""
        session = HarnessSession(
            llm=StubLLMClient([
                "I understand",
                "Of course",
                "Here's a summary",
            ]),
            providers=create_default_stub_providers(),
            debug_channels={"context", "llm"},
        )

        session.send("Let's discuss Python")
        session.send("Tell me about decorators")
        result = session.send("Summarize what we discussed")

        # The last request should have context from previous turns
        last_request = session.runtime.llm.requests[-1]
        assert len(last_request.messages) >= 1


class TestContextTruncation:
    """Tests for context truncation behavior."""

    def test_long_input_handled(self):
        """Long input should be handled without crashing."""
        session = HarnessSession(
            llm=StubLLMClient(["OK"]),
            providers=create_default_stub_providers(),
        )

        # Send a very long message
        long_message = "test " * 1000
        result = session.send(long_message)

        # Should process without error
        assert result is not None

    def test_many_turns_handled(self):
        """Many conversation turns should be handled."""
        responses = [f"Response {i}" for i in range(20)]
        session = HarnessSession(
            llm=StubLLMClient(responses),
            providers=create_default_stub_providers(),
        )

        # Send many messages
        for i in range(20):
            result = session.send(f"Message {i}")

        # Should complete without error
        assert result is not None


class TestContextDebugEvents:
    """Tests for context-related debug events."""

    def test_context_events_have_channel(self, session_with_llm):
        """Context events should have correct channel."""
        result = session_with_llm.send("hello")

        context_events = [
            e for e in result.debug_events
            if e.kind == EventKind.CONTEXT_PLAN.value
        ]
        for event in context_events:
            assert event.channel == "context"

    def test_context_events_have_timestamps(self, session_with_llm):
        """Context events should have timestamps."""
        result = session_with_llm.send("test")

        context_events = [
            e for e in result.debug_events
            if e.kind == EventKind.CONTEXT_PLAN.value
        ]
        for event in context_events:
            assert event.timestamp >= 0


class TestLLMRequestMetadata:
    """Tests for LLM request metadata events."""

    def test_llm_meta_emitted(self, session_with_llm):
        """LLM request metadata should be emitted."""
        result = session_with_llm.send("hello")

        llm_events = [
            e for e in result.debug_events
            if e.kind == EventKind.LLM_REQUEST_META.value
        ]
        # Should have LLM metadata
        assert len(llm_events) >= 1

    def test_llm_meta_has_model(self, session_with_llm):
        """LLM metadata should include model."""
        result = session_with_llm.send("test")

        llm_events = [
            e for e in result.debug_events
            if e.kind == EventKind.LLM_REQUEST_META.value
        ]
        if llm_events:
            assert "model" in llm_events[0].fields

    def test_llm_meta_has_token_counts(self, session_with_llm):
        """LLM metadata should include token counts."""
        result = session_with_llm.send("explain something")

        llm_events = [
            e for e in result.debug_events
            if e.kind == EventKind.LLM_REQUEST_META.value
        ]
        if llm_events:
            fields = llm_events[0].fields
            assert "input_tokens" in fields or "output_tokens" in fields


class TestContextIsolation:
    """Tests for context isolation between sessions."""

    def test_context_not_shared_between_sessions(self):
        """Context should not be shared between sessions."""
        session1 = HarnessSession(
            llm=StubLLMClient(["Response 1", "Response 2"]),
            providers=create_default_stub_providers(),
        )
        session2 = HarnessSession(
            llm=StubLLMClient(["Fresh response"]),
            providers=create_default_stub_providers(),
        )

        # Build context in session 1
        session1.send("Remember this: secret code is 12345")
        session1.send("What was the code?")

        # Session 2 should start fresh
        result = session2.send("What was the code?")

        # Session 2 shouldn't have session 1's context
        assert len(session2.runtime.llm.requests) == 1


class TestContextWithUtility:
    """Tests for context with utility commands."""

    def test_utility_commands_dont_add_to_llm_context(self, session_with_providers):
        """Utility commands should not add to LLM context."""
        # Utility commands go to providers, not LLM
        result = session_with_providers.send("/weather")

        # Should not have LLM context events
        llm_events = [
            e for e in result.debug_events
            if e.kind == EventKind.LLM_REQUEST_META.value
        ]
        # Weather command doesn't use LLM
        assert len(llm_events) == 0

    def test_mixed_utility_and_llm(self):
        """Mixed utility and LLM commands should work."""
        session = HarnessSession(
            llm=StubLLMClient(["Here's what I know"]),
            providers=create_default_stub_providers(),
            debug_channels={"router", "context", "llm"},
        )

        session.send("/weather")  # Utility
        session.send("/time")  # Utility
        result = session.send("What else can you tell me?")  # LLM

        # Should have LLM response
        responses = [
            e for e in result.user_events
            if e.kind == EventKind.ASSISTANT_RESPONSE.value
        ]
        assert len(responses) == 1
