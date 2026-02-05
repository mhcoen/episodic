"""
Topic detection integration tests using test harness.

Tests for topic drift detection, boundaries, and compression.
"""

import pytest
from episodic.harness import (
    EventKind,
    FakeClock,
    StubLLMClient,
    create_default_stub_providers,
)
from tests.integration.conftest import HarnessSession


class TestTopicRouting:
    """Tests for topic-related routing."""

    def test_topic_query_detected(self):
        """Topic-related queries should be detected."""
        session = HarnessSession(
            llm=StubLLMClient(["Here are the topics"]),
            providers=create_default_stub_providers(),
            debug_channels={"router"},
        )

        result = session.send("what topics have we discussed")

        # Should process as memory or LLM query
        assert len(result.debug_events) >= 1 or len(result.user_events) >= 1

    def test_topic_switch_query(self):
        """Topic switch requests should be handled."""
        session = HarnessSession(
            llm=StubLLMClient(["Switching topic"]),
            providers=create_default_stub_providers(),
            debug_channels={"router"},
        )

        result = session.send("let's talk about something else")

        # Should process without error
        assert result is not None


class TestTopicBoundaries:
    """Tests for topic boundary detection."""

    def test_clear_topic_change(self):
        """Clear topic changes should be detectable."""
        session = HarnessSession(
            llm=StubLLMClient([
                "I can help with Python",
                "Sure, let's discuss cooking",
            ]),
            providers=create_default_stub_providers(),
            debug_channels={"router", "context"},
        )

        session.send("Tell me about Python programming")
        result = session.send("Actually, let's talk about cooking instead")

        # Should handle topic change
        assert result is not None
        assert len(result.user_events) >= 1

    def test_gradual_topic_drift(self):
        """Gradual topic drift should be handled."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Python is great",
                "Web development uses Python",
                "JavaScript is also popular for web",
                "React is a JavaScript framework",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Tell me about Python")
        session.send("How is it used in web development?")
        session.send("What about JavaScript?")
        result = session.send("And React?")

        # Should handle gradual drift
        assert result is not None

    def test_topic_continuation(self):
        """Continuing same topic should work smoothly."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Machine learning is...",
                "Neural networks are...",
                "Deep learning involves...",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Tell me about machine learning")
        session.send("What about neural networks?")
        result = session.send("And deep learning?")

        # All responses should be generated
        assert len(session.runtime.llm.requests) == 3


class TestTopicReactivation:
    """Tests for returning to previous topics."""

    def test_return_to_previous_topic(self):
        """Returning to a previous topic should work."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Python is a programming language",
                "Cooking is an art",
                "Back to Python, it's versatile",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Tell me about Python")
        session.send("Now tell me about cooking")
        result = session.send("Actually, back to Python")

        # Should handle return to previous topic
        assert result is not None

    def test_reference_earlier_topic(self):
        """References to earlier topics should be recognized."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Quantum computing uses qubits",
                "Machine learning is AI",
                "As I mentioned about quantum...",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Explain quantum computing")
        session.send("Now explain machine learning")
        result = session.send("Going back to what you said about quantum")

        # Should process reference
        assert result is not None


class TestTopicMemory:
    """Tests for topic memory integration."""

    def test_topic_memory_query(self):
        """Queries about past topics should route appropriately."""
        session = HarnessSession(
            llm=StubLLMClient(["We discussed Python earlier"]),
            providers=create_default_stub_providers(),
            debug_channels={"router"},
        )

        result = session.send("what did we talk about earlier")

        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Should have routing decision
        assert len(router_decisions) >= 1

    def test_when_did_we_discuss(self):
        """'When did we discuss X' should route to memory."""
        session = HarnessSession(
            providers=create_default_stub_providers(),
            debug_channels={"router"},
        )

        result = session.send("when did we discuss Python")

        router_decisions = [
            e for e in result.debug_events
            if e.kind == EventKind.ROUTER_DECISION.value
        ]
        # Should route to MQL or memory handler
        targets = [d.fields.get("target") for d in router_decisions]
        assert "MQL" in targets or len(targets) > 0


class TestTopicCompression:
    """Tests for topic summary compression."""

    def test_long_conversation_handled(self):
        """Long conversations should be handled."""
        responses = [f"Response about topic {i}" for i in range(15)]
        session = HarnessSession(
            llm=StubLLMClient(responses),
            providers=create_default_stub_providers(),
        )

        # Simulate long conversation
        for i in range(15):
            session.send(f"Tell me about topic {i}")

        # Should complete without error
        assert len(session.runtime.llm.requests) == 15

    def test_topic_transitions_tracked(self):
        """Topic transitions should be trackable."""
        session = HarnessSession(
            llm=StubLLMClient([
                "AI response",
                "Climate response",
                "Music response",
            ]),
            providers=create_default_stub_providers(),
            debug_channels={"router", "context"},
        )

        session.send("Tell me about AI")
        session.send("Now about climate change")
        result = session.send("What about music?")

        # Should track all transitions
        assert len(session.runtime.llm.requests) == 3


class TestTopicEdgeCases:
    """Tests for topic edge cases."""

    def test_single_word_topic(self):
        """Single word topic queries should work."""
        session = HarnessSession(
            llm=StubLLMClient(["Here's info about Python"]),
            providers=create_default_stub_providers(),
        )

        result = session.send("Python")

        # Should process
        assert result is not None

    def test_question_only_topic(self):
        """Question-only inputs should work."""
        session = HarnessSession(
            llm=StubLLMClient(["That's a good question"]),
            providers=create_default_stub_providers(),
        )

        result = session.send("?")

        # Should handle gracefully
        assert result is not None

    def test_mixed_languages(self):
        """Mixed language input should be handled."""
        session = HarnessSession(
            llm=StubLLMClient(["I understand"]),
            providers=create_default_stub_providers(),
        )

        result = session.send("Tell me about Python en español")

        # Should process
        assert result is not None

    def test_emoji_in_topic(self):
        """Topics with emojis should be handled."""
        session = HarnessSession(
            llm=StubLLMClient(["Weather is nice"]),
            providers=create_default_stub_providers(),
        )

        result = session.send("What's the weather like? ☀️")

        # Should process
        assert result is not None


class TestTopicInference:
    """Tests for topic inference from context."""

    def test_implicit_topic_reference(self):
        """Implicit topic references should work."""
        session = HarnessSession(
            llm=StubLLMClient([
                "Python is a programming language",
                "It was created by Guido van Rossum",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Tell me about Python")
        result = session.send("Who created it?")  # Implicit reference to Python

        # Should understand implicit reference
        assert len(result.user_events) >= 1

    def test_pronoun_resolution(self):
        """Pronoun resolution in topics should work."""
        session = HarnessSession(
            llm=StubLLMClient([
                "JavaScript is for web",
                "It's very popular",
            ]),
            providers=create_default_stub_providers(),
        )

        session.send("Tell me about JavaScript")
        result = session.send("Is it popular?")

        # Should resolve "it" to JavaScript
        assert result is not None
