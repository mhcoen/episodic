"""
Determinism tests for replay consistency.

These tests verify that:
1. Same inputs with same seed produce identical outputs
2. Event streams are reproducible
3. Random choices are deterministic when seeded
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


def events_equal(events1, events2, ignore_timestamps=True):
    """Compare two event lists for equality."""
    if len(events1) != len(events2):
        return False

    for e1, e2 in zip(events1, events2):
        if e1.kind != e2.kind:
            return False
        if e1.level != e2.level:
            return False
        if not ignore_timestamps and e1.timestamp != e2.timestamp:
            return False
        # Compare fields (ignore timestamp-related fields)
        f1 = {k: v for k, v in e1.fields.items() if "timestamp" not in k.lower()}
        f2 = {k: v for k, v in e2.fields.items() if "timestamp" not in k.lower()}
        if f1 != f2:
            return False
    return True


class TestIdenticalReplay:
    """Tests that identical inputs produce identical outputs."""

    def test_command_replay_identical(self):
        """Same command with same setup produces identical events."""
        def run_session():
            session = HarnessSession(
                rng_seed=42,
                providers=create_default_stub_providers(),
                debug_channels={"router", "grammar", "providers"},
            )
            return session.send("/weather")

        result1 = run_session()
        result2 = run_session()

        assert events_equal(result1.debug_events, result2.debug_events)
        assert events_equal(result1.user_events, result2.user_events)

    def test_utterance_replay_identical(self):
        """Same utterance with same setup produces identical events."""
        def run_session():
            session = HarnessSession(
                rng_seed=42,
                providers=create_default_stub_providers(),
                debug_channels={"router", "grammar"},
            )
            return session.send("what time is it")

        result1 = run_session()
        result2 = run_session()

        assert events_equal(result1.debug_events, result2.debug_events)

    def test_multi_turn_replay_identical(self):
        """Multi-turn conversation produces identical events on replay."""
        def run_session():
            session = HarnessSession(
                rng_seed=42,
                llm=StubLLMClient([
                    "Response 1",
                    "Response 2",
                    "Response 3",
                ]),
                providers=create_default_stub_providers(),
                debug_channels={"router", "grammar", "llm"},
            )
            events = []
            events.append(session.send("hello"))
            events.append(session.send("tell me more"))
            events.append(session.send("thanks"))
            return events

        results1 = run_session()
        results2 = run_session()

        for r1, r2 in zip(results1, results2):
            assert events_equal(r1.debug_events, r2.debug_events)


class TestSeededRNG:
    """Tests that seeded RNG produces deterministic behavior."""

    def test_different_seeds_different_results(self):
        """Different seeds should produce different random choices."""
        session1 = HarnessSession(rng_seed=42)
        session2 = HarnessSession(rng_seed=123)

        # Get random values from each session's RNG
        val1 = session1.runtime.rng.random()
        val2 = session2.runtime.rng.random()

        assert val1 != val2

    def test_same_seed_same_sequence(self):
        """Same seed should produce same random sequence."""
        session1 = HarnessSession(rng_seed=42)
        session2 = HarnessSession(rng_seed=42)

        # Generate sequence from each
        seq1 = [session1.runtime.rng.random() for _ in range(10)]
        seq2 = [session2.runtime.rng.random() for _ in range(10)]

        assert seq1 == seq2


class TestClockDeterminism:
    """Tests that FakeClock produces deterministic timestamps."""

    def test_fake_clock_deterministic(self):
        """FakeClock advances deterministically."""
        clock1 = FakeClock(start=1000.0)
        clock2 = FakeClock(start=1000.0)

        clock1.advance(100)
        clock2.advance(100)

        assert clock1.monotonic() == clock2.monotonic()
        assert clock1.now() == clock2.now()

    def test_event_timestamps_deterministic(self):
        """Events have deterministic timestamps with FakeClock."""
        def run_session():
            clock = FakeClock(start=1000.0)
            session = HarnessSession(
                clock=clock,
                rng_seed=42,
                providers=create_default_stub_providers(),
            )
            result = session.send("/time")
            return result

        result1 = run_session()
        result2 = run_session()

        # Timestamps should match (not ignored)
        for e1, e2 in zip(result1.user_events, result2.user_events):
            assert e1.timestamp == e2.timestamp


class TestProviderDeterminism:
    """Tests that stub providers are deterministic."""

    def test_weather_provider_deterministic(self):
        """StubWeatherProvider returns same data for same location."""
        provider = StubWeatherProvider({
            "Madison, WI": WeatherResult(temp=33, condition="Cloudy"),
        })

        result1 = provider.get("weather_now", {"place": "Madison, WI"})
        result2 = provider.get("weather_now", {"place": "Madison, WI"})

        assert result1.payload == result2.payload
        assert result1.speech_text == result2.speech_text

    def test_news_provider_deterministic(self):
        """StubNewsProvider returns same headlines."""
        provider = StubNewsProvider()

        result1 = provider.get("news_headlines", {"category": "general"})
        result2 = provider.get("news_headlines", {"category": "general"})

        assert result1.payload == result2.payload


class TestEventStreamDeterminism:
    """Tests that event streams are deterministic."""

    def test_debug_event_order_deterministic(self):
        """Debug events appear in deterministic order."""
        def run_session():
            session = HarnessSession(
                rng_seed=42,
                providers=create_default_stub_providers(),
                debug_channels={"router", "grammar", "providers"},
            )
            return session.send("/weather Madison, WI")

        result1 = run_session()
        result2 = run_session()

        kinds1 = [e.kind for e in result1.debug_events]
        kinds2 = [e.kind for e in result2.debug_events]

        assert kinds1 == kinds2

    def test_user_event_order_deterministic(self):
        """User events appear in deterministic order."""
        def run_session():
            session = HarnessSession(
                rng_seed=42,
                providers=create_default_stub_providers(),
            )
            result = session.send("/weather")
            return result

        result1 = run_session()
        result2 = run_session()

        kinds1 = [e.kind for e in result1.user_events]
        kinds2 = [e.kind for e in result2.user_events]

        assert kinds1 == kinds2


class TestLLMDeterminism:
    """Tests that LLM stub is deterministic."""

    def test_stub_llm_sequential(self):
        """StubLLMClient returns responses in order."""
        responses = ["First", "Second", "Third"]

        llm1 = StubLLMClient(responses.copy())
        llm2 = StubLLMClient(responses.copy())

        from episodic.harness.runtime import LLMRequest

        for expected in responses:
            r1 = llm1.complete(LLMRequest(messages=[], model="test"))
            r2 = llm2.complete(LLMRequest(messages=[], model="test"))
            assert r1.content == r2.content == expected

    def test_llm_requests_recorded(self):
        """LLM requests are recorded for assertions."""
        llm = StubLLMClient(["Response"])
        from episodic.harness.runtime import LLMRequest

        request = LLMRequest(messages=[{"role": "user", "content": "test"}], model="gpt-4")
        llm.complete(request)

        assert len(llm.requests) == 1
        assert llm.requests[0].model == "gpt-4"
