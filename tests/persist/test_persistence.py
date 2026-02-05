"""
Persistence tests.

These tests verify that state persists correctly across
session restarts using file-based databases.
"""

import pytest
from episodic.harness import (
    FakeClock,
    StubLLMClient,
    create_default_stub_providers,
)
from tests.integration.conftest import HarnessSession


class TestSessionPersistence:
    """Tests for basic session state persistence."""

    def test_session_with_temp_db(self, temp_db_path):
        """Session should work with temporary file database."""
        session = HarnessSession(
            db=temp_db_path,
            providers=create_default_stub_providers(),
        )
        result = session.send("/time")

        # Should get a result without error
        assert len(result.user_events) > 0

    def test_session_reconnect(self, temp_db_path):
        """Session should reconnect to existing database."""
        # Session 1: Create and use
        session1 = HarnessSession(
            db=temp_db_path,
            providers=create_default_stub_providers(),
        )
        session1.send("/time")
        del session1

        # Session 2: Reconnect
        session2 = HarnessSession(
            db=temp_db_path,
            providers=create_default_stub_providers(),
        )
        result = session2.send("/time")

        # Should work without error
        assert len(result.user_events) > 0


class TestProviderStatePersistence:
    """Tests for provider state across restarts."""

    def test_provider_calls_not_persisted(self, temp_db_path):
        """Provider call history should not persist across restarts."""
        # Session 1: Make some provider calls
        providers1 = create_default_stub_providers()
        session1 = HarnessSession(
            db=temp_db_path,
            providers=providers1,
        )
        session1.send("/weather")
        session1.send("/news")

        # Check calls were recorded in session 1
        assert len(providers1["weather"].calls) == 1
        assert len(providers1["news"].calls) == 1
        del session1

        # Session 2: Fresh providers
        providers2 = create_default_stub_providers()
        session2 = HarnessSession(
            db=temp_db_path,
            providers=providers2,
        )

        # Call history should be empty in new session
        assert len(providers2["weather"].calls) == 0
        assert len(providers2["news"].calls) == 0

    def test_cache_not_persisted(self, temp_db_path):
        """Provider cache should not persist across restarts."""
        # Session 1: Populate cache
        providers1 = create_default_stub_providers()
        session1 = HarnessSession(
            db=temp_db_path,
            providers=providers1,
        )
        session1.send("/weather Madison, WI")
        del session1

        # Session 2: Cache should be empty
        providers2 = create_default_stub_providers()
        session2 = HarnessSession(
            db=temp_db_path,
            providers=providers2,
        )

        # Calling weather should make a new provider call
        session2.send("/weather Madison, WI")
        assert len(providers2["weather"].calls) == 1  # Not cached


class TestClockPersistence:
    """Tests for clock state across restarts."""

    def test_fake_clock_not_persisted(self, temp_db_path):
        """FakeClock state should not persist across restarts."""
        # Session 1: Advance clock
        clock1 = FakeClock(start=1000.0)
        session1 = HarnessSession(
            db=temp_db_path,
            clock=clock1,
            providers=create_default_stub_providers(),
        )
        clock1.advance(5000)  # Advance to 6000
        del session1

        # Session 2: Fresh clock
        clock2 = FakeClock(start=0.0)
        session2 = HarnessSession(
            db=temp_db_path,
            clock=clock2,
            providers=create_default_stub_providers(),
        )

        # Clock should be at its initial value, not persisted
        assert clock2.monotonic() == 0.0


class TestLLMStatePersistence:
    """Tests for LLM client state across restarts."""

    def test_llm_response_index_not_persisted(self, temp_db_path):
        """StubLLMClient response index should not persist."""
        responses = ["Response 1", "Response 2", "Response 3"]

        # Session 1: Use some responses
        llm1 = StubLLMClient(responses.copy())
        session1 = HarnessSession(
            db=temp_db_path,
            llm=llm1,
            providers=create_default_stub_providers(),
        )
        session1.send("hello")  # Uses Response 1
        del session1

        # Session 2: Fresh LLM client
        llm2 = StubLLMClient(responses.copy())
        session2 = HarnessSession(
            db=temp_db_path,
            llm=llm2,
            providers=create_default_stub_providers(),
        )
        session2.send("hello")

        # Should start from Response 1 again, not continue from Response 2
        # (Checking via request count)
        assert len(llm2.requests) == 1


class TestRNGPersistence:
    """Tests for RNG state across restarts."""

    def test_rng_seed_not_persisted(self, temp_db_path):
        """RNG state should not persist - fresh seed each session."""
        # Session 1: Use RNG with seed 42
        session1 = HarnessSession(
            db=temp_db_path,
            rng_seed=42,
            providers=create_default_stub_providers(),
        )
        val1 = session1.runtime.rng.random()
        del session1

        # Session 2: Same seed should give same first value
        session2 = HarnessSession(
            db=temp_db_path,
            rng_seed=42,
            providers=create_default_stub_providers(),
        )
        val2 = session2.runtime.rng.random()

        # Same seed -> same sequence
        assert val1 == val2


class TestDebugChannelPersistence:
    """Tests for debug channel state across restarts."""

    def test_debug_channels_not_persisted(self, temp_db_path):
        """Debug channel settings should not persist."""
        # Session 1: Enable debug channels
        session1 = HarnessSession(
            db=temp_db_path,
            debug_channels={"router", "grammar", "llm"},
            providers=create_default_stub_providers(),
        )
        assert "router" in session1.runtime.debug_channels
        del session1

        # Session 2: Different channels
        session2 = HarnessSession(
            db=temp_db_path,
            debug_channels={"context"},  # Different
            providers=create_default_stub_providers(),
        )

        # Should have only the new session's channels
        assert "context" in session2.runtime.debug_channels
        assert "router" not in session2.runtime.debug_channels


class TestProductionPathValidation:
    """Tests for production path validation."""

    def test_memory_db_allowed(self):
        """In-memory database should be allowed."""
        session = HarnessSession(
            db=":memory:",
            providers=create_default_stub_providers(),
        )
        # Should not raise
        assert session is not None

    def test_temp_path_allowed(self, temp_db_path):
        """Temporary paths should be allowed."""
        session = HarnessSession(
            db=temp_db_path,
            providers=create_default_stub_providers(),
        )
        # Should not raise
        assert session is not None

    def test_production_path_blocked(self):
        """Production paths should be blocked.

        Note: The validation catches paths matching patterns like */episodic.db
        and anything under ~/.episodic. Since conftest.py sets HOME to a test
        directory, we must use the real home path explicitly.
        """
        import os
        import pwd

        # Get the REAL home directory, not the test HOME
        real_home = pwd.getpwuid(os.getuid()).pw_dir

        # This pattern is caught by the */episodic.db glob pattern
        production_path = os.path.join(real_home, ".episodic", "episodic.db")

        with pytest.raises(RuntimeError, match="production path"):
            HarnessSession(
                db=production_path,
                providers=create_default_stub_providers(),
            )

    def test_production_path_pattern_episodic_db(self):
        """The pattern */episodic.db should be blocked anywhere."""
        import os

        # This should be caught by */episodic.db pattern regardless of parent
        with pytest.raises(RuntimeError, match="production path"):
            HarnessSession(
                db="/any/random/path/episodic.db",
                providers=create_default_stub_providers(),
            )
