"""
Fixtures for integration tests.

Provides TestSession and related testing infrastructure.
"""

import os
import fnmatch
import pytest
import tempfile

from episodic.harness import (
    RuntimeState,
    FakeClock,
    StubLLMClient,
    EphemeralEventStore,
    process_input,
    StubWeatherProvider,
    StubNewsProvider,
    StubTimerProvider,
    WeatherResult,
    create_default_stub_providers,
)


class HarnessSession:
    """
    Programmatic interface for testing.

    Calls same process_input as CLI would, with injected dependencies.
    """

    def __init__(
        self,
        db: str = ":memory:",
        clock=None,
        llm=None,
        providers=None,
        rng_seed: int = 42,
        debug_channels=None,
    ):
        self._validate_not_prod(db)

        import sqlite3
        import random

        self.runtime = RuntimeState(
            db=sqlite3.connect(db) if db != ":memory:" else sqlite3.connect(":memory:"),
            clock=clock or FakeClock(start=0),
            llm=llm or StubLLMClient([]),
            event_store=EphemeralEventStore(),
            rng=random.Random(rng_seed),
            providers=providers or {},
            debug_channels=debug_channels or {"router", "grammar", "context"},
        )

    def send(self, text: str):
        """Process input through same pipeline as CLI."""
        return process_input(text, self.runtime)

    def advance_time(self, seconds: float):
        """Advance fake clock."""
        if hasattr(self.runtime.clock, "advance"):
            self.runtime.clock.advance(seconds)

    def reset(self):
        """Clear state for next test."""
        if hasattr(self.runtime.event_store, "clear"):
            self.runtime.event_store.clear()

    @staticmethod
    def _validate_not_prod(path: str):
        """Fail fast if test would touch production."""
        if path == ":memory:":
            return

        forbidden_patterns = [
            "~/.episodic",
            "~/.episodic/*",
            "*/.episodic/episodic.db",
            "*/episodic.db",
        ]

        resolved = os.path.realpath(os.path.expanduser(path))

        for pattern in forbidden_patterns:
            expanded = os.path.expanduser(pattern)
            if fnmatch.fnmatch(resolved, expanded):
                raise RuntimeError(f"Test cannot use production path: {path}")

        # Also check explicit home directory
        home_episodic = os.path.expanduser("~/.episodic")
        if resolved.startswith(home_episodic):
            raise RuntimeError(f"Test cannot use production path: {path}")


@pytest.fixture
def test_session():
    """Standard test session with stubs."""
    session = HarnessSession(
        providers=create_default_stub_providers(),
        debug_channels={"router", "grammar", "context", "providers"}
    )
    yield session


@pytest.fixture
def session_with_providers():
    """Test session with custom stub providers for provider testing."""
    providers = {
        "weather": StubWeatherProvider({
            "current": WeatherResult(temp=72, condition="Sunny", location="Test City"),
            "Madison, WI": WeatherResult(temp=33, condition="Cloudy", location="Madison", high=36, low=28),
            "New York": WeatherResult(temp=55, condition="Rainy", location="New York", emoji="🌧️"),
        }),
        "news": StubNewsProvider(),
        "timer": StubTimerProvider(),
    }
    session = HarnessSession(
        providers=providers,
        debug_channels={"router", "grammar", "context", "providers"}
    )
    yield session


@pytest.fixture
def session_with_llm():
    """Test session with LLM responses configured."""
    session = HarnessSession(
        llm=StubLLMClient([
            "This is a test response.",
            "Here is another response.",
            "And a third one.",
        ]),
        debug_channels={"router", "grammar", "context", "llm"},
    )
    yield session


@pytest.fixture
def temp_db_path():
    """Temporary database file for persistence tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass
