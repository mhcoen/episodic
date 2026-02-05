"""
RuntimeState - Dependency injection container.

Holds all injectable dependencies for input processing.
"""

import random
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Set

from .clock import Clock, SystemClock
from .events import EventStream


class LLMClient(Protocol):
    """Protocol for LLM completion."""

    def complete(self, request: "LLMRequest") -> "LLMResponse":
        """Complete a prompt and return response."""
        ...


class EventStore(Protocol):
    """Protocol for event persistence."""

    def write(self, event: Any) -> None:
        """Write an event."""
        ...

    def query(self, **filters: Any) -> list:
        """Query events."""
        ...


@dataclass
class LLMRequest:
    """Request to LLM."""
    messages: list
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    tokens_used: int
    model: str = ""
    finish_reason: str = "stop"


class StubLLMClient:
    """
    Test LLM client with scripted responses.

    Can be configured with:
    - List of responses (used sequentially)
    - Dict mapping prompt hashes to responses
    """

    def __init__(self, responses: list | Dict[tuple, str] | None = None):
        self._responses = responses or []
        self._index = 0
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return next scripted response."""
        self.requests.append(request)

        if isinstance(self._responses, dict):
            # Hash-based lookup (for deterministic matching)
            # TODO: implement proper hashing
            key = (hash(str(request.messages)), request.model, request.temperature)
            if key in self._responses:
                return LLMResponse(content=self._responses[key], tokens_used=100)
            return LLMResponse(content="[No matching response]", tokens_used=10)

        # Sequential responses
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return LLMResponse(content=response, tokens_used=100)

        return LLMResponse(content="[No more stubbed responses]", tokens_used=10)

    def reset(self) -> None:
        """Reset response index."""
        self._index = 0
        self.requests.clear()


class EphemeralEventStore:
    """In-memory event store for testing."""

    def __init__(self):
        self._events: list = []

    def write(self, event: Any) -> None:
        """Store event in memory."""
        self._events.append(event)

    def query(self, **filters: Any) -> list:
        """Query events (basic filtering)."""
        results = self._events
        for key, value in filters.items():
            results = [e for e in results if getattr(e, key, None) == value]
        return results

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


@dataclass
class RuntimeState:
    """
    Dependency injection container for input processing.

    All dependencies are injectable for testing.
    Production code creates this with real implementations.
    Tests create with stubs/fakes.
    """
    # Database connection
    db: Optional[sqlite3.Connection] = None

    # Time source
    clock: Clock = field(default_factory=SystemClock)

    # LLM client
    llm: Optional[LLMClient] = None

    # Event persistence
    event_store: Optional[EventStore] = None

    # Seeded RNG for deterministic behavior
    rng: random.Random = field(default_factory=lambda: random.Random(42))

    # Data providers (weather, news, etc.)
    providers: Dict[str, Any] = field(default_factory=dict)

    # Enabled debug channels
    debug_channels: Set[str] = field(default_factory=set)

    # User timezone
    timezone: str = "America/Chicago"

    # Configuration getter (for accessing config values)
    config_getter: Optional[Callable[[str, Any], Any]] = None

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if self.config_getter:
            return self.config_getter(key, default)
        return default

    def is_debug_enabled(self, channel: str) -> bool:
        """Check if a debug channel is enabled."""
        return channel in self.debug_channels or "all" in self.debug_channels

    def emit_debug(
        self,
        kind: str,
        channel: str,
        fields: Dict[str, Any],
        stream: EventStream,
    ) -> None:
        """
        Emit a debug event if channel is enabled.

        Args:
            kind: Event kind
            channel: Debug channel
            fields: Event data
            stream: EventStream to add event to
        """
        if self.is_debug_enabled(channel):
            from .events import Event
            event = Event.debug(
                kind=kind,
                channel=channel,
                fields=fields,
                timestamp=self.clock.monotonic(),
            )
            stream.add_debug_event(event)

    @classmethod
    def for_testing(
        cls,
        rng_seed: int = 42,
        debug_channels: Optional[Set[str]] = None,
        llm_responses: Optional[list] = None,
        providers: Optional[Dict[str, Any]] = None,
    ) -> "RuntimeState":
        """
        Create a RuntimeState configured for testing.

        Args:
            rng_seed: Seed for deterministic random behavior
            debug_channels: Debug channels to enable
            llm_responses: Scripted LLM responses
            providers: Stub providers
        """
        from .clock import FakeClock

        return cls(
            db=sqlite3.connect(":memory:"),
            clock=FakeClock(start=0),
            llm=StubLLMClient(llm_responses or []),
            event_store=EphemeralEventStore(),
            rng=random.Random(rng_seed),
            providers=providers or {},
            debug_channels=debug_channels or {"router", "grammar", "context"},
        )
