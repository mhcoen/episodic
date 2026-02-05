"""
Test Harness Infrastructure.

Provides deterministic testing infrastructure for Episodic including:
- Event/EventStream for structured output
- Clock protocol for time injection
- EventStore for event persistence
- RuntimeState for dependency injection
- TestSession for programmatic testing
"""

from .events import Event, EventStream, EventKind, EventLevel
from .clock import Clock, SystemClock, FakeClock
from .runtime import RuntimeState, LLMClient, LLMRequest, LLMResponse, StubLLMClient, EventStore, EphemeralEventStore
from .processor import process_input
from .providers import (
    StubWeatherProvider,
    StubNewsProvider,
    StubTimerProvider,
    WeatherResult,
    NewsItem,
    create_default_stub_providers,
)

__all__ = [
    "Event",
    "EventStream",
    "EventKind",
    "EventLevel",
    "Clock",
    "SystemClock",
    "FakeClock",
    "RuntimeState",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "StubLLMClient",
    "EventStore",
    "EphemeralEventStore",
    "process_input",
    "StubWeatherProvider",
    "StubNewsProvider",
    "StubTimerProvider",
    "WeatherResult",
    "NewsItem",
    "create_default_stub_providers",
]
