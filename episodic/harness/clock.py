"""
Clock protocol and implementations.

Provides injectable time source for deterministic testing.
"""

import time
from typing import Protocol


class Clock(Protocol):
    """Protocol for time sources."""

    def now(self) -> float:
        """Return current wall-clock time as Unix timestamp."""
        ...

    def monotonic(self) -> float:
        """Return monotonic time for duration calculations."""
        ...


class SystemClock:
    """Production clock using system time."""

    def now(self) -> float:
        """Return current wall-clock time."""
        return time.time()

    def monotonic(self) -> float:
        """Return monotonic time."""
        return time.monotonic()


class FakeClock:
    """
    Test clock with manual time control.

    Time only advances when explicitly advanced.
    Both now() and monotonic() return the same value for simplicity.
    """

    def __init__(self, start: float = 0.0):
        """
        Initialize fake clock.

        Args:
            start: Initial time value (defaults to 0)
        """
        self._time = start

    def now(self) -> float:
        """Return current fake wall-clock time."""
        return self._time

    def monotonic(self) -> float:
        """Return current fake monotonic time."""
        return self._time

    def advance(self, seconds: float) -> None:
        """
        Advance time by specified seconds.

        Args:
            seconds: Number of seconds to advance
        """
        if seconds < 0:
            raise ValueError("Cannot advance time backwards")
        self._time += seconds

    def set(self, timestamp: float) -> None:
        """
        Set time to specific value.

        Args:
            timestamp: New time value
        """
        self._time = timestamp

    def __repr__(self) -> str:
        return f"FakeClock(time={self._time})"
