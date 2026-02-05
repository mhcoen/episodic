"""
Event and EventStream definitions.

Events are structured records of system activity.
EventStream collects events from a single input processing cycle.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


# Schema version - bump on breaking changes
EVENT_SCHEMA_VERSION = 1


class EventLevel(str, Enum):
    """Event severity level."""
    INFO = "info"
    DEBUG = "debug"
    WARN = "warn"
    ERROR = "error"


class EventKind(str, Enum):
    """
    Event kinds for structured logging.

    User-visible events (displayed/spoken):
    - ASSISTANT_RESPONSE: Main LLM output text
    - UTILITY_RESULT: Weather, news, timer confirmations
    - ERROR: User-facing error messages
    - NOTIFICATION: Timer fired, alarm fired, reminder

    Debug events (gated by channels):
    - ROUTER_DECISION: Route decisions
    - PARSE_ATTEMPT: Voice grammar parsing
    - CONTEXT_PLAN: Context assembly planning
    - CONTEXT_FINAL: Final context sent to LLM
    - SCHEDULER_TICK: Background refresh activity
    - PROVIDER_CALL: Weather/news/timer calls
    - LLM_REQUEST_META: LLM request metadata (no content)
    """
    # User-visible
    ASSISTANT_RESPONSE = "assistant_response"
    UTILITY_RESULT = "utility_result"
    UTILITY_EXECUTED = "utility_executed"
    ERROR = "error"
    NOTIFICATION = "notification"
    COMMAND_RESULT = "command_result"

    # Debug
    ROUTER_DECISION = "router_decision"
    PARSE_ATTEMPT = "parse_attempt"
    CONTEXT_PLAN = "context_plan"
    CONTEXT_FINAL = "context_final"
    SCHEDULER_TICK = "scheduler_tick"
    PROVIDER_CALL = "provider_call"
    LLM_REQUEST_META = "llm_request_meta"


@dataclass
class Event:
    """
    A structured event record.

    Attributes:
        kind: Event type (from EventKind enum or string for extensibility)
        level: Severity level
        timestamp: Monotonic time from injected clock
        schema_version: For detecting incompatible changes
        fields: Event-specific data
        channel: Debug channel (for debug events)
    """
    kind: str
    level: EventLevel
    timestamp: float
    fields: Dict[str, Any]
    schema_version: int = EVENT_SCHEMA_VERSION
    channel: Optional[str] = None

    @classmethod
    def user(
        cls,
        kind: str,
        fields: Dict[str, Any],
        timestamp: float = 0.0,
    ) -> "Event":
        """Create a user-visible event."""
        return cls(
            kind=kind,
            level=EventLevel.INFO,
            timestamp=timestamp,
            fields=fields,
        )

    @classmethod
    def debug(
        cls,
        kind: str,
        channel: str,
        fields: Dict[str, Any],
        timestamp: float = 0.0,
    ) -> "Event":
        """Create a debug event."""
        return cls(
            kind=kind,
            level=EventLevel.DEBUG,
            timestamp=timestamp,
            fields=fields,
            channel=channel,
        )

    @classmethod
    def error(
        cls,
        message: str,
        timestamp: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> "Event":
        """Create an error event."""
        fields = {"message": message}
        if details:
            fields.update(details)
        return cls(
            kind=EventKind.ERROR.value,
            level=EventLevel.ERROR,
            timestamp=timestamp,
            fields=fields,
        )


@dataclass
class EventStream:
    """
    Collection of events from a single input processing cycle.

    Separates user-visible events (for CLI/TTS) from debug events
    (for testing and diagnostics).
    """
    user_events: List[Event] = field(default_factory=list)
    debug_events: List[Event] = field(default_factory=list)

    def add_user_event(self, event: Event) -> None:
        """Add a user-visible event."""
        self.user_events.append(event)

    def add_debug_event(self, event: Event) -> None:
        """Add a debug event."""
        self.debug_events.append(event)

    def add(self, event: Event) -> None:
        """Add event to appropriate list based on level."""
        if event.level == EventLevel.DEBUG:
            self.debug_events.append(event)
        else:
            self.user_events.append(event)

    def get_by_kind(self, kind: str) -> List[Event]:
        """Get all events of a specific kind (from both lists)."""
        return [
            e for e in self.user_events + self.debug_events
            if e.kind == kind
        ]

    def get_debug_by_channel(self, channel: str) -> List[Event]:
        """Get debug events for a specific channel."""
        return [
            e for e in self.debug_events
            if e.channel == channel
        ]

    @property
    def has_errors(self) -> bool:
        """Check if any error events occurred."""
        return any(e.level == EventLevel.ERROR for e in self.user_events)

    def merge(self, other: "EventStream") -> "EventStream":
        """Merge another EventStream into this one."""
        return EventStream(
            user_events=self.user_events + other.user_events,
            debug_events=self.debug_events + other.debug_events,
        )
