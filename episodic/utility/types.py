"""
Utility Commands Type Definitions.

UtilityQuery: AST node for parsed utility commands.
UtilityResult: Result from handler execution.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class UtilityCategory(Enum):
    """Categories of utility commands."""
    TIME = "time"
    TIMER = "timer"
    ALARM = "alarm"
    WEATHER = "weather"
    CALC = "calc"
    NOTE = "note"
    REMINDER = "reminder"
    MEDIA = "media"
    SYSTEM = "system"
    ROUTINE = "routine"
    CALENDAR = "calendar"
    EMAIL = "email"


@dataclass(frozen=True)
class UtilityQuery:
    """
    AST node for utility commands.

    Produced by the utility parser when a command pattern is matched.
    Consumed by the dispatcher to route to the appropriate handler.
    """
    category: str           # time, weather, alarm, timer, calendar, media, calc, system, note, routine
    command: str            # category-specific action (e.g., "time_now", "timer_set")
    args: Dict[str, Any]    # normalized arguments
    confidence: float       # 0.0-1.0 (1.0 for slash commands)
    source: str             # "cli" | "voice"
    raw_input: str

    def to_dict(self) -> dict:
        """Canonical serialization."""
        return {
            "ast_kind": "UtilityQuery",
            "category": self.category,
            "command": self.command,
            "args": self.args,
            "confidence": self.confidence,
            "source": self.source,
        }

    def is_mutating(self) -> bool:
        """Check if this command modifies state."""
        return self.command in MUTATING_COMMANDS


class ResultStatus(Enum):
    """Status of utility command execution."""
    OK = "ok"
    ERROR = "error"
    CONFIRM = "confirm"  # Needs user confirmation
    FALLBACK = "fallback"  # Fall through to LLM


@dataclass
class UtilityResult:
    """
    Result from executing a utility command.

    Contains both display text (for CLI) and speech text (for TTS).
    """
    status: ResultStatus
    display_text: str           # Text to show in terminal
    speech_text: Optional[str] = None  # Text for TTS (None = use display_text)
    data: Dict[str, Any] = field(default_factory=dict)  # Structured data
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    side_effects: Optional[Dict[str, Any]] = None  # For undo support

    @classmethod
    def ok(cls, display: str, speech: Optional[str] = None, **data) -> "UtilityResult":
        """Create a successful result."""
        return cls(
            status=ResultStatus.OK,
            display_text=display,
            speech_text=speech,
            data=data,
        )

    @classmethod
    def error(cls, error_type: str, message: str) -> "UtilityResult":
        """Create an error result."""
        return cls(
            status=ResultStatus.ERROR,
            display_text=f"Error: {message}",
            error_type=error_type,
            error_message=message,
        )

    @classmethod
    def confirm(cls, prompt: str) -> "UtilityResult":
        """Create a confirmation request."""
        return cls(
            status=ResultStatus.CONFIRM,
            display_text=prompt,
        )

    @classmethod
    def fallback(cls) -> "UtilityResult":
        """Signal to fall through to LLM."""
        return cls(
            status=ResultStatus.FALLBACK,
            display_text="",
        )


# Commands that modify state (require higher confidence or confirmation)
MUTATING_COMMANDS = {
    # Timer
    "timer_set", "timer_cancel", "timer_pause", "timer_resume",
    # Alarm
    "alarm_set", "alarm_cancel", "alarm_snooze",
    # Reminder
    "remind_set", "remind_cancel",
    # Note
    "note_add", "note_delete",
    # Media
    "media_play", "media_pause", "media_stop", "media_next", "media_prev",
    "volume_up", "volume_down", "volume_mute", "volume_set",
    # System
    "stop", "cancel", "undo",
    # Calendar (legacy)
    "cal_add", "cal_cancel",
    # Calendar (MCP)
    "calendar.create", "calendar.delete", "calendar.reschedule",
    # Email (MCP)
    "email.create_draft", "email.reply", "email.forward", "email.delete_draft",
    # Routine
    "routine_run",
}
