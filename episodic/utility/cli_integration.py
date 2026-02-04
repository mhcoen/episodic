"""
CLI Integration for Utility Commands.

Wires up slash commands to the utility dispatcher.
Handles argument parsing and output display.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from ..config import config
from .types import UtilityQuery, UtilityResult, ResultStatus
from .dispatcher import dispatch_utility, create_utility_query
from .scheduler import Scheduler
from .adapters.base import AdapterRegistry
from .adapters.radio import RadioAdapter, NullRadioAdapter


# Global instances (initialized on first use)
_scheduler: Optional[Scheduler] = None
_adapter_registry: Optional[AdapterRegistry] = None
_last_result: Optional[UtilityResult] = None
_schema_initialized: bool = False


def _ensure_utility_schema() -> None:
    """Ensure utility database schema exists."""
    global _schema_initialized
    if _schema_initialized:
        return

    from ..db_connection import get_connection
    from .db import init_utility_schema

    with get_connection() as conn:
        init_utility_schema(conn)

    _schema_initialized = True


def get_scheduler() -> Scheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        import sqlite3
        from ..db_connection import get_db_path

        # Ensure schema exists first
        _ensure_utility_schema()

        user_tz = config.get("timezone", "America/Chicago")

        # Create a dedicated connection for the scheduler (not from pool)
        # The scheduler needs a persistent connection for its background thread
        db_path = get_db_path()
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        _scheduler = Scheduler(conn=conn, user_tz=user_tz)
        _scheduler.start()
    return _scheduler


def get_adapter_registry() -> AdapterRegistry:
    """Get or create the global adapter registry."""
    global _adapter_registry
    if _adapter_registry is None:
        _adapter_registry = AdapterRegistry()
        # Register radio adapter (uses NullRadioAdapter if VLC not available)
        try:
            radio = RadioAdapter()
            _adapter_registry.register(radio)
        except Exception:
            # Fall back to null adapter for testing
            _adapter_registry.register(NullRadioAdapter())
    return _adapter_registry


def _parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse duration string to seconds.

    Supports: 10s, 5m, 1h, 1h30m, 90m
    """
    duration_str = duration_str.lower().strip()

    # Handle combined formats like "1h30m"
    total_seconds = 0

    # Pattern: extract all (number, unit) pairs
    pattern = r'(\d+)\s*(h|hr|hour|hours|m|min|mins|minutes|s|sec|secs|seconds?)?'
    matches = re.findall(pattern, duration_str)

    if not matches:
        return None

    for value_str, unit in matches:
        value = int(value_str)

        if not unit or unit.startswith('s'):
            total_seconds += value
        elif unit.startswith('m'):
            total_seconds += value * 60
        elif unit.startswith('h'):
            total_seconds += value * 3600

    return total_seconds if total_seconds > 0 else None


def _parse_time(time_str: str, user_tz: str = "America/Chicago") -> Optional[datetime]:
    """
    Parse time string to datetime.

    Supports: 7am, 7:00am, 19:00, 7:00, 7:30pm
    """
    time_str = time_str.lower().strip()
    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    # Pattern for various time formats
    patterns = [
        # 7am, 7pm
        (r'^(\d{1,2})\s*(am|pm)$', lambda m: (int(m.group(1)), 0, m.group(2))),
        # 7:00am, 7:30pm
        (r'^(\d{1,2}):(\d{2})\s*(am|pm)$', lambda m: (int(m.group(1)), int(m.group(2)), m.group(3))),
        # 19:00, 7:00 (24-hour format)
        (r'^(\d{1,2}):(\d{2})$', lambda m: (int(m.group(1)), int(m.group(2)), None)),
    ]

    for pattern, extractor in patterns:
        match = re.match(pattern, time_str)
        if match:
            hour, minute, meridiem = extractor(match)

            # Convert to 24-hour format
            if meridiem:
                if meridiem == 'pm' and hour != 12:
                    hour += 12
                elif meridiem == 'am' and hour == 12:
                    hour = 0

            # Validate hour/minute ranges
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                return None

            # Create datetime for today
            alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # If time has passed today, schedule for tomorrow
            if alarm_time <= now:
                alarm_time += timedelta(days=1)

            return alarm_time

    return None


def _parse_remind_args(text: str, user_tz: str = "America/Chicago") -> Tuple[Optional[str], Optional[int], Optional[datetime]]:
    """
    Parse reminder text and time.

    Supports:
    - "call mom in 2h" → ("call mom", 7200, None)
    - "meeting at 3pm" → ("meeting", None, datetime)

    Returns (reminder_text, duration_seconds, alarm_datetime)
    """
    text = text.strip()

    # Try "X in Y" pattern
    in_match = re.match(r'^(.+?)\s+in\s+(.+)$', text, re.IGNORECASE)
    if in_match:
        reminder_text = in_match.group(1).strip()
        duration_str = in_match.group(2).strip()
        duration_s = _parse_duration(duration_str)
        if duration_s:
            return (reminder_text, duration_s, None)

    # Try "X at Y" pattern
    at_match = re.match(r'^(.+?)\s+at\s+(.+)$', text, re.IGNORECASE)
    if at_match:
        reminder_text = at_match.group(1).strip()
        time_str = at_match.group(2).strip()
        alarm_time = _parse_time(time_str, user_tz)
        if alarm_time:
            return (reminder_text, None, alarm_time)

    return (None, None, None)


def handle_utility_command(cmd: str, args_str: str) -> Optional[UtilityResult]:
    """
    Handle a utility slash command.

    Args:
        cmd: Command name without slash (e.g., "timer", "alarm")
        args_str: Arguments as a string

    Returns:
        UtilityResult if handled, None if not a utility command
    """
    global _last_result

    user_tz = config.get("timezone", "America/Chicago")

    # First, validate arguments and create query without accessing scheduler
    # This allows validation errors to be returned quickly
    query = None

    # Parse command and create query
    if cmd == "stop":
        query = create_utility_query("system", "stop", source="cli")

    elif cmd == "timer":
        args = args_str.split() if args_str else []
        if not args:
            # Show timer status
            query = create_utility_query("timer", "timer_status", source="cli")
        else:
            duration_s = _parse_duration(args[0])
            if duration_s is None:
                return UtilityResult.error("invalid_duration", f"Could not parse duration: {args[0]}")

            label = " ".join(args[1:]) if len(args) > 1 else None
            query = create_utility_query(
                "timer", "timer_set",
                args={"duration_s": duration_s, "label": label},
                source="cli",
                raw_input=f"timer {args_str}",
            )

    elif cmd == "alarm":
        args = args_str.split() if args_str else []
        if not args:
            # List alarms
            query = create_utility_query("alarm", "alarm_list", source="cli")
        else:
            # First arg is time, rest is label
            # Pass the raw time string - handler will parse it
            label = " ".join(args[1:]) if len(args) > 1 else None
            query = create_utility_query(
                "alarm", "alarm_set",
                args={"time": args[0], "label": label},
                source="cli",
                raw_input=f"alarm {args_str}",
            )

    elif cmd == "time":
        query = create_utility_query("time", "time_now", source="cli")

    elif cmd == "calc":
        if not args_str:
            return UtilityResult.error("missing_expression", "Usage: /calc <expression>")
        query = create_utility_query(
            "calc", "calc_expr",
            args={"expr": args_str},
            source="cli",
            raw_input=f"calc {args_str}",
        )

    elif cmd == "note":
        if not args_str:
            # List notes
            query = create_utility_query("note", "note_list", source="cli")
        else:
            query = create_utility_query(
                "note", "note_add",
                args={"text": args_str},
                source="cli",
                raw_input=f"note {args_str}",
            )

    elif cmd == "remind":
        if not args_str:
            # List reminders
            query = create_utility_query("reminder", "remind_list", source="cli")
        else:
            reminder_text, duration_s, alarm_time = _parse_remind_args(args_str, user_tz)
            if reminder_text is None:
                return UtilityResult.error(
                    "invalid_format",
                    "Usage: /remind <text> in <duration> or /remind <text> at <time>"
                )

            args_dict = {"text": reminder_text}
            if duration_s:
                args_dict["minutes"] = duration_s // 60
            elif alarm_time:
                args_dict["at_time"] = alarm_time.isoformat()

            query = create_utility_query(
                "reminder", "remind_set",
                args=args_dict,
                source="cli",
                raw_input=f"remind {args_str}",
            )

    elif cmd == "play":
        if not args_str:
            return UtilityResult.error("missing_query", "Usage: /play <station>")
        query = create_utility_query(
            "media", "media_play",
            args={"query": args_str, "source": "radio"},
            source="cli",
            raw_input=f"play {args_str}",
        )

    elif cmd == "pause":
        query = create_utility_query("media", "media_pause", source="cli")

    elif cmd == "cancel":
        args = args_str.split() if args_str else []
        if not args:
            # Cancel most recent timer/alarm
            query = create_utility_query("system", "cancel", source="cli")
        elif args[0].lower() == "timer":
            query = create_utility_query("timer", "timer_cancel", source="cli")
        elif args[0].lower() == "alarm":
            query = create_utility_query("alarm", "alarm_cancel", source="cli")
        else:
            query = create_utility_query("system", "cancel", source="cli")

    elif cmd == "undo":
        query = create_utility_query("system", "undo", source="cli")

    elif cmd == "dnd":
        args = args_str.split() if args_str else []
        if not args:
            # Toggle DND
            query = create_utility_query("system", "dnd_on", source="cli")
        elif args[0].lower() == "on":
            query = create_utility_query("system", "dnd_on", source="cli")
        elif args[0].lower() == "off":
            query = create_utility_query("system", "dnd_off", source="cli")
        else:
            # Duration specified
            duration_s = _parse_duration(args[0])
            if duration_s:
                query = create_utility_query(
                    "system", "dnd_on",
                    args={"duration_minutes": duration_s // 60},
                    source="cli",
                )
            else:
                return UtilityResult.error("invalid_duration", f"Could not parse DND duration: {args[0]}")

    elif cmd == "status":
        query = create_utility_query("system", "status", source="cli")

    elif cmd == "weather":
        # /weather [location]
        place = args_str.strip() if args_str else "current"
        query = create_utility_query(
            "weather", "weather_now",
            args={"place": place},
            source="cli",
            raw_input=f"weather {args_str}" if args_str else "weather",
        )

    elif cmd == "forecast":
        # /forecast [location]
        place = args_str.strip() if args_str else "current"
        query = create_utility_query(
            "weather", "weather_forecast",
            args={"place": place},
            source="cli",
            raw_input=f"forecast {args_str}" if args_str else "forecast",
        )

    elif cmd == "news":
        # /news [category]
        category = args_str.strip().lower() if args_str else "general"
        query = create_utility_query(
            "news", "news_headlines",
            args={"category": category},
            source="cli",
            raw_input=f"news {args_str}" if args_str else "news",
        )

    else:
        # Not a utility command
        return None

    if query is None:
        return None

    # Initialize only the services needed for this command category
    scheduler = None
    adapter_registry = None

    # Categories that need scheduler
    if query.category in ("timer", "alarm", "reminder"):
        scheduler = get_scheduler()

    # Categories that need adapter registry
    if query.category == "media":
        adapter_registry = get_adapter_registry()

    # System commands: stop needs adapters, others may need scheduler
    if query.category == "system":
        if query.command in ("stop",):
            adapter_registry = get_adapter_registry()
        elif query.command in ("cancel", "status"):
            scheduler = get_scheduler()
            adapter_registry = get_adapter_registry()

    # Ensure utility schema exists (for event logging)
    _ensure_utility_schema()

    # Get database connection
    from ..db_connection import get_connection

    # Dispatch the query
    with get_connection() as conn:
        result = dispatch_utility(
            query,
            conn=conn,
            user_tz=user_tz,
            scheduler=scheduler,
            adapter_registry=adapter_registry,
            last_result=_last_result,
        )

    # Store for undo/repeat
    _last_result = result

    return result


def display_utility_result(result: UtilityResult) -> None:
    """Display utility result using Episodic's standard output mechanism."""
    from ..unified_streaming import unified_stream_text
    from ..configuration import get_system_color

    if result.status == ResultStatus.OK:
        # Display success message
        text = result.display_text or "Done"
        unified_stream_text(text, color=get_system_color(), enable_tts=False)
    elif result.status == ResultStatus.ERROR:
        # Display error
        error_msg = result.error_message or "An error occurred"
        unified_stream_text(f"Error: {error_msg}", color="red", enable_tts=False)
    elif result.status == ResultStatus.CONFIRM:
        # Display confirmation prompt
        text = result.display_text or "Confirm?"
        unified_stream_text(text, color="yellow", enable_tts=False)
    elif result.status == ResultStatus.FALLBACK:
        # Fall back to LLM (shouldn't happen for CLI commands)
        unified_stream_text("Command not understood", color="yellow", enable_tts=False)


def is_utility_command(cmd: str) -> bool:
    """Check if a command is a utility command."""
    utility_commands = {
        "stop", "timer", "alarm", "time", "calc", "note",
        "remind", "play", "pause", "cancel", "undo", "dnd", "status",
        "weather", "forecast", "news"
    }
    return cmd in utility_commands


def shutdown_utility_services() -> None:
    """Shutdown utility services (scheduler, adapters)."""
    global _scheduler, _adapter_registry

    if _scheduler is not None:
        # Close the scheduler's dedicated connection
        if _scheduler._conn is not None:
            try:
                _scheduler._conn.close()
            except Exception:
                pass
        _scheduler.stop()
        _scheduler = None

    if _adapter_registry is not None:
        # Stop any playing adapters
        for adapter in _adapter_registry.list_adapters():
            try:
                adapter.stop()
            except Exception:
                pass
        _adapter_registry = None
