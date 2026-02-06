"""
System Control Handlers.

Handles system-level utility commands:
- stop: Stop TTS, audio player, media
- cancel: Cancel timer/alarm (delegates to specific handlers)
- undo: Execute inverse command from undo stack
- repeat: Re-speak last result
- status: Show active timers, alarms, what's playing
- dnd_on/dnd_off: Toggle Do Not Disturb
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo

from ..types import UtilityQuery, UtilityResult


def handle_stop(
    query: UtilityQuery,
    scheduler=None,
    audio_player=None,
    tts_engine=None,
    media_adapters: Optional[Dict[str, Any]] = None,
) -> UtilityResult:
    """
    Handle stop command.

    Priority order:
    1. Stop sounding alarm/timer first (they demand attention)
    2. Only stop media (radio) if no alarm/timer is sounding
    """
    stopped = []

    # First priority: Check if an alarm or timer is sounding
    alarm_timer_sounding = False
    sound_info = None
    if audio_player is not None:
        try:
            if hasattr(audio_player, "is_alarm_or_timer_sounding"):
                alarm_timer_sounding = audio_player.is_alarm_or_timer_sounding()
                if alarm_timer_sounding and hasattr(audio_player, "get_current_sound_info"):
                    sound_info = audio_player.get_current_sound_info()
        except Exception:
            pass

    if alarm_timer_sounding:
        # Stop only the alarm/timer sound
        try:
            audio_player.stop()
            if sound_info:
                sound_type, label = sound_info
                if label:
                    stopped.append(f"{label} alarm" if sound_type and sound_type.name == "ALARM" else f"{label} timer")
                else:
                    stopped.append("alarm" if sound_type and sound_type.name == "ALARM" else "timer")
            else:
                stopped.append("alarm")
        except Exception:
            pass
    else:
        # No alarm/timer detected — try stopping audio anyway (catch-all)
        if audio_player is not None:
            try:
                if audio_player.is_playing():
                    audio_player.stop()
                    stopped.append("audio")
            except Exception:
                pass

        # Stop TTS
        if tts_engine is not None:
            try:
                tts_engine.stop()
                stopped.append("speech")
            except Exception:
                pass

        # Stop media adapters (radio, etc.)
        if media_adapters:
            for name, adapter in media_adapters.items():
                try:
                    was_playing = False
                    playing_info = name
                    if hasattr(adapter, "is_playing"):
                        was_playing = adapter.is_playing()
                    if hasattr(adapter, "_current_station") and adapter._current_station:
                        was_playing = True
                        playing_info = adapter._current_station.get("name", name)

                    adapter.stop()
                    if was_playing:
                        stopped.append(playing_info)
                except Exception:
                    pass

    if stopped:
        display = f"Stopped: {', '.join(stopped)}"
        speech = "Stopped"
    else:
        display = "Nothing to stop"
        speech = "Nothing to stop"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        stopped=stopped,
    )


def handle_cancel(
    query: UtilityQuery,
    scheduler=None,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle cancel command.

    Cancels timer or alarm by label or most recent.
    Delegates to timer_cancel or alarm_cancel.
    """
    from .timer import handle_timer_cancel
    from .alarm import handle_alarm_cancel

    scope = query.args.get("scope")  # "timer", "alarm", or None (both)
    label = query.args.get("label")

    cancelled = []

    if scope is None or scope == "timer":
        # Try to cancel timer
        timer_query = UtilityQuery(
            category="timer",
            command="timer_cancel",
            args={"label": label} if label else {},
            confidence=query.confidence,
            source=query.source,
            raw_input=query.raw_input,
        )
        timer_result = handle_timer_cancel(timer_query, scheduler, conn)
        if timer_result.status.value == "ok":
            cancelled.append("timer")

    if scope is None or scope == "alarm":
        # Try to cancel alarm
        alarm_query = UtilityQuery(
            category="alarm",
            command="alarm_cancel",
            args={"label": label} if label else {},
            confidence=query.confidence,
            source=query.source,
            raw_input=query.raw_input,
        )
        alarm_result = handle_alarm_cancel(alarm_query, scheduler, conn, "America/Chicago")
        if alarm_result.status.value == "ok":
            cancelled.append("alarm")

    if cancelled:
        display = f"Cancelled: {', '.join(cancelled)}"
        speech = "Cancelled"
    else:
        display = "Nothing to cancel"
        speech = "Nothing to cancel"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        cancelled=cancelled,
    )


def handle_undo(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    dispatcher=None,
) -> UtilityResult:
    """
    Handle undo command.

    Pops most recent action from undo stack and executes inverse.
    """
    if conn is None:
        return UtilityResult.error("no_database", "Undo requires database connection")

    cursor = conn.cursor()

    # Get most recent unexecuted undo
    cursor.execute("""
        SELECT id, event_id, inverse_command_json
        FROM undo_stack
        WHERE executed = 0
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    if row is None:
        return UtilityResult.ok(
            display="Nothing to undo",
            speech="Nothing to undo",
        )

    undo_id, event_id, inverse_json = row

    try:
        inverse_command = json.loads(inverse_json)
    except json.JSONDecodeError:
        return UtilityResult.error("invalid_undo", "Invalid undo command")

    # Mark as executed first (prevent double-undo)
    cursor.execute(
        "UPDATE undo_stack SET executed = 1 WHERE id = ?",
        (undo_id,)
    )
    conn.commit()

    # Execute the inverse command
    if dispatcher is not None:
        inverse_query = UtilityQuery(
            category=inverse_command.get("category", "system"),
            command=inverse_command.get("command", ""),
            args=inverse_command.get("args", {}),
            confidence=1.0,
            source="undo",
            raw_input="",
        )
        result = dispatcher(inverse_query)
        return UtilityResult.ok(
            display=f"Undone: {inverse_command.get('command', 'action')}",
            speech="Undone",
            inverse_command=inverse_command,
            nested_result=result.to_dict() if hasattr(result, 'to_dict') else None,
        )

    return UtilityResult.ok(
        display=f"Undone: {inverse_command.get('command', 'action')}",
        speech="Undone",
        inverse_command=inverse_command,
    )


def push_undo(
    conn: sqlite3.Connection,
    event_id: int,
    inverse_command: Dict[str, Any],
) -> int:
    """
    Push an undo action onto the stack.

    Args:
        conn: Database connection
        event_id: ID of the event being made undoable
        inverse_command: The command to run to undo

    Returns:
        undo_id
    """
    import time

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO undo_stack (event_id, inverse_command_json, created_at, executed)
        VALUES (?, ?, ?, 0)
    """, (
        event_id,
        json.dumps(inverse_command),
        int(time.time()),
    ))
    conn.commit()
    return cursor.lastrowid


def handle_repeat(
    query: UtilityQuery,
    last_result: Optional[UtilityResult] = None,
    tts_engine=None,
) -> UtilityResult:
    """
    Handle repeat command.

    Re-speaks the last result.
    """
    if last_result is None:
        return UtilityResult.ok(
            display="Nothing to repeat",
            speech="Nothing to repeat",
        )

    speech = last_result.speech_text or last_result.display_text

    # Speak it again
    if tts_engine is not None:
        try:
            tts_engine.speak(speech)
        except Exception:
            pass

    return UtilityResult.ok(
        display=f"Repeating: {speech}",
        speech=speech,
    )


def _format_ago(dt: datetime) -> str:
    """Format a datetime as a human-readable 'ago' string."""
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def handle_status(
    query: UtilityQuery,
    scheduler=None,
    conn: Optional[sqlite3.Connection] = None,
    audio_player=None,
    media_adapters: Optional[Dict[str, Any]] = None,
) -> UtilityResult:
    """
    Handle status command.

    Shows active timers/alarms, media, and data provider status.
    Output uses colon-aligned labels with no emoji prefix.
    """
    from ..scheduler import TaskType

    # Collect (label, value) pairs for colon-aligned display
    items: List[tuple] = []
    data = {}

    # --- What's currently sounding? ---
    if audio_player and audio_player.is_playing():
        sound_info = None
        if hasattr(audio_player, "get_current_sound_info"):
            sound_info = audio_player.get_current_sound_info()
        if sound_info:
            sound_type, label = sound_info
            kind = sound_type.value if sound_type else "sound"
            desc = f"{label} {kind}" if label else kind
            items.append(("Alert", f"{desc} (sounding)"))
        else:
            items.append(("Audio", "playing"))
        data["audio_playing"] = True

    # Media adapters
    if media_adapters:
        for name, adapter in media_adapters.items():
            try:
                if hasattr(adapter, "is_playing") and adapter.is_playing():
                    playing_info = name
                    if hasattr(adapter, "_current_station") and adapter._current_station:
                        playing_info = adapter._current_station.get("name", name)
                    items.append(("Playing", playing_info))
                    data[f"{name}_playing"] = True
            except Exception:
                pass

    # Active timers
    if scheduler:
        timers = scheduler.list_pending(TaskType.TIMER)
        for i, t in enumerate(timers, 1):
            remaining = scheduler.get_timer_remaining(t.id)
            if remaining is not None:
                label = t.label or f"#{i}"
                items.append(("Timer", f"{label} ({_format_remaining(remaining)} remaining)"))
        if timers:
            data["timers"] = len(timers)

    # Active alarms
    if scheduler:
        alarms = scheduler.list_pending(TaskType.ALARM)
        for a in alarms:
            time_str = a.next_run_wall.strftime("%I:%M %p").lstrip("0")
            label = f"{a.label} at {time_str}" if a.label else time_str
            items.append(("Alarm", label))
        if alarms:
            data["alarms"] = len(alarms)

    # DND status
    if scheduler and scheduler.is_dnd_active():
        items.append(("DND", "on"))
        data["dnd_active"] = True

    # --- Data providers ---
    try:
        from ..data_refresh import get_data_refresh_scheduler
        refresh_scheduler = get_data_refresh_scheduler()
        refresh_status = refresh_scheduler.status()

        if refresh_status.get("running"):
            for key, job in refresh_status.get("jobs", {}).items():
                last = job.get("last_refresh")
                errors = job.get("error_count", 0)
                provider_name = job.get("provider", key)

                if last:
                    dt = datetime.fromisoformat(last)
                    items.append((provider_name.title(), f"updated {_format_ago(dt)}"))
                elif errors > 0:
                    items.append((provider_name.title(), "unavailable"))
                else:
                    items.append((provider_name.title(), "pending"))
    except Exception:
        pass

    # --- Build colon-aligned output ---
    if items:
        max_label = max(len(label) for label, _ in items)
        lines = []
        for label, value in items:
            lines.append(f"{label:>{max_label}}: {value}")
        display = "\n".join(lines)
        speech = f"You have {len(items)} status items" if len(items) > 1 else f"{items[0][0]}: {items[0][1]}"
    else:
        display = "Nothing active"
        speech = "Nothing active"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        _command="status",
        **data,
    )


def handle_dnd_on(
    query: UtilityQuery,
    scheduler=None,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle dnd_on command.

    Enables Do Not Disturb mode.
    """
    duration_m = query.args.get("duration_m")  # Optional duration in minutes
    until_time = query.args.get("until")  # Optional end time

    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    if duration_m:
        until = now + timedelta(minutes=duration_m)
    elif until_time:
        # Parse time string
        try:
            if ":" in until_time:
                parts = until_time.replace("am", "").replace("pm", "").strip().split(":")
                h = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 else 0

                if "pm" in until_time.lower() and h < 12:
                    h += 12
                elif "am" in until_time.lower() and h == 12:
                    h = 0

                until = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if until <= now:
                    until += timedelta(days=1)
            else:
                until = now + timedelta(hours=1)  # Default 1 hour
        except (ValueError, IndexError):
            until = now + timedelta(hours=1)
    else:
        # Default: until manually turned off (use far future)
        until = now + timedelta(hours=24)

    if scheduler:
        scheduler.set_dnd(until)

    # Persist preference
    if conn:
        from ..db import set_preference
        set_preference(conn, "dnd_enabled", True)
        set_preference(conn, "dnd_until", until.isoformat())

    time_str = until.strftime("%I:%M %p").lstrip("0")
    display = f"Do Not Disturb enabled until {time_str}"
    speech = f"Do Not Disturb enabled until {time_str}"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        dnd_until=until.isoformat(),
    )


def handle_dnd_off(
    query: UtilityQuery,
    scheduler=None,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle dnd_off command.

    Disables Do Not Disturb mode.
    """
    if scheduler:
        scheduler.set_dnd(None)

    if conn:
        from ..db import set_preference
        set_preference(conn, "dnd_enabled", False)
        set_preference(conn, "dnd_until", None)

    return UtilityResult.ok(
        display="Do Not Disturb disabled",
        speech="Do Not Disturb disabled",
    )


def _format_remaining(seconds: int) -> str:
    """Format remaining seconds as human-readable."""
    if seconds < 60:
        return f"{seconds}s"

    minutes = seconds // 60
    secs = seconds % 60

    if minutes < 60:
        if secs:
            return f"{minutes}m {secs}s"
        return f"{minutes}m"

    hours = minutes // 60
    mins = minutes % 60

    if mins:
        return f"{hours}h {mins}m"
    return f"{hours}h"


# Command routing for system category
SYSTEM_HANDLERS = {
    "stop": handle_stop,
    "cancel": handle_cancel,
    "undo": handle_undo,
    "repeat": handle_repeat,
    "status": handle_status,
    "dnd_on": handle_dnd_on,
    "dnd_off": handle_dnd_off,
}


def dispatch_system_command(
    query: UtilityQuery,
    scheduler=None,
    conn: Optional[sqlite3.Connection] = None,
    audio_player=None,
    tts_engine=None,
    media_adapters: Optional[Dict[str, Any]] = None,
    last_result: Optional[UtilityResult] = None,
    dispatcher=None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """Dispatch a system category command to the appropriate handler."""
    handler = SYSTEM_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown system command: {query.command}"
        )

    # Each handler has different signature requirements
    if query.command == "stop":
        return handler(query, scheduler, audio_player, tts_engine, media_adapters)
    elif query.command == "cancel":
        return handler(query, scheduler, conn)
    elif query.command == "undo":
        return handler(query, conn, dispatcher)
    elif query.command == "repeat":
        return handler(query, last_result, tts_engine)
    elif query.command == "status":
        return handler(query, scheduler, conn, audio_player, media_adapters)
    elif query.command == "dnd_on":
        return handler(query, scheduler, conn, user_tz)
    elif query.command == "dnd_off":
        return handler(query, scheduler, conn)

    return handler(query)
