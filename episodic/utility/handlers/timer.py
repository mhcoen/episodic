"""
Timer Handler.

Handles timer-related utility commands:
- timer_set: Create a new countdown timer
- timer_cancel: Cancel an active timer
- timer_status: Check time remaining on timers
- timer_pause: Pause a running timer
- timer_resume: Resume a paused timer
- timer_list: List all active timers
"""

import uuid
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..types import UtilityQuery, UtilityResult
from ..scheduler import (
    Scheduler,
    TaskType,
    TaskStatus,
    TaskResult,
    create_timer_task,
)


from episodic.utility.handlers.timer_helpers import (  # noqa: F401  (re-exported)
    _format_duration,
    _format_time_remaining,
    _persist_timer,
    _update_timer_status,
    _get_active_timers,
    _get_timer_by_id,
    _find_timer_by_label,
)

def handle_timer_set(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    audio_player=None,
) -> UtilityResult:
    """
    Handle timer_set command.

    Args in query:
        duration_s: Duration in seconds
        label: Optional timer label
    """
    duration_s = query.args.get("duration_s")
    label = query.args.get("label")

    if duration_s is None:
        return UtilityResult.error("missing_duration", "No duration specified")

    if duration_s <= 0:
        return UtilityResult.error("invalid_duration", "Duration must be positive")

    if duration_s > 86400 * 7:  # 1 week max
        return UtilityResult.error("duration_too_long", "Timer cannot exceed 1 week")

    # Generate IDs
    timer_id = str(uuid.uuid4())

    # Create callback for when timer fires
    def timer_callback() -> TaskResult:
        # Play sound first (before DB which may fail)
        if audio_player:
            audio_player.play_timer(label)

        # Update timer status with a fresh connection
        # (the original conn is closed by the time the timer fires)
        try:
            from ...db_connection import get_connection
            with get_connection() as fresh_conn:
                _update_timer_status(fresh_conn, timer_id, "expired")
        except Exception:
            pass

        return TaskResult(
            status=TaskStatus.COMPLETED,
            output=f"{label} timer done" if label else "Timer done",
            side_effects=["timer_expired", timer_id],
        )

    # Create task
    task = create_timer_task(
        duration_s=duration_s,
        label=label,
        callback=timer_callback,
        reference_id=timer_id,
        user_tz=user_tz,
    )

    # Persist timer
    if conn:
        _persist_timer(conn, timer_id, duration_s, label, task.next_run_wall, task.id)

    # Add to scheduler
    scheduler.add_task(task)

    # Format response
    duration_str = _format_duration(duration_s)
    if label:
        display = f"Timer set: {label} ({duration_str})"
        speech = f"{label} timer set for {duration_str}"
    else:
        display = f"Timer set for {duration_str}"
        speech = f"Timer set for {duration_str}"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        _command="timer_set",
        timer_id=timer_id,
        task_id=task.id,
        duration=duration_s,
        label=label,
        expires_at=task.next_run_wall.isoformat(),
    )


def handle_timer_cancel(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle timer_cancel command.

    Args in query:
        timer_id: Specific timer ID (optional)
        label: Timer label to match (optional)
        all: Cancel all timers (optional)
    """
    timer_id = query.args.get("timer_id")
    label = query.args.get("label")
    cancel_all = query.args.get("all", False)

    if cancel_all:
        # Cancel all timers
        count = scheduler.cancel_by_type(TaskType.TIMER)

        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE timers SET status = 'cancelled' WHERE status IN ('active', 'paused')"
            )
            conn.commit()

        if count == 0:
            return UtilityResult.ok(
                display="No active timers to cancel",
                speech="No active timers",
            )

        return UtilityResult.ok(
            display=f"Cancelled {count} timer{'s' if count != 1 else ''}",
            speech=f"Cancelled {count} timer{'s' if count != 1 else ''}",
            cancelled_count=count,
        )

    # Find specific timer
    timer = None

    if timer_id and conn:
        timer = _get_timer_by_id(conn, timer_id)
    elif label and conn:
        timer = _find_timer_by_label(conn, label)
    elif conn:
        # Cancel most recent timer
        timers = _get_active_timers(conn)
        if timers:
            timer = timers[-1]  # Most recently created

    if timer is None:
        return UtilityResult.error(
            "timer_not_found",
            "No active timer found"
        )

    # Cancel in scheduler
    if timer.get("task_id"):
        scheduler.cancel_task(timer["task_id"])

    # Update database
    if conn:
        _update_timer_status(conn, timer["id"], "cancelled")

    timer_label = timer.get("label")
    if timer_label:
        display = f"Cancelled {timer_label} timer"
        speech = f"Cancelled {timer_label} timer"
    else:
        display = "Timer cancelled"
        speech = "Timer cancelled"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        timer_id=timer["id"],
    )


def handle_timer_status(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle timer_status command.

    Args in query:
        timer_id: Specific timer ID (optional)
        label: Timer label to match (optional)
    """
    timer_id = query.args.get("timer_id")
    label = query.args.get("label")

    # Find specific timer or get all
    if timer_id and conn:
        timer = _get_timer_by_id(conn, timer_id)
        timers = [timer] if timer else []
    elif label and conn:
        timer = _find_timer_by_label(conn, label)
        timers = [timer] if timer else []
    elif conn:
        timers = _get_active_timers(conn)
    else:
        # No database, get from scheduler
        tasks = scheduler.list_pending(TaskType.TIMER)
        timers = [{"task_id": t.id, "label": t.label} for t in tasks]

    # Build response, filtering out expired timers
    lines = []
    timer_data = []
    expired_ids = []

    for timer in timers:
        task_id = timer.get("task_id")
        remaining = scheduler.get_timer_remaining(task_id) if task_id else None

        if remaining is None and timer.get("expires_ts"):
            # Calculate from database
            expires = datetime.fromtimestamp(timer["expires_ts"])
            remaining = max(0, int((expires - datetime.now()).total_seconds()))

        status = timer.get("status", "active")
        timer_label = timer.get("label")

        # Skip expired timers and clean them up
        if status != "paused" and (remaining is None or remaining <= 0):
            if conn and timer.get("id"):
                expired_ids.append(timer["id"])
            continue

        if status == "paused":
            time_str = f"paused ({_format_time_remaining(remaining or 0)} remaining)"
        else:
            time_str = f"{_format_time_remaining(remaining)} remaining"

        if timer_label:
            lines.append(f"  {timer_label}: {time_str}")
        else:
            lines.append(f"  Timer: {time_str}")

        timer_data.append({
            "id": timer.get("id"),
            "label": timer_label,
            "remaining_s": remaining,
            "status": status,
        })

    # Clean up expired timers in DB
    for timer_id in expired_ids:
        _update_timer_status(conn, timer_id, "expired")

    if not timer_data:
        return UtilityResult.ok(
            display="No active timers",
            speech="No active timers",
            timers=[],
        )

    if len(timer_data) == 1:
        entry = timer_data[0]
        remaining = entry["remaining_s"]
        timer_label = entry["label"]

        if entry["status"] == "paused":
            if timer_label:
                speech = f"{timer_label} timer paused with {_format_time_remaining(remaining or 0)} remaining"
            else:
                speech = f"Timer paused with {_format_time_remaining(remaining or 0)} remaining"
        else:
            if timer_label:
                speech = f"{timer_label} timer has {_format_time_remaining(remaining)} remaining"
            else:
                speech = f"{_format_time_remaining(remaining)} remaining"
    else:
        speech = f"You have {len(timer_data)} active timer{'s' if len(timer_data) != 1 else ''}"

    display = "Timers:\n" + "\n".join(lines)

    return UtilityResult.ok(
        display=display,
        speech=speech,
        timers=timer_data,
    )


def handle_timer_pause(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle timer_pause command.

    Args in query:
        timer_id: Specific timer ID (optional)
        label: Timer label to match (optional)
    """
    timer_id = query.args.get("timer_id")
    label = query.args.get("label")

    # Find timer
    timer = None

    if timer_id and conn:
        timer = _get_timer_by_id(conn, timer_id)
    elif label and conn:
        timer = _find_timer_by_label(conn, label)
    elif conn:
        timers = _get_active_timers(conn)
        active = [t for t in timers if t["status"] == "active"]
        if active:
            timer = active[-1]

    if timer is None:
        return UtilityResult.error("timer_not_found", "No active timer found")

    if timer.get("status") == "paused":
        return UtilityResult.error("already_paused", "Timer is already paused")

    # Pause in scheduler
    task_id = timer.get("task_id")
    if task_id:
        success = scheduler.pause_timer(task_id)
        if not success:
            return UtilityResult.error("pause_failed", "Could not pause timer")

    # Update database
    if conn:
        _update_timer_status(conn, timer["id"], "paused")

    timer_label = timer.get("label")
    if timer_label:
        display = f"Paused {timer_label} timer"
        speech = f"Paused {timer_label} timer"
    else:
        display = "Timer paused"
        speech = "Timer paused"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        timer_id=timer["id"],
    )


def handle_timer_resume(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle timer_resume command.

    Args in query:
        timer_id: Specific timer ID (optional)
        label: Timer label to match (optional)
    """
    timer_id = query.args.get("timer_id")
    label = query.args.get("label")

    # Find timer
    timer = None

    if timer_id and conn:
        timer = _get_timer_by_id(conn, timer_id)
    elif label and conn:
        timer = _find_timer_by_label(conn, label)
    elif conn:
        timers = _get_active_timers(conn)
        paused = [t for t in timers if t["status"] == "paused"]
        if paused:
            timer = paused[-1]

    if timer is None:
        return UtilityResult.error("timer_not_found", "No paused timer found")

    if timer.get("status") != "paused":
        return UtilityResult.error("not_paused", "Timer is not paused")

    # Resume in scheduler
    task_id = timer.get("task_id")
    if task_id:
        success = scheduler.resume_timer(task_id)
        if not success:
            return UtilityResult.error("resume_failed", "Could not resume timer")

    # Update database
    if conn:
        _update_timer_status(conn, timer["id"], "active")

    timer_label = timer.get("label")
    if timer_label:
        display = f"Resumed {timer_label} timer"
        speech = f"Resumed {timer_label} timer"
    else:
        display = "Timer resumed"
        speech = "Timer resumed"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        timer_id=timer["id"],
    )


def handle_timer_list(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle timer_list command.

    Lists all active and paused timers.
    """
    # Delegate to status handler with no filters
    return handle_timer_status(query, scheduler, conn)


# Command routing for timer category
TIMER_HANDLERS = {
    "timer_set": handle_timer_set,
    "timer_cancel": handle_timer_cancel,
    "timer_status": handle_timer_status,
    "timer_pause": handle_timer_pause,
    "timer_resume": handle_timer_resume,
    "timer_list": handle_timer_list,
}


def dispatch_timer_command(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    audio_player=None,
) -> UtilityResult:
    """Dispatch a timer category command to the appropriate handler."""
    handler = TIMER_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown timer command: {query.command}"
        )

    # Different handlers have different signatures
    if query.command == "timer_set":
        return handler(query, scheduler, conn, user_tz, audio_player)
    elif query.command in ("timer_cancel", "timer_status", "timer_pause", "timer_resume", "timer_list"):
        return handler(query, scheduler, conn)

    return handler(query, scheduler, conn)
