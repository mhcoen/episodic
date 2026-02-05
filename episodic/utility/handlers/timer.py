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
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

from ..types import UtilityQuery, UtilityResult
from ..scheduler import (
    Scheduler,
    ScheduledTask,
    TaskType,
    TaskStatus,
    TaskResult,
    create_timer_task,
)


def _format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"

    minutes = seconds // 60
    secs = seconds % 60

    if minutes < 60:
        if secs:
            return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} second{'s' if secs != 1 else ''}"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"

    hours = minutes // 60
    mins = minutes % 60

    if hours < 24:
        parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
        if mins:
            parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
        return " ".join(parts)

    days = hours // 24
    hrs = hours % 24

    parts = [f"{days} day{'s' if days != 1 else ''}"]
    if hrs:
        parts.append(f"{hrs} hour{'s' if hrs != 1 else ''}")
    return " ".join(parts)


def _format_time_remaining(seconds: int) -> str:
    """Format remaining time for display."""
    if seconds <= 0:
        return "done"

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

    if secs:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{hours}h {mins}m"
    return f"{hours}h"


def _persist_timer(
    conn: sqlite3.Connection,
    timer_id: str,
    duration_s: int,
    label: Optional[str],
    expires_at: datetime,
    task_id: str,
) -> None:
    """Persist timer to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO timers (id, duration_s, label, status, created_ts, expires_ts, task_id)
        VALUES (?, ?, ?, 'active', ?, ?, ?)
    """, (
        timer_id,
        duration_s,
        label,
        int(datetime.now().timestamp()),
        int(expires_at.timestamp()),
        task_id,
    ))
    conn.commit()


def _update_timer_status(conn: sqlite3.Connection, timer_id: str, status: str) -> None:
    """Update timer status in database."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE timers SET status = ? WHERE id = ?",
        (status, timer_id)
    )
    conn.commit()


def _get_active_timers(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all active timers from database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, duration_s, label, status, created_ts, expires_ts, task_id
        FROM timers
        WHERE status IN ('active', 'paused')
        ORDER BY expires_ts
    """)

    timers = []
    for row in cursor.fetchall():
        timers.append({
            "id": row[0],
            "duration_s": row[1],
            "label": row[2],
            "status": row[3],
            "created_ts": row[4],
            "expires_ts": row[5],
            "task_id": row[6],
        })

    return timers


def _get_timer_by_id(conn: sqlite3.Connection, timer_id: str) -> Optional[Dict[str, Any]]:
    """Get timer by ID."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, duration_s, label, status, created_ts, expires_ts, task_id
        FROM timers
        WHERE id = ?
    """, (timer_id,))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "duration_s": row[1],
        "label": row[2],
        "status": row[3],
        "created_ts": row[4],
        "expires_ts": row[5],
        "task_id": row[6],
    }


def _find_timer_by_label(conn: sqlite3.Connection, label: str) -> Optional[Dict[str, Any]]:
    """Find active timer by label."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, duration_s, label, status, created_ts, expires_ts, task_id
        FROM timers
        WHERE label LIKE ? AND status IN ('active', 'paused')
        ORDER BY created_ts DESC
        LIMIT 1
    """, (f"%{label}%",))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "duration_s": row[1],
        "label": row[2],
        "status": row[3],
        "created_ts": row[4],
        "expires_ts": row[5],
        "task_id": row[6],
    }


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
        # Update timer status
        if conn:
            _update_timer_status(conn, timer_id, "expired")

        # Play sound
        if audio_player:
            audio_player.play_timer(label)

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

    if not timers:
        return UtilityResult.ok(
            display="No active timers",
            speech="No active timers",
            timers=[],
        )

    # Build response
    lines = []
    timer_data = []

    for timer in timers:
        task_id = timer.get("task_id")
        remaining = scheduler.get_timer_remaining(task_id) if task_id else None

        if remaining is None and timer.get("expires_ts"):
            # Calculate from database
            expires = datetime.fromtimestamp(timer["expires_ts"])
            remaining = max(0, int((expires - datetime.now()).total_seconds()))

        status = timer.get("status", "active")
        timer_label = timer.get("label")

        if status == "paused":
            time_str = f"paused ({_format_time_remaining(remaining or 0)} remaining)"
        elif remaining is not None and remaining > 0:
            time_str = _format_time_remaining(remaining)
        else:
            time_str = "done"

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

    if len(timers) == 1:
        timer = timers[0]
        remaining = timer_data[0]["remaining_s"]
        timer_label = timer.get("label")

        if timer_data[0]["status"] == "paused":
            if timer_label:
                speech = f"{timer_label} timer paused with {_format_time_remaining(remaining or 0)} remaining"
            else:
                speech = f"Timer paused with {_format_time_remaining(remaining or 0)} remaining"
        elif remaining and remaining > 0:
            if timer_label:
                speech = f"{timer_label} timer has {_format_time_remaining(remaining)} remaining"
            else:
                speech = f"{_format_time_remaining(remaining)} remaining"
        else:
            speech = "Timer done"
    else:
        speech = f"You have {len(timers)} active timer{'s' if len(timers) != 1 else ''}"

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
