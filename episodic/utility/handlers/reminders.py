"""
Reminders Handler.

Handles reminder-related utility commands:
- remind_set: Create a reminder at a specific time
- remind_list: List pending reminders
- remind_cancel: Cancel a reminder
"""

import uuid
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

from ..types import UtilityQuery, UtilityResult
from ..scheduler import (
    Scheduler,
    TaskType,
    TaskStatus,
    TaskResult,
    create_reminder_task,
)


def _persist_reminder(
    conn: sqlite3.Connection,
    reminder_id: str,
    text: str,
    due_at: datetime,
    rrule: Optional[str] = None,
) -> None:
    """Insert reminder into database."""
    now = int(time.time())
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reminders (id, text, due_at, rrule, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, 1, ?, ?)
    """, (
        reminder_id,
        text,
        int(due_at.timestamp()),
        rrule,
        now,
        now,
    ))
    conn.commit()


def _get_pending_reminders(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Get pending reminders from database."""
    now = int(time.time())
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, text, due_at, rrule, enabled
        FROM reminders
        WHERE enabled = 1 AND due_at >= ?
        ORDER BY due_at
        LIMIT ?
    """, (now, limit))

    reminders = []
    for row in cursor.fetchall():
        reminders.append({
            "id": row[0],
            "text": row[1],
            "due_at": row[2],
            "rrule": row[3],
            "enabled": bool(row[4]),
        })

    return reminders


def _get_reminder_by_id(
    conn: sqlite3.Connection,
    reminder_id: str,
) -> Optional[Dict[str, Any]]:
    """Get reminder by ID."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, text, due_at, rrule, enabled
        FROM reminders
        WHERE id = ?
    """, (reminder_id,))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "text": row[1],
        "due_at": row[2],
        "rrule": row[3],
        "enabled": bool(row[4]),
    }


def _find_reminder_by_text(
    conn: sqlite3.Connection,
    text: str,
) -> Optional[Dict[str, Any]]:
    """Find reminder by text match."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, text, due_at, rrule, enabled
        FROM reminders
        WHERE text LIKE ? AND enabled = 1
        ORDER BY due_at
        LIMIT 1
    """, (f"%{text}%",))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "text": row[1],
        "due_at": row[2],
        "rrule": row[3],
        "enabled": bool(row[4]),
    }


def _disable_reminder(conn: sqlite3.Connection, reminder_id: str) -> bool:
    """Disable reminder by ID."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE reminders SET enabled = 0, updated_at = ? WHERE id = ?",
        (int(time.time()), reminder_id)
    )
    conn.commit()
    return cursor.rowcount > 0


def _parse_reminder_time(
    time_str: Optional[str],
    minutes: Optional[int],
    hours: Optional[int],
    at_time: Optional[str],
    user_tz: str,
) -> Optional[datetime]:
    """
    Parse reminder time from various formats.

    Supports:
    - minutes: "in X minutes"
    - hours: "in X hours"
    - at_time: "at 3pm", "at 15:30"
    - time_str: Natural language (simplified)
    """
    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    if minutes is not None:
        return now + timedelta(minutes=minutes)

    if hours is not None:
        return now + timedelta(hours=hours)

    if at_time:
        at_time = at_time.lower().strip()

        if ":" in at_time:
            parts = at_time.replace("am", "").replace("pm", "").strip().split(":")
            try:
                h = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 else 0

                if "pm" in at_time and h < 12:
                    h += 12
                elif "am" in at_time and h == 12:
                    h = 0

                remind_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if remind_time <= now:
                    remind_time += timedelta(days=1)
                return remind_time
            except (ValueError, IndexError):
                pass

        # Try just hour with am/pm
        try:
            clean = at_time.replace("am", "").replace("pm", "").strip()
            h = int(clean)
            if "pm" in at_time and h < 12:
                h += 12
            elif "am" in at_time and h == 12:
                h = 0

            remind_time = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if remind_time <= now:
                remind_time += timedelta(days=1)
            return remind_time
        except ValueError:
            pass

    if time_str:
        time_str = time_str.lower().strip()

        # "in X minutes/hours"
        if "minute" in time_str:
            try:
                parts = time_str.split()
                for i, p in enumerate(parts):
                    if p.isdigit():
                        return now + timedelta(minutes=int(p))
            except (ValueError, IndexError):
                pass

        if "hour" in time_str:
            try:
                parts = time_str.split()
                for i, p in enumerate(parts):
                    if p.isdigit():
                        return now + timedelta(hours=int(p))
            except (ValueError, IndexError):
                pass

        # "tomorrow"
        if "tomorrow" in time_str:
            return now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)

    return None


def _format_reminder_time(due_at: int, user_tz: str = "America/Chicago") -> str:
    """Format reminder time for display."""
    tz = ZoneInfo(user_tz)
    dt = datetime.fromtimestamp(due_at, tz=tz)
    now = datetime.now(tz)

    time_str = dt.strftime("%I:%M %p").lstrip("0")

    if dt.date() == now.date():
        return f"Today at {time_str}"
    elif dt.date() == (now + timedelta(days=1)).date():
        return f"Tomorrow at {time_str}"
    else:
        return dt.strftime("%A at ") + time_str


def handle_remind_set(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    tts_engine=None,
) -> UtilityResult:
    """
    Handle remind_set command.

    Args in query:
        text: Reminder message
        minutes: Minutes from now (optional)
        hours: Hours from now (optional)
        at_time: Specific time (optional)
        time_str: Natural language time (optional)
        rrule: Recurrence rule (optional)
    """
    text = query.args.get("text", "").strip()

    if not text:
        return UtilityResult.error("missing_text", "No reminder text provided")

    # Parse time
    due_at = _parse_reminder_time(
        time_str=query.args.get("time_str"),
        minutes=query.args.get("minutes"),
        hours=query.args.get("hours"),
        at_time=query.args.get("at_time"),
        user_tz=user_tz,
    )

    if due_at is None:
        return UtilityResult.error("invalid_time", "Could not parse reminder time")

    rrule = query.args.get("rrule")

    # Generate ID
    reminder_id = str(uuid.uuid4())

    # Create callback for when reminder fires
    def reminder_callback() -> TaskResult:
        # Update reminder status with a fresh connection
        # (the original conn is closed by the time the reminder fires)
        try:
            from ...db_connection import get_connection
            with get_connection() as fresh_conn:
                _disable_reminder(fresh_conn, reminder_id)
        except Exception:
            pass

        # Speak reminder
        if tts_engine:
            try:
                tts_engine.speak(f"Reminder: {text}")
            except Exception:
                pass

        return TaskResult(
            status=TaskStatus.COMPLETED,
            output=f"Reminder: {text}",
            side_effects=["reminder_fired", reminder_id],
        )

    # Create task
    task = create_reminder_task(
        remind_time=due_at,
        text=text,
        callback=reminder_callback,
        reference_id=reminder_id,
        user_tz=user_tz,
    )

    # Persist reminder
    if conn:
        _persist_reminder(conn, reminder_id, text, due_at, rrule)

    # Add to scheduler
    scheduler.add_task(task)

    # Format response
    time_display = _format_reminder_time(int(due_at.timestamp()), user_tz)

    return UtilityResult.ok(
        display=f"Reminder set: {text} ({time_display})",
        speech=f"I'll remind you {time_display.lower()}",
        _command="reminder_set",
        reminder_id=reminder_id,
        text=text,
        due_at=due_at.isoformat(),
    )


def handle_remind_list(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle remind_list command.

    Lists pending reminders.
    """
    if conn:
        reminders = _get_pending_reminders(conn, limit=10)
    else:
        # Get from scheduler
        tasks = scheduler.list_pending(TaskType.REMINDER)
        reminders = [{
            "text": t.label,
            "due_at": int(t.next_run_wall.timestamp()),
        } for t in tasks]

    if not reminders:
        return UtilityResult.ok(
            display="No pending reminders",
            speech="No pending reminders",
            reminders=[],
        )

    # Build display
    lines = []
    for reminder in reminders:
        text = reminder.get("text", "")[:40]
        time_str = _format_reminder_time(reminder["due_at"], user_tz)
        lines.append(f"  {text}: {time_str}")

    display = "Reminders:\n" + "\n".join(lines)

    if len(reminders) == 1:
        r = reminders[0]
        speech = f"You have one reminder: {r.get('text', '')[:30]}"
    else:
        speech = f"You have {len(reminders)} reminders"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        reminders=[{
            "id": r.get("id"),
            "text": r.get("text"),
            "due_at": r["due_at"],
        } for r in reminders],
    )


def handle_remind_cancel(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle remind_cancel command.

    Args in query:
        reminder_id: Specific reminder ID (optional)
        text: Text to match (optional)
        all: Cancel all reminders (optional)
    """
    reminder_id = query.args.get("reminder_id")
    text = query.args.get("text")
    cancel_all = query.args.get("all", False)

    if cancel_all:
        # Cancel all reminders
        count = scheduler.cancel_by_type(TaskType.REMINDER)

        if conn:
            now = int(time.time())
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE reminders SET enabled = 0, updated_at = ? WHERE enabled = 1",
                (now,)
            )
            conn.commit()

        if count == 0:
            return UtilityResult.ok(
                display="No reminders to cancel",
                speech="No reminders to cancel",
            )

        return UtilityResult.ok(
            display=f"Cancelled {count} reminder{'s' if count != 1 else ''}",
            speech=f"Cancelled {count} reminder{'s' if count != 1 else ''}",
            cancelled_count=count,
        )

    # Find specific reminder
    reminder = None

    if reminder_id and conn:
        reminder = _get_reminder_by_id(conn, reminder_id)
    elif text and conn:
        reminder = _find_reminder_by_text(conn, text)
    elif conn:
        # Cancel next reminder
        reminders = _get_pending_reminders(conn, limit=1)
        if reminders:
            reminder = reminders[0]

    if reminder is None:
        return UtilityResult.error("reminder_not_found", "No reminder found")

    # Cancel in scheduler
    scheduler.cancel_by_reference(reminder["id"])

    # Disable in database
    if conn:
        _disable_reminder(conn, reminder["id"])

    reminder_text = reminder.get("text", "")[:30]
    display = f"Cancelled reminder: {reminder_text}"
    speech = "Reminder cancelled"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        reminder_id=reminder["id"],
    )


# Command routing for reminder category
REMINDER_HANDLERS = {
    "remind_set": handle_remind_set,
    "remind_list": handle_remind_list,
    "remind_cancel": handle_remind_cancel,
}


def dispatch_reminder_command(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    tts_engine=None,
) -> UtilityResult:
    """Dispatch a reminder category command to the appropriate handler."""
    handler = REMINDER_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown reminder command: {query.command}"
        )

    if query.command == "remind_set":
        return handler(query, scheduler, conn, user_tz, tts_engine)
    elif query.command == "remind_list":
        return handler(query, scheduler, conn, user_tz)
    elif query.command == "remind_cancel":
        return handler(query, scheduler, conn)

    return handler(query, scheduler, conn)
