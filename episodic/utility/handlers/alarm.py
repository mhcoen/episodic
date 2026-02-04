"""
Alarm Handler.

Handles alarm-related utility commands:
- alarm_set: Create a new alarm at a specific time
- alarm_cancel: Cancel an alarm
- alarm_list: List all alarms
- alarm_snooze: Snooze a firing alarm
"""

import uuid
import sqlite3
from datetime import datetime, timedelta, time as dt_time
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

from ..types import UtilityQuery, UtilityResult
from ..scheduler import (
    Scheduler,
    ScheduledTask,
    TaskType,
    TaskStatus,
    TaskResult,
    create_alarm_task,
)


def _format_time(dt: datetime) -> str:
    """Format time for display."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _format_date(dt: datetime) -> str:
    """Format date for display."""
    today = datetime.now(dt.tzinfo).date() if dt.tzinfo else datetime.now().date()
    alarm_date = dt.date()

    if alarm_date == today:
        return "Today"
    elif alarm_date == today + timedelta(days=1):
        return "Tomorrow"
    else:
        return dt.strftime("%A, %B %d")


def _parse_alarm_time(
    time_str: Optional[str],
    hour: Optional[int],
    minute: Optional[int],
    user_tz: str,
) -> Optional[datetime]:
    """
    Parse alarm time from various formats.

    Returns datetime with timezone.
    """
    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    if hour is not None:
        # Hour (and optionally minute) specified directly
        minute = minute or 0

        # Handle 24-hour format
        if hour > 23:
            return None

        alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # If time has passed today, schedule for tomorrow
        if alarm_time <= now:
            alarm_time += timedelta(days=1)

        return alarm_time

    if time_str:
        # Parse time string
        time_str = time_str.lower().strip()

        try:
            # Handle "X:XX" format (7:00am, 19:00)
            if ":" in time_str:
                parts = time_str.replace("am", "").replace("pm", "").strip().split(":")
                h = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 else 0
            else:
                # Handle "Xam/pm" format (7am, 7pm)
                import re
                match = re.match(r'^(\d{1,2})\s*(am|pm)?$', time_str)
                if match:
                    h = int(match.group(1))
                    m = 0
                    # If no am/pm and hour > 12, assume 24-hour format
                else:
                    return None

            # Handle AM/PM
            if "pm" in time_str and h < 12:
                h += 12
            elif "am" in time_str and h == 12:
                h = 0

            if h > 23 or m > 59:
                return None

            alarm_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if alarm_time <= now:
                alarm_time += timedelta(days=1)
            return alarm_time
        except (ValueError, IndexError):
            return None

    return None


def _persist_alarm(
    conn: sqlite3.Connection,
    alarm_id: str,
    alarm_time: datetime,
    label: Optional[str],
    task_id: str,
    recurrence: Optional[str] = None,
    dnd_override: bool = False,
) -> None:
    """Persist alarm to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alarms (id, time, label, enabled, rrule, dnd_override, task_id)
        VALUES (?, ?, ?, 1, ?, ?, ?)
    """, (
        alarm_id,
        alarm_time.strftime("%H:%M"),
        label,
        recurrence,
        1 if dnd_override else 0,
        task_id,
    ))
    conn.commit()


def _update_alarm_enabled(conn: sqlite3.Connection, alarm_id: str, enabled: bool) -> None:
    """Update alarm enabled status in database."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE alarms SET enabled = ? WHERE id = ?",
        (1 if enabled else 0, alarm_id)
    )
    conn.commit()


def _delete_alarm(conn: sqlite3.Connection, alarm_id: str) -> None:
    """Delete alarm from database."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alarms WHERE id = ?", (alarm_id,))
    conn.commit()


def _get_alarms(conn: sqlite3.Connection, enabled_only: bool = True) -> List[Dict[str, Any]]:
    """Get alarms from database."""
    cursor = conn.cursor()

    if enabled_only:
        cursor.execute("""
            SELECT id, time, label, enabled, rrule, dnd_override, task_id
            FROM alarms
            WHERE enabled = 1
            ORDER BY time
        """)
    else:
        cursor.execute("""
            SELECT id, time, label, enabled, rrule, dnd_override, task_id
            FROM alarms
            ORDER BY time
        """)

    alarms = []
    for row in cursor.fetchall():
        alarms.append({
            "id": row[0],
            "time": row[1],
            "label": row[2],
            "enabled": bool(row[3]),
            "rrule": row[4],
            "dnd_override": bool(row[5]),
            "task_id": row[6],
        })

    return alarms


def _get_alarm_by_id(conn: sqlite3.Connection, alarm_id: str) -> Optional[Dict[str, Any]]:
    """Get alarm by ID."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, time, label, enabled, rrule, dnd_override, task_id
        FROM alarms
        WHERE id = ?
    """, (alarm_id,))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "time": row[1],
        "label": row[2],
        "enabled": bool(row[3]),
        "rrule": row[4],
        "dnd_override": bool(row[5]),
        "task_id": row[6],
    }


def _find_alarm_by_label(conn: sqlite3.Connection, label: str) -> Optional[Dict[str, Any]]:
    """Find alarm by label."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, time, label, enabled, rrule, dnd_override, task_id
        FROM alarms
        WHERE label LIKE ? AND enabled = 1
        ORDER BY time
        LIMIT 1
    """, (f"%{label}%",))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "time": row[1],
        "label": row[2],
        "enabled": bool(row[3]),
        "rrule": row[4],
        "dnd_override": bool(row[5]),
        "task_id": row[6],
    }


def _find_alarm_by_time(conn: sqlite3.Connection, time_str: str) -> Optional[Dict[str, Any]]:
    """Find alarm by time."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, time, label, enabled, rrule, dnd_override, task_id
        FROM alarms
        WHERE time = ? AND enabled = 1
        LIMIT 1
    """, (time_str,))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "time": row[1],
        "label": row[2],
        "enabled": bool(row[3]),
        "rrule": row[4],
        "dnd_override": bool(row[5]),
        "task_id": row[6],
    }


def handle_alarm_set(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    audio_player=None,
) -> UtilityResult:
    """
    Handle alarm_set command.

    Args in query:
        time: Time string (e.g., "7:00 AM")
        hour: Hour (0-23)
        minute: Minute (0-59)
        label: Optional alarm label
        recurrence: Optional RRULE string
        dnd_override: Whether alarm can break through DND
    """
    time_str = query.args.get("time")
    hour = query.args.get("hour")
    minute = query.args.get("minute")
    label = query.args.get("label")
    recurrence = query.args.get("recurrence")
    dnd_override = query.args.get("dnd_override", False)

    # Parse alarm time
    alarm_time = _parse_alarm_time(time_str, hour, minute, user_tz)

    if alarm_time is None:
        return UtilityResult.error(
            "invalid_time",
            "Could not parse alarm time"
        )

    # Generate IDs
    alarm_id = str(uuid.uuid4())

    # Create callback for when alarm fires
    def alarm_callback() -> TaskResult:
        # Play alarm sound
        if audio_player:
            audio_player.play_alarm(label)

        return TaskResult(
            status=TaskStatus.COMPLETED,
            output=label if label else "Alarm",
            side_effects=["alarm_fired", alarm_id],
        )

    # Create task
    task = create_alarm_task(
        alarm_time=alarm_time,
        label=label,
        callback=alarm_callback,
        reference_id=alarm_id,
        dnd_override=dnd_override,
        recurrence=recurrence,
        user_tz=user_tz,
    )

    # Persist alarm
    if conn:
        _persist_alarm(conn, alarm_id, alarm_time, label, task.id, recurrence, dnd_override)

    # Add to scheduler
    scheduler.add_task(task)

    # Format response
    time_display = _format_time(alarm_time)
    date_display = _format_date(alarm_time)

    if label:
        display = f"Alarm set: {label} at {time_display} ({date_display})"
        speech = f"{label} alarm set for {time_display}"
    else:
        display = f"Alarm set for {time_display} ({date_display})"
        speech = f"Alarm set for {time_display}"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        alarm_id=alarm_id,
        task_id=task.id,
        time=alarm_time.isoformat(),
        label=label,
    )


def handle_alarm_cancel(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle alarm_cancel command.

    Args in query:
        alarm_id: Specific alarm ID (optional)
        label: Alarm label to match (optional)
        time: Time to match (optional)
        all: Cancel all alarms (optional)
    """
    alarm_id = query.args.get("alarm_id")
    label = query.args.get("label")
    time_str = query.args.get("time")
    cancel_all = query.args.get("all", False)

    if cancel_all:
        # Cancel all alarms
        count = scheduler.cancel_by_type(TaskType.ALARM)

        if conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alarms WHERE enabled = 1")
            conn.commit()

        if count == 0:
            return UtilityResult.ok(
                display="No active alarms to cancel",
                speech="No active alarms",
            )

        return UtilityResult.ok(
            display=f"Cancelled {count} alarm{'s' if count != 1 else ''}",
            speech=f"Cancelled {count} alarm{'s' if count != 1 else ''}",
            cancelled_count=count,
        )

    # Find specific alarm
    alarm = None

    if alarm_id and conn:
        alarm = _get_alarm_by_id(conn, alarm_id)
    elif label and conn:
        alarm = _find_alarm_by_label(conn, label)
    elif time_str and conn:
        # Parse time and find matching alarm
        alarm_time = _parse_alarm_time(time_str, None, None, user_tz)
        if alarm_time:
            time_key = alarm_time.strftime("%H:%M")
            alarm = _find_alarm_by_time(conn, time_key)
    elif conn:
        # Cancel next alarm
        alarms = _get_alarms(conn, enabled_only=True)
        if alarms:
            alarm = alarms[0]

    if alarm is None:
        return UtilityResult.error(
            "alarm_not_found",
            "No active alarm found"
        )

    # Cancel in scheduler
    if alarm.get("task_id"):
        scheduler.cancel_task(alarm["task_id"])

    # Delete from database
    if conn:
        _delete_alarm(conn, alarm["id"])

    alarm_label = alarm.get("label")
    alarm_time = alarm.get("time")

    if alarm_label:
        display = f"Cancelled {alarm_label} alarm"
        speech = f"Cancelled {alarm_label} alarm"
    elif alarm_time:
        display = f"Cancelled {alarm_time} alarm"
        speech = f"Cancelled alarm"
    else:
        display = "Alarm cancelled"
        speech = "Alarm cancelled"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        alarm_id=alarm["id"],
    )


def handle_alarm_list(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle alarm_list command.

    Lists all enabled alarms.
    """
    if conn:
        alarms = _get_alarms(conn, enabled_only=True)
    else:
        # Get from scheduler
        tasks = scheduler.list_pending(TaskType.ALARM)
        alarms = [{"time": t.next_run_wall.strftime("%H:%M"), "label": t.label} for t in tasks]

    if not alarms:
        return UtilityResult.ok(
            display="No alarms set",
            speech="No alarms set",
            alarms=[],
        )

    # Build response
    lines = []
    alarm_data = []

    for alarm in alarms:
        alarm_time = alarm.get("time", "")
        alarm_label = alarm.get("label")
        recurrence = alarm.get("rrule")

        # Parse time for better display
        try:
            h, m = map(int, alarm_time.split(":"))
            dt = datetime.now().replace(hour=h, minute=m)
            time_display = _format_time(dt)
        except (ValueError, AttributeError):
            time_display = alarm_time

        if alarm_label:
            line = f"  {time_display}: {alarm_label}"
        else:
            line = f"  {time_display}"

        if recurrence:
            line += " (recurring)"

        lines.append(line)

        alarm_data.append({
            "id": alarm.get("id"),
            "time": alarm_time,
            "label": alarm_label,
            "recurring": bool(recurrence),
        })

    if len(alarms) == 1:
        alarm = alarms[0]
        alarm_label = alarm.get("label")
        alarm_time = alarm.get("time", "")

        try:
            h, m = map(int, alarm_time.split(":"))
            dt = datetime.now().replace(hour=h, minute=m)
            time_display = _format_time(dt)
        except (ValueError, AttributeError):
            time_display = alarm_time

        if alarm_label:
            speech = f"You have one alarm: {alarm_label} at {time_display}"
        else:
            speech = f"You have one alarm at {time_display}"
    else:
        speech = f"You have {len(alarms)} alarms"

    display = "Alarms:\n" + "\n".join(lines)

    return UtilityResult.ok(
        display=display,
        speech=speech,
        alarms=alarm_data,
    )


def handle_alarm_snooze(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    audio_player=None,
) -> UtilityResult:
    """
    Handle alarm_snooze command.

    Args in query:
        duration_m: Snooze duration in minutes (default 9)
        alarm_id: Specific alarm to snooze (optional)
    """
    duration_m = query.args.get("duration_m", 9)  # Default 9 minutes
    alarm_id = query.args.get("alarm_id")

    # Stop any playing alarm sound
    if audio_player:
        audio_player.stop()

    # Find alarm to snooze
    alarm = None

    if alarm_id and conn:
        alarm = _get_alarm_by_id(conn, alarm_id)
    elif conn:
        # Find most recently fired alarm (for now, just the first enabled one)
        alarms = _get_alarms(conn, enabled_only=True)
        if alarms:
            alarm = alarms[0]

    if alarm is None:
        # Just snooze without tracking
        tz = ZoneInfo(user_tz)
        snooze_time = datetime.now(tz) + timedelta(minutes=duration_m)

        def snooze_callback() -> TaskResult:
            if audio_player:
                audio_player.play_alarm()
            return TaskResult(status=TaskStatus.COMPLETED, output="Alarm")

        task = create_alarm_task(
            alarm_time=snooze_time,
            label="Snooze",
            callback=snooze_callback,
            user_tz=user_tz,
        )

        scheduler.add_task(task)

        display = f"Snoozed for {duration_m} minutes"
        speech = f"Snoozed for {duration_m} minutes"

        return UtilityResult.ok(
            display=display,
            speech=speech,
            snooze_until=snooze_time.isoformat(),
        )

    # Create snooze alarm
    tz = ZoneInfo(user_tz)
    snooze_time = datetime.now(tz) + timedelta(minutes=duration_m)
    alarm_label = alarm.get("label")

    def snooze_callback() -> TaskResult:
        if audio_player:
            audio_player.play_alarm(alarm_label)
        return TaskResult(
            status=TaskStatus.COMPLETED,
            output=alarm_label or "Alarm",
            side_effects=["alarm_fired", alarm["id"]],
        )

    task = create_alarm_task(
        alarm_time=snooze_time,
        label=alarm_label,
        callback=snooze_callback,
        reference_id=alarm["id"],
        dnd_override=alarm.get("dnd_override", False),
        user_tz=user_tz,
    )

    scheduler.add_task(task)

    if alarm_label:
        display = f"Snoozed {alarm_label} for {duration_m} minutes"
        speech = f"Snoozed {alarm_label} for {duration_m} minutes"
    else:
        display = f"Snoozed for {duration_m} minutes"
        speech = f"Snoozed for {duration_m} minutes"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        alarm_id=alarm["id"],
        snooze_until=snooze_time.isoformat(),
    )


# Command routing for alarm category
ALARM_HANDLERS = {
    "alarm_set": handle_alarm_set,
    "alarm_cancel": handle_alarm_cancel,
    "alarm_list": handle_alarm_list,
    "alarm_snooze": handle_alarm_snooze,
}


def dispatch_alarm_command(
    query: UtilityQuery,
    scheduler: Scheduler,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    audio_player=None,
) -> UtilityResult:
    """Dispatch an alarm category command to the appropriate handler."""
    handler = ALARM_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown alarm command: {query.command}"
        )

    # Different handlers have different signatures
    if query.command in ("alarm_set", "alarm_snooze"):
        return handler(query, scheduler, conn, user_tz, audio_player)
    elif query.command in ("alarm_cancel", "alarm_list"):
        return handler(query, scheduler, conn, user_tz)

    return handler(query, scheduler, conn)
