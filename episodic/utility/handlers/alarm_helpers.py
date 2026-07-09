"""Alarm formatting, time parsing, and persistence helpers.

Split out of handlers/alarm.py to keep it under the size limit. Re-imported
there so the handlers resolve unchanged.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo


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


