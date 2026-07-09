"""Timer formatting and persistence helpers.

Split out of handlers/timer.py to keep it under the size limit. Re-imported
there so the handlers (and cli_integration's _update_timer_status import)
resolve unchanged.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any


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


