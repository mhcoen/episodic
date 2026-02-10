"""
Utility Commands Database Schema.

Creates and manages the SQLite tables for utility command persistence:
- preferences
- alarms, timers, reminders
- notes, lists
- routines
- event log, undo stack
- scheduled tasks, cache metadata
"""

import sqlite3
from typing import Optional

SCHEMA_VERSION = "1.2"

SCHEMA_SQL = """
-- Utility Commands SQLite Schema
-- Version: 1.0

-- ============================================================================
-- PREFERENCES
-- ============================================================================

CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at INTEGER NOT NULL  -- Unix timestamp
);

-- ============================================================================
-- ALARMS
-- ============================================================================

CREATE TABLE IF NOT EXISTS alarms (
    id TEXT PRIMARY KEY,
    time TEXT NOT NULL,             -- ISO format datetime
    label TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    rrule TEXT,                     -- iCalendar RRULE (NULL = one-shot)
    dnd_override INTEGER NOT NULL DEFAULT 0,
    task_id TEXT                    -- FK to scheduled_tasks
);

CREATE INDEX IF NOT EXISTS idx_alarms_enabled ON alarms(enabled);

-- ============================================================================
-- TIMERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS timers (
    id TEXT PRIMARY KEY,
    duration_s INTEGER NOT NULL,
    label TEXT,
    status TEXT NOT NULL,           -- "running" | "paused" | "expired" | "cancelled"
    created_ts INTEGER NOT NULL,    -- Unix timestamp when created
    expires_ts INTEGER NOT NULL,    -- Unix timestamp when expires
    task_id TEXT                    -- FK to scheduled_tasks
);

CREATE INDEX IF NOT EXISTS idx_timers_status ON timers(status);
CREATE INDEX IF NOT EXISTS idx_timers_expires ON timers(expires_ts);

-- ============================================================================
-- REMINDERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    due_at INTEGER NOT NULL,        -- Unix timestamp
    rrule TEXT,                     -- iCalendar RRULE (NULL = one-shot)
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reminders_enabled ON reminders(enabled);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at);

-- ============================================================================
-- NOTES
-- ============================================================================

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

-- ============================================================================
-- LISTS (shopping, todo, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS lists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,      -- "shopping", "todo", etc.
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS list_items (
    id TEXT PRIMARY KEY,
    list_id TEXT NOT NULL REFERENCES lists(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    checked INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_list_items_list ON list_items(list_id);

-- ============================================================================
-- ROUTINES
-- ============================================================================

CREATE TABLE IF NOT EXISTS routines (
    name TEXT PRIMARY KEY,          -- "good morning", "good night"
    steps_json TEXT NOT NULL,       -- Array of UtilityQuery objects
    enabled_flag INTEGER NOT NULL DEFAULT 1,
    updated_at INTEGER NOT NULL
);

-- ============================================================================
-- EVENT LOG (audit trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS utility_event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,            -- Unix timestamp
    source TEXT NOT NULL,           -- "cli" | "voice" | "scheduler" | "routine"
    category TEXT NOT NULL,
    command TEXT NOT NULL,
    args_json TEXT,
    result_status TEXT NOT NULL,    -- "ok" | "error"
    result_payload_json TEXT,
    error_type TEXT,                -- NULL if success
    error_message TEXT,             -- NULL if success
    latency_us INTEGER,
    side_effects_json TEXT          -- For undo support
);

CREATE INDEX IF NOT EXISTS idx_event_log_ts ON utility_event_log(ts);
CREATE INDEX IF NOT EXISTS idx_event_log_category ON utility_event_log(category);

-- ============================================================================
-- UNDO STACK
-- ============================================================================

CREATE TABLE IF NOT EXISTS undo_stack (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES utility_event_log(id),
    inverse_command_json TEXT NOT NULL,  -- UtilityQuery to reverse the action
    created_at INTEGER NOT NULL,
    executed INTEGER NOT NULL DEFAULT 0  -- 1 if undo was performed
);

-- ============================================================================
-- SCHEDULED TASKS (persistent across restart)
-- ============================================================================

CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,        -- "ALARM" | "TIMER" | "REMINDER" | "REFRESH"
    priority INTEGER NOT NULL DEFAULT 1,
    next_run_ts INTEGER NOT NULL,   -- Unix timestamp (wall clock)
    reference_id TEXT,              -- FK to alarms/timers/reminders (NULL for refresh)
    label TEXT,
    dnd_override INTEGER NOT NULL DEFAULT 0,
    duration_s INTEGER,             -- Original duration for timers
    paused_remaining REAL,          -- Remaining time when paused (seconds)
    recurrence_json TEXT            -- JSON: RRULE string or interval seconds
);

CREATE INDEX IF NOT EXISTS idx_scheduled_next ON scheduled_tasks(next_run_ts);

-- ============================================================================
-- CACHE METADATA (optional, for cache warming on restart)
-- ============================================================================

CREATE TABLE IF NOT EXISTS cache_metadata (
    cache_key TEXT PRIMARY KEY,     -- "weather:current", "traffic:home_work"
    fetched_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    provider TEXT NOT NULL
);

-- ============================================================================
-- SCHEMA VERSION
-- ============================================================================

CREATE TABLE IF NOT EXISTS utility_schema_version (
    version TEXT PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
"""

DEFAULT_PREFERENCES = {
    "user_tz": "America/Chicago",
    "temp_unit": "F",
    "location_home": None,
    "location_work": None,
    "location_detected": None,  # Auto-detected from IP geolocation
    "dnd_enabled": False,
    "dnd_start": "22:00",
    "dnd_end": "07:00",
    "confirm_mutations": False,
    "default_timer_sound": "default",
    "default_alarm_sound": "default",
    "default_media_source": "local",
}


def init_utility_schema(conn: sqlite3.Connection) -> bool:
    """
    Initialize the utility commands schema.

    Creates all tables if they don't exist.
    Returns True if schema was created, False if already existed.
    """
    import json
    import time

    cursor = conn.cursor()

    # Check if schema already exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='utility_schema_version'
    """)
    exists = cursor.fetchone() is not None

    if exists:
        # Check version
        cursor.execute("SELECT version FROM utility_schema_version LIMIT 1")
        row = cursor.fetchone()
        if row and row[0] == SCHEMA_VERSION:
            return False  # Already up to date

        # Version mismatch - drop utility tables to recreate with new schema
        # (This is acceptable for dev; production would use migrations)
        utility_tables = [
            "alarms", "timers", "reminders", "notes", "lists", "list_items",
            "routines", "utility_event_log", "undo_stack", "scheduled_tasks",
            "cache_metadata", "utility_schema_version"
        ]
        for table in utility_tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # Create schema
    cursor.executescript(SCHEMA_SQL)

    # Record version
    now = int(time.time())
    cursor.execute("""
        INSERT OR REPLACE INTO utility_schema_version (version, applied_at)
        VALUES (?, ?)
    """, (SCHEMA_VERSION, now))

    # Insert default preferences if not exist
    for key, value in DEFAULT_PREFERENCES.items():
        cursor.execute("""
            INSERT OR IGNORE INTO preferences (key, value_json, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), now))

    conn.commit()
    return True


def get_preference(conn: sqlite3.Connection, key: str) -> Optional[any]:
    """Get a preference value."""
    import json
    cursor = conn.cursor()
    cursor.execute("SELECT value_json FROM preferences WHERE key = ?", (key,))
    row = cursor.fetchone()
    if row:
        return json.loads(row[0])
    return None


def set_preference(conn: sqlite3.Connection, key: str, value: any) -> None:
    """Set a preference value."""
    import json
    import time
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO preferences (key, value_json, updated_at)
        VALUES (?, ?, ?)
    """, (key, json.dumps(value), int(time.time())))
    conn.commit()
