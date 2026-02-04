"""
Tests for System, Notes, and Reminders handlers.

Tests cover:
1. System controls (stop, cancel, status, DND)
2. Notes (add, list, search, delete)
3. Reminders (set, list, cancel)
"""

import pytest
import sqlite3
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from episodic.utility.types import UtilityQuery, ResultStatus
from episodic.utility.scheduler import Scheduler, TaskType
from episodic.utility.audio import NullAudioPlayer
from episodic.utility.handlers.system import (
    handle_stop,
    handle_cancel,
    handle_status,
    handle_dnd_on,
    handle_dnd_off,
    handle_undo,
    handle_repeat,
    dispatch_system_command,
)
from episodic.utility.handlers.notes import (
    handle_note_add,
    handle_note_list,
    handle_note_search,
    handle_note_delete,
    dispatch_note_command,
)
from episodic.utility.handlers.reminders import (
    handle_remind_set,
    handle_remind_list,
    handle_remind_cancel,
    dispatch_reminder_command,
)
from episodic.utility.dispatcher import create_utility_query


@pytest.fixture
def test_db():
    """Create in-memory test database with full schema."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE timers (
            id TEXT PRIMARY KEY,
            duration_s INTEGER NOT NULL,
            label TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_ts INTEGER NOT NULL,
            expires_ts INTEGER NOT NULL,
            task_id TEXT
        );

        CREATE TABLE alarms (
            id TEXT PRIMARY KEY,
            time TEXT NOT NULL,
            label TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            rrule TEXT,
            dnd_override INTEGER NOT NULL DEFAULT 0,
            task_id TEXT
        );

        CREATE TABLE notes (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE reminders (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            due_at INTEGER NOT NULL,
            rrule TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE undo_stack (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            inverse_command_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            executed INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE preferences (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            priority INTEGER NOT NULL,
            next_run_ts INTEGER NOT NULL,
            reference_id TEXT,
            label TEXT,
            dnd_override INTEGER NOT NULL DEFAULT 0,
            duration_s INTEGER,
            paused_remaining REAL,
            recurrence_json TEXT
        );
    """)
    conn.commit()

    yield conn
    conn.close()


@pytest.fixture
def scheduler(test_db):
    """Create test scheduler."""
    sched = Scheduler(conn=test_db, user_tz="America/Chicago")
    yield sched
    sched.stop()


@pytest.fixture
def audio():
    """Create null audio player for testing."""
    return NullAudioPlayer()


class TestSystemHandlers:
    """Tests for system control handlers."""

    def test_handle_stop_nothing_playing(self):
        """Stop when nothing is playing."""
        query = create_utility_query("system", "stop")
        result = handle_stop(query)

        assert result.status == ResultStatus.OK
        assert "Nothing to stop" in result.display_text

    def test_handle_stop_with_audio(self, audio):
        """Stop with audio player."""
        query = create_utility_query("system", "stop")
        result = handle_stop(query, audio_player=audio)

        assert result.status == ResultStatus.OK

    def test_handle_status_empty(self, scheduler, test_db):
        """Status when nothing is active."""
        query = create_utility_query("system", "status")
        result = handle_status(query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "Nothing active" in result.display_text

    def test_handle_status_with_timer(self, scheduler, test_db, audio):
        """Status shows active timer."""
        from episodic.utility.handlers.timer import handle_timer_set

        # Set a timer first
        timer_query = create_utility_query("timer", "timer_set", args={
            "duration_s": 300,
            "label": "eggs"
        })
        handle_timer_set(timer_query, scheduler, test_db, "America/Chicago", audio)

        # Check status
        query = create_utility_query("system", "status")
        result = handle_status(query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "Timer" in result.display_text or "eggs" in result.display_text

    def test_handle_dnd_on(self, scheduler, test_db):
        """Enable DND."""
        query = create_utility_query("system", "dnd_on", args={"duration_m": 60})
        result = handle_dnd_on(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert "Do Not Disturb enabled" in result.display_text
        assert scheduler.is_dnd_active()

    def test_handle_dnd_off(self, scheduler, test_db):
        """Disable DND."""
        # First enable it
        on_query = create_utility_query("system", "dnd_on")
        handle_dnd_on(on_query, scheduler, test_db, "America/Chicago")

        # Then disable it
        off_query = create_utility_query("system", "dnd_off")
        result = handle_dnd_off(off_query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "Do Not Disturb disabled" in result.display_text
        assert not scheduler.is_dnd_active()

    def test_handle_undo_empty_stack(self, test_db):
        """Undo when stack is empty."""
        query = create_utility_query("system", "undo")
        result = handle_undo(query, test_db)

        assert result.status == ResultStatus.OK
        assert "Nothing to undo" in result.display_text

    def test_handle_repeat_nothing(self):
        """Repeat when nothing to repeat."""
        query = create_utility_query("system", "repeat")
        result = handle_repeat(query)

        assert result.status == ResultStatus.OK
        assert "Nothing to repeat" in result.display_text

    def test_dispatch_system_command(self, scheduler, test_db):
        """System dispatcher routes correctly."""
        query = create_utility_query("system", "status")
        result = dispatch_system_command(query, scheduler, test_db)

        assert result.status == ResultStatus.OK

    def test_dispatch_system_unknown_command(self, scheduler, test_db):
        """System dispatcher handles unknown commands."""
        query = create_utility_query("system", "unknown")
        result = dispatch_system_command(query, scheduler, test_db)

        assert result.status == ResultStatus.ERROR


class TestNotesHandlers:
    """Tests for notes handlers."""

    def test_note_add(self, test_db):
        """Can add a note."""
        query = create_utility_query("note", "note_add", args={
            "text": "Buy milk"
        })
        result = handle_note_add(query, test_db)

        assert result.status == ResultStatus.OK
        assert "Note saved" in result.display_text
        assert result.data["text"] == "Buy milk"

    def test_note_add_missing_text(self, test_db):
        """Note add without text fails."""
        query = create_utility_query("note", "note_add", args={})
        result = handle_note_add(query, test_db)

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "missing_text"

    def test_note_list(self, test_db):
        """Can list notes."""
        # Add some notes
        for text in ["Note 1", "Note 2", "Note 3"]:
            query = create_utility_query("note", "note_add", args={"text": text})
            handle_note_add(query, test_db)

        # List them
        list_query = create_utility_query("note", "note_list")
        result = handle_note_list(list_query, test_db)

        assert result.status == ResultStatus.OK
        assert len(result.data["notes"]) == 3

    def test_note_list_empty(self, test_db):
        """List when no notes."""
        query = create_utility_query("note", "note_list")
        result = handle_note_list(query, test_db)

        assert result.status == ResultStatus.OK
        assert "No notes" in result.display_text

    def test_note_search(self, test_db):
        """Can search notes."""
        # Add notes
        handle_note_add(
            create_utility_query("note", "note_add", args={"text": "Buy milk"}),
            test_db
        )
        handle_note_add(
            create_utility_query("note", "note_add", args={"text": "Call mom"}),
            test_db
        )

        # Search
        query = create_utility_query("note", "note_search", args={"query_text": "milk"})
        result = handle_note_search(query, test_db)

        assert result.status == ResultStatus.OK
        assert len(result.data["notes"]) == 1
        assert "milk" in result.data["notes"][0]["text"]

    def test_note_search_no_results(self, test_db):
        """Search with no matches."""
        handle_note_add(
            create_utility_query("note", "note_add", args={"text": "Buy milk"}),
            test_db
        )

        query = create_utility_query("note", "note_search", args={"query_text": "eggs"})
        result = handle_note_search(query, test_db)

        assert result.status == ResultStatus.OK
        assert len(result.data["notes"]) == 0

    def test_note_delete(self, test_db):
        """Can delete a note."""
        # Add a note
        add_result = handle_note_add(
            create_utility_query("note", "note_add", args={"text": "Delete me"}),
            test_db
        )
        note_id = add_result.data["note_id"]

        # Delete it
        query = create_utility_query("note", "note_delete", args={"note_id": note_id})
        result = handle_note_delete(query, test_db)

        assert result.status == ResultStatus.OK
        assert "Deleted" in result.display_text

        # Verify it's gone
        list_result = handle_note_list(create_utility_query("note", "note_list"), test_db)
        assert len(list_result.data["notes"]) == 0

    def test_dispatch_note_command(self, test_db):
        """Note dispatcher routes correctly."""
        query = create_utility_query("note", "note_add", args={"text": "Test"})
        result = dispatch_note_command(query, test_db)

        assert result.status == ResultStatus.OK

    def test_dispatch_note_unknown_command(self, test_db):
        """Note dispatcher handles unknown commands."""
        query = create_utility_query("note", "unknown")
        result = dispatch_note_command(query, test_db)

        assert result.status == ResultStatus.ERROR


class TestRemindersHandlers:
    """Tests for reminders handlers."""

    def test_remind_set_minutes(self, scheduler, test_db):
        """Can set reminder in minutes."""
        query = create_utility_query("reminder", "remind_set", args={
            "text": "Take medicine",
            "minutes": 30,
        })
        result = handle_remind_set(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert result.data["text"] == "Take medicine"
        assert "reminder_id" in result.data

    def test_remind_set_hours(self, scheduler, test_db):
        """Can set reminder in hours."""
        query = create_utility_query("reminder", "remind_set", args={
            "text": "Meeting",
            "hours": 2,
        })
        result = handle_remind_set(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert "Meeting" in result.display_text

    def test_remind_set_at_time(self, scheduler, test_db):
        """Can set reminder at specific time."""
        query = create_utility_query("reminder", "remind_set", args={
            "text": "Lunch",
            "at_time": "12:00pm",
        })
        result = handle_remind_set(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK

    def test_remind_set_missing_text(self, scheduler, test_db):
        """Reminder without text fails."""
        query = create_utility_query("reminder", "remind_set", args={
            "minutes": 30,
        })
        result = handle_remind_set(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "missing_text"

    def test_remind_set_missing_time(self, scheduler, test_db):
        """Reminder without time fails."""
        query = create_utility_query("reminder", "remind_set", args={
            "text": "Test",
        })
        result = handle_remind_set(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "invalid_time"

    def test_remind_list(self, scheduler, test_db):
        """Can list reminders."""
        # Set some reminders
        for i, text in enumerate(["Reminder 1", "Reminder 2"]):
            query = create_utility_query("reminder", "remind_set", args={
                "text": text,
                "minutes": 30 + i * 10,
            })
            handle_remind_set(query, scheduler, test_db, "America/Chicago")

        # List them
        list_query = create_utility_query("reminder", "remind_list")
        result = handle_remind_list(list_query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert len(result.data["reminders"]) == 2

    def test_remind_list_empty(self, scheduler, test_db):
        """List when no reminders."""
        query = create_utility_query("reminder", "remind_list")
        result = handle_remind_list(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert "No pending reminders" in result.display_text

    def test_remind_cancel(self, scheduler, test_db):
        """Can cancel a reminder."""
        # Set a reminder
        set_result = handle_remind_set(
            create_utility_query("reminder", "remind_set", args={
                "text": "Cancel me",
                "minutes": 60,
            }),
            scheduler, test_db, "America/Chicago"
        )
        reminder_id = set_result.data["reminder_id"]

        # Cancel it
        query = create_utility_query("reminder", "remind_cancel", args={
            "reminder_id": reminder_id
        })
        result = handle_remind_cancel(query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "Reminder cancelled" in result.speech_text

    def test_remind_cancel_by_text(self, scheduler, test_db):
        """Can cancel reminder by text match."""
        # Set a reminder
        handle_remind_set(
            create_utility_query("reminder", "remind_set", args={
                "text": "Call dentist",
                "minutes": 60,
            }),
            scheduler, test_db, "America/Chicago"
        )

        # Cancel by text
        query = create_utility_query("reminder", "remind_cancel", args={"text": "dentist"})
        result = handle_remind_cancel(query, scheduler, test_db)

        assert result.status == ResultStatus.OK

    def test_remind_cancel_all(self, scheduler, test_db):
        """Can cancel all reminders."""
        # Set multiple reminders
        for i in range(3):
            handle_remind_set(
                create_utility_query("reminder", "remind_set", args={
                    "text": f"Reminder {i}",
                    "minutes": 30 + i * 10,
                }),
                scheduler, test_db, "America/Chicago"
            )

        # Cancel all
        query = create_utility_query("reminder", "remind_cancel", args={"all": True})
        result = handle_remind_cancel(query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert result.data["cancelled_count"] == 3

    def test_dispatch_reminder_command(self, scheduler, test_db):
        """Reminder dispatcher routes correctly."""
        query = create_utility_query("reminder", "remind_set", args={
            "text": "Test",
            "minutes": 10,
        })
        result = dispatch_reminder_command(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK

    def test_dispatch_reminder_unknown_command(self, scheduler, test_db):
        """Reminder dispatcher handles unknown commands."""
        query = create_utility_query("reminder", "unknown")
        result = dispatch_reminder_command(query, scheduler, test_db)

        assert result.status == ResultStatus.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
