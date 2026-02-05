"""
Tests for Timer and Alarm handlers.

Tests cover:
1. Timer CRUD operations (set, cancel, status, pause, resume)
2. Alarm CRUD operations (set, cancel, list, snooze)
3. Scheduler integration
4. DND suppression
"""

import pytest
import sqlite3
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from episodic.utility.types import UtilityQuery, ResultStatus
from episodic.utility.scheduler import (
    Scheduler,
    ScheduledTask,
    TaskType,
    TaskStatus,
    TaskResult,
    create_timer_task,
    create_alarm_task,
)
from episodic.utility.audio import NullAudioPlayer
from episodic.utility.handlers.timer import (
    handle_timer_set,
    handle_timer_cancel,
    handle_timer_status,
    handle_timer_pause,
    handle_timer_resume,
    dispatch_timer_command,
)
from episodic.utility.handlers.alarm import (
    handle_alarm_set,
    handle_alarm_cancel,
    handle_alarm_list,
    handle_alarm_snooze,
    dispatch_alarm_command,
)
from episodic.utility.dispatcher import create_utility_query


@pytest.fixture
def test_db():
    """Create in-memory test database with timer/alarm schema."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create minimal schema for testing
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


class TestScheduler:
    """Tests for Scheduler class."""

    def test_scheduler_create(self, test_db):
        """Scheduler can be created."""
        sched = Scheduler(conn=test_db)
        assert sched is not None
        assert not sched.is_running()

    def test_scheduler_start_stop(self, test_db):
        """Scheduler can start and stop."""
        sched = Scheduler(conn=test_db)
        sched.start()
        assert sched.is_running()
        sched.stop()
        assert not sched.is_running()

    def test_add_task(self, scheduler):
        """Can add task to scheduler."""
        task = create_timer_task(
            duration_s=60,
            label="test",
            user_tz="America/Chicago",
        )

        task_id = scheduler.add_task(task)
        assert task_id == task.id

        # Verify task is in queue
        tasks = scheduler.list_pending(TaskType.TIMER)
        assert len(tasks) == 1
        assert tasks[0].id == task_id

    def test_cancel_task(self, scheduler):
        """Can cancel task."""
        task = create_timer_task(duration_s=60, label="test")
        scheduler.add_task(task)

        cancelled = scheduler.cancel_task(task.id)
        assert cancelled

        tasks = scheduler.list_pending(TaskType.TIMER)
        assert len(tasks) == 0

    def test_cancel_nonexistent_task(self, scheduler):
        """Cancelling nonexistent task returns False."""
        cancelled = scheduler.cancel_task("nonexistent")
        assert not cancelled

    def test_cancel_by_type(self, scheduler):
        """Can cancel all tasks of a type."""
        for i in range(3):
            task = create_timer_task(duration_s=60 + i, label=f"timer{i}")
            scheduler.add_task(task)

        alarm = create_alarm_task(
            alarm_time=datetime.now(ZoneInfo("America/Chicago")) + timedelta(hours=1),
            label="alarm",
        )
        scheduler.add_task(alarm)

        cancelled = scheduler.cancel_by_type(TaskType.TIMER)
        assert cancelled == 3

        # Alarm should still exist
        tasks = scheduler.list_pending(TaskType.ALARM)
        assert len(tasks) == 1

    def test_get_timer_remaining(self, scheduler):
        """Can get remaining time on timer."""
        task = create_timer_task(duration_s=60, label="test")
        scheduler.add_task(task)

        remaining = scheduler.get_timer_remaining(task.id)
        assert remaining is not None
        assert 58 <= remaining <= 60  # Allow small timing variance

    def test_pause_resume_timer(self, scheduler):
        """Can pause and resume timer."""
        task = create_timer_task(duration_s=60, label="test")
        scheduler.add_task(task)

        # Pause
        paused = scheduler.pause_timer(task.id)
        assert paused

        # Verify paused
        remaining = scheduler.get_timer_remaining(task.id)
        assert remaining is not None

        # Wait a bit
        time.sleep(0.1)

        # Resume
        resumed = scheduler.resume_timer(task.id)
        assert resumed

        # Remaining should be approximately same as before
        new_remaining = scheduler.get_timer_remaining(task.id)
        assert new_remaining is not None
        assert abs(new_remaining - remaining) < 2  # Allow 2s variance

    def test_dnd_active(self, scheduler):
        """Can set and check DND."""
        assert not scheduler.is_dnd_active()

        until = datetime.now(ZoneInfo("America/Chicago")) + timedelta(hours=1)
        scheduler.set_dnd(until)
        assert scheduler.is_dnd_active()

        scheduler.set_dnd(None)
        assert not scheduler.is_dnd_active()


class TestTimerHandlers:
    """Tests for timer command handlers."""

    def test_timer_set_basic(self, scheduler, test_db, audio):
        """Can set a basic timer."""
        query = create_utility_query("timer", "timer_set", args={"duration_s": 300})
        result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert "timer_id" in result.data
        assert result.data["duration"] == 300
        assert "5 minute" in result.display_text

    def test_timer_set_with_label(self, scheduler, test_db, audio):
        """Can set a labeled timer."""
        query = create_utility_query("timer", "timer_set", args={
            "duration_s": 600,
            "label": "pasta"
        })
        result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert "pasta" in result.display_text
        assert result.data["label"] == "pasta"

    def test_timer_set_missing_duration(self, scheduler, test_db, audio):
        """Timer set without duration fails."""
        query = create_utility_query("timer", "timer_set", args={})
        result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "missing_duration"

    def test_timer_set_invalid_duration(self, scheduler, test_db, audio):
        """Timer set with negative duration fails."""
        query = create_utility_query("timer", "timer_set", args={"duration_s": -10})
        result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "invalid_duration"

    def test_timer_cancel(self, scheduler, test_db, audio):
        """Can cancel a timer."""
        # First set a timer
        query = create_utility_query("timer", "timer_set", args={"duration_s": 300})
        set_result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)
        timer_id = set_result.data["timer_id"]

        # Then cancel it
        cancel_query = create_utility_query("timer", "timer_cancel", args={"timer_id": timer_id})
        result = handle_timer_cancel(cancel_query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "cancelled" in result.display_text.lower()

    def test_timer_cancel_by_label(self, scheduler, test_db, audio):
        """Can cancel timer by label."""
        # Set a labeled timer
        query = create_utility_query("timer", "timer_set", args={
            "duration_s": 300,
            "label": "eggs"
        })
        handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)

        # Cancel by label
        cancel_query = create_utility_query("timer", "timer_cancel", args={"label": "eggs"})
        result = handle_timer_cancel(cancel_query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "eggs" in result.display_text.lower()

    def test_timer_cancel_all(self, scheduler, test_db, audio):
        """Can cancel all timers."""
        # Set multiple timers
        for i in range(3):
            query = create_utility_query("timer", "timer_set", args={"duration_s": 300 + i * 60})
            handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)

        # Cancel all
        cancel_query = create_utility_query("timer", "timer_cancel", args={"all": True})
        result = handle_timer_cancel(cancel_query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert result.data["cancelled_count"] == 3

    def test_timer_status_single(self, scheduler, test_db, audio):
        """Can get status of single timer."""
        # Set a timer
        query = create_utility_query("timer", "timer_set", args={
            "duration_s": 300,
            "label": "tea"
        })
        set_result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)
        timer_id = set_result.data["timer_id"]

        # Check status
        status_query = create_utility_query("timer", "timer_status", args={"timer_id": timer_id})
        result = handle_timer_status(status_query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert len(result.data["timers"]) == 1
        assert result.data["timers"][0]["label"] == "tea"

    def test_timer_status_no_timers(self, scheduler, test_db):
        """Status when no timers returns empty list."""
        query = create_utility_query("timer", "timer_status", args={})
        result = handle_timer_status(query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "No active timers" in result.display_text

    def test_timer_pause(self, scheduler, test_db, audio):
        """Can pause a timer."""
        # Set a timer
        query = create_utility_query("timer", "timer_set", args={"duration_s": 300})
        set_result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)
        task_id = set_result.data["task_id"]

        # Pause it
        pause_query = create_utility_query("timer", "timer_pause", args={
            "timer_id": set_result.data["timer_id"]
        })
        result = handle_timer_pause(pause_query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "paused" in result.display_text.lower()

    def test_timer_resume(self, scheduler, test_db, audio):
        """Can resume a paused timer."""
        # Set and pause a timer
        query = create_utility_query("timer", "timer_set", args={"duration_s": 300})
        set_result = handle_timer_set(query, scheduler, test_db, "America/Chicago", audio)
        timer_id = set_result.data["timer_id"]

        pause_query = create_utility_query("timer", "timer_pause", args={"timer_id": timer_id})
        handle_timer_pause(pause_query, scheduler, test_db)

        # Resume it
        resume_query = create_utility_query("timer", "timer_resume", args={"timer_id": timer_id})
        result = handle_timer_resume(resume_query, scheduler, test_db)

        assert result.status == ResultStatus.OK
        assert "resumed" in result.display_text.lower()

    def test_dispatch_timer_command(self, scheduler, test_db, audio):
        """Timer dispatcher routes correctly."""
        query = create_utility_query("timer", "timer_set", args={"duration_s": 120})
        result = dispatch_timer_command(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert result.data["duration"] == 120

    def test_dispatch_timer_unknown_command(self, scheduler, test_db):
        """Timer dispatcher handles unknown commands."""
        query = create_utility_query("timer", "unknown_command", args={})
        result = dispatch_timer_command(query, scheduler, test_db)

        assert result.status == ResultStatus.ERROR
        assert "unknown_command" in result.error_type


class TestAlarmHandlers:
    """Tests for alarm command handlers."""

    def test_alarm_set_by_hour_minute(self, scheduler, test_db, audio):
        """Can set alarm by hour and minute."""
        query = create_utility_query("alarm", "alarm_set", args={
            "hour": 7,
            "minute": 30,
        })
        result = handle_alarm_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert "alarm_id" in result.data
        assert "7:30" in result.display_text

    def test_alarm_set_with_label(self, scheduler, test_db, audio):
        """Can set labeled alarm."""
        query = create_utility_query("alarm", "alarm_set", args={
            "hour": 6,
            "minute": 0,
            "label": "Wake up",
        })
        result = handle_alarm_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert "Wake up" in result.display_text

    def test_alarm_set_invalid_time(self, scheduler, test_db, audio):
        """Alarm set with invalid time fails."""
        query = create_utility_query("alarm", "alarm_set", args={})
        result = handle_alarm_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "invalid_time"

    def test_alarm_cancel(self, scheduler, test_db, audio):
        """Can cancel an alarm."""
        # Set an alarm
        set_query = create_utility_query("alarm", "alarm_set", args={
            "hour": 8,
            "minute": 0,
        })
        set_result = handle_alarm_set(set_query, scheduler, test_db, "America/Chicago", audio)
        alarm_id = set_result.data["alarm_id"]

        # Cancel it
        cancel_query = create_utility_query("alarm", "alarm_cancel", args={"alarm_id": alarm_id})
        result = handle_alarm_cancel(cancel_query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert "cancelled" in result.display_text.lower()

    def test_alarm_cancel_by_label(self, scheduler, test_db, audio):
        """Can cancel alarm by label."""
        # Set a labeled alarm
        set_query = create_utility_query("alarm", "alarm_set", args={
            "hour": 9,
            "minute": 30,
            "label": "Meeting",
        })
        handle_alarm_set(set_query, scheduler, test_db, "America/Chicago", audio)

        # Cancel by label
        cancel_query = create_utility_query("alarm", "alarm_cancel", args={"label": "Meeting"})
        result = handle_alarm_cancel(cancel_query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert "Meeting" in result.display_text

    def test_alarm_cancel_all(self, scheduler, test_db, audio):
        """Can cancel all alarms."""
        # Set multiple alarms
        for hour in [7, 8, 9]:
            query = create_utility_query("alarm", "alarm_set", args={"hour": hour, "minute": 0})
            handle_alarm_set(query, scheduler, test_db, "America/Chicago", audio)

        # Cancel all
        cancel_query = create_utility_query("alarm", "alarm_cancel", args={"all": True})
        result = handle_alarm_cancel(cancel_query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert result.data["cancelled_count"] == 3

    def test_alarm_list(self, scheduler, test_db, audio):
        """Can list alarms."""
        # Set some alarms
        for hour in [7, 8]:
            query = create_utility_query("alarm", "alarm_set", args={"hour": hour, "minute": 0})
            handle_alarm_set(query, scheduler, test_db, "America/Chicago", audio)

        # List them
        list_query = create_utility_query("alarm", "alarm_list", args={})
        result = handle_alarm_list(list_query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert len(result.data["alarms"]) == 2

    def test_alarm_list_empty(self, scheduler, test_db):
        """Listing when no alarms returns empty."""
        query = create_utility_query("alarm", "alarm_list", args={})
        result = handle_alarm_list(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.OK
        assert "No alarms" in result.display_text

    def test_alarm_snooze(self, scheduler, test_db, audio):
        """Can snooze an alarm."""
        # Set an alarm
        set_query = create_utility_query("alarm", "alarm_set", args={
            "hour": 7,
            "minute": 0,
        })
        set_result = handle_alarm_set(set_query, scheduler, test_db, "America/Chicago", audio)

        # Snooze it
        snooze_query = create_utility_query("alarm", "alarm_snooze", args={
            "duration_m": 5,
            "alarm_id": set_result.data["alarm_id"],
        })
        result = handle_alarm_snooze(snooze_query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert "5 minute" in result.display_text.lower()

    def test_alarm_snooze_default_duration(self, scheduler, test_db, audio):
        """Snooze defaults to 9 minutes."""
        snooze_query = create_utility_query("alarm", "alarm_snooze", args={})
        result = handle_alarm_snooze(snooze_query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert "9 minute" in result.display_text.lower()

    def test_dispatch_alarm_command(self, scheduler, test_db, audio):
        """Alarm dispatcher routes correctly."""
        query = create_utility_query("alarm", "alarm_set", args={"hour": 10, "minute": 30})
        result = dispatch_alarm_command(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK

    def test_dispatch_alarm_unknown_command(self, scheduler, test_db):
        """Alarm dispatcher handles unknown commands."""
        query = create_utility_query("alarm", "unknown_command", args={})
        result = dispatch_alarm_command(query, scheduler, test_db, "America/Chicago")

        assert result.status == ResultStatus.ERROR
        assert "unknown_command" in result.error_type


class TestTaskFactories:
    """Tests for task factory functions."""

    def test_create_timer_task(self):
        """Can create timer task."""
        task = create_timer_task(
            duration_s=300,
            label="test",
            user_tz="America/Chicago",
        )

        assert task.task_type == TaskType.TIMER
        assert task.duration_s == 300
        assert task.label == "test"
        assert task.priority == 1

    def test_create_alarm_task(self):
        """Can create alarm task."""
        alarm_time = datetime.now(ZoneInfo("America/Chicago")) + timedelta(hours=1)
        task = create_alarm_task(
            alarm_time=alarm_time,
            label="morning",
            dnd_override=True,
            user_tz="America/Chicago",
        )

        assert task.task_type == TaskType.ALARM
        assert task.label == "morning"
        assert task.dnd_override is True
        assert task.priority == 1


class TestDNDIntegration:
    """Tests for DND (Do Not Disturb) integration."""

    def test_dnd_suppresses_alarm(self, scheduler, test_db, audio):
        """DND suppresses alarm audio."""
        # Enable DND
        until = datetime.now(ZoneInfo("America/Chicago")) + timedelta(hours=2)
        scheduler.set_dnd(until)

        # Set an alarm (it should respect DND)
        query = create_utility_query("alarm", "alarm_set", args={
            "hour": 23,  # Use late hour to avoid time conflicts
            "minute": 59,
        })
        result = handle_alarm_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK
        assert scheduler.is_dnd_active()

    def test_dnd_override_alarm(self, scheduler, test_db, audio):
        """DND override allows alarm through."""
        # Enable DND
        until = datetime.now(ZoneInfo("America/Chicago")) + timedelta(hours=2)
        scheduler.set_dnd(until)

        # Set an alarm with DND override
        query = create_utility_query("alarm", "alarm_set", args={
            "hour": 23,
            "minute": 58,
            "dnd_override": True,
        })
        result = handle_alarm_set(query, scheduler, test_db, "America/Chicago", audio)

        assert result.status == ResultStatus.OK

        # The alarm task should have dnd_override set
        tasks = scheduler.list_pending(TaskType.ALARM)
        assert len(tasks) == 1
        assert tasks[0].dnd_override is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
