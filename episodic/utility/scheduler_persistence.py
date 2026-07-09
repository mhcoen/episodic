"""Scheduler persistence: SQLite load/save of scheduled tasks.

Mixin split out of scheduler.py; Scheduler inherits it, so these methods run as
part of the Scheduler instance (self._conn, self._lock, self._tasks, and calls
to self._wall_to_monotonic / self._roll_forward all resolve on the instance).
"""

import heapq
import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from episodic.utility.scheduler_types import ScheduledTask, TaskType

logger = logging.getLogger(__name__)


class _SchedulerPersistenceMixin:
    """SQLite persistence methods for Scheduler."""

    def _persist_task(self, task: ScheduledTask) -> None:
        """Persist task to SQLite. Persistence failures are logged, not
        raised — a task that can't be persisted should still run this
        session, and a DB error must never kill the scheduler thread."""
        if self._conn is None:
            return

        try:
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO scheduled_tasks
                    (id, task_type, priority, next_run_ts, reference_id, label, dnd_override,
                     duration_s, paused_remaining, recurrence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id,
                    task.task_type.name,
                    task.priority,
                    int(task.next_run_wall.timestamp()),
                    task.reference_id,
                    task.label,
                    1 if task.dnd_override else 0,
                    task.duration_s,
                    task.paused_remaining,
                    json.dumps(task.recurrence) if task.recurrence else None,
                ))
                self._conn.commit()
        except Exception:
            logger.exception("Failed to persist task %s", task.id)

    def _delete_persisted_task(self, task_id: str) -> None:
        """Delete task from SQLite (failures logged, not raised)."""
        if self._conn is None:
            return

        try:
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
                self._conn.commit()
        except Exception:
            logger.exception("Failed to delete persisted task %s", task_id)

    def _load_persisted_tasks(self) -> None:
        """Load tasks from SQLite on startup."""
        if self._conn is None:
            return

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT id, task_type, priority, next_run_ts, reference_id, label,
                   dnd_override, duration_s, paused_remaining, recurrence_json
            FROM scheduled_tasks
        """)

        now = datetime.now(ZoneInfo(self._user_tz))

        for row in cursor.fetchall():
            task_id, task_type_str, priority, next_run_ts, reference_id, label, \
                dnd_override, duration_s, paused_remaining, recurrence_json = row

            try:
                task_type = TaskType[task_type_str]
            except KeyError:
                continue

            next_run_wall = datetime.fromtimestamp(next_run_ts, tz=ZoneInfo(self._user_tz))

            task = ScheduledTask(
                id=task_id,
                task_type=task_type,
                priority=priority,
                next_run_monotonic=0.0,  # set below, after missed-task handling
                next_run_wall=next_run_wall,
                created_at=now,  # Approximate
                reference_id=reference_id,
                label=label,
                dnd_override=bool(dnd_override),
                duration_s=duration_s,
                paused_remaining=paused_remaining,
                recurrence=json.loads(recurrence_json) if recurrence_json else None,
            )

            # Handle missed tasks
            if next_run_wall < now and paused_remaining is None:
                late_s = (now - next_run_wall).total_seconds()
                fireable = task_type in (
                    TaskType.ALARM, TaskType.TIMER, TaskType.REMINDER
                )

                if fireable and late_s <= self.STALE_GRACE_S:
                    # Recently missed — fire now
                    task.next_run_wall = now
                elif task.recurrence:
                    # Too stale for this occurrence — schedule the next one
                    rolled = self._roll_forward(task, now)
                    if rolled is None:
                        self._delete_persisted_task(task_id)
                        continue
                    task.next_run_wall = rolled
                    self._persist_task(task)
                else:
                    # Too stale to fire — clean up so it doesn't haunt
                    # every subsequent startup
                    self._delete_persisted_task(task_id)
                    if task_type == TaskType.TIMER and reference_id:
                        self._update_timer_status(reference_id, "expired")
                    continue

            task.next_run_monotonic = self._wall_to_monotonic(task.next_run_wall)

            # Closures can't be persisted; rebuild the callback so restored
            # tasks actually ring instead of firing silently.
            if self._callback_factory is not None:
                try:
                    task.callback = self._callback_factory(task)
                except Exception:
                    logger.exception("Failed to rebuild callback for task %s", task_id)

            self._tasks[task.id] = task
            if paused_remaining is None:
                heapq.heappush(self._queue, task)

    def _update_timer_status(self, timer_id: str, status: str) -> None:
        """Update timer status in database (failures logged, not raised)."""
        if self._conn is None:
            return

        try:
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute(
                    "UPDATE timers SET status = ? WHERE id = ?",
                    (status, timer_id)
                )
                self._conn.commit()
        except Exception:
            logger.exception("Failed to update timer %s status", timer_id)

