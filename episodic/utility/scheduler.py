"""
Scheduler for Utility Commands.

Background thread managing a priority queue of scheduled tasks.
Handles alarms, timers, reminders, and other scheduled events.

Uses monotonic time for scheduling correctness, wall time for display.
"""

import heapq
import logging
import threading
import time
import uuid
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of scheduled tasks."""
    ALARM = auto()
    TIMER = auto()
    REMINDER = auto()
    REFRESH = auto()
    ROUTINE_STEP = auto()
    SYSTEM = auto()


class TaskStatus(Enum):
    """Status of a scheduled task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class TaskResult:
    """Result from task callback execution."""
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    reschedule_at: Optional[datetime] = None
    next_task: Optional["ScheduledTask"] = None
    side_effects: Optional[List[str]] = None


@dataclass
class ScheduledTask:
    """A task scheduled for future execution."""
    id: str
    task_type: TaskType
    priority: int  # 0=highest (stop), 1=alarms/timers, 2=reminders, 3=routines, 4=refresh

    # Timing
    next_run_monotonic: float
    next_run_wall: datetime
    created_at: datetime

    # Execution
    callback: Optional[Callable[[], TaskResult]] = None
    reference_id: Optional[str] = None  # FK to alarms/timers/reminders table

    # Recurrence
    recurrence: Optional[Union[str, int]] = None  # RRULE string or interval seconds

    # Metadata
    label: Optional[str] = None
    dnd_override: bool = False

    # Timer-specific
    duration_s: Optional[int] = None  # Original duration for timers
    paused_remaining: Optional[float] = None  # Remaining time when paused

    def __lt__(self, other: "ScheduledTask") -> bool:
        """Comparison for heapq (earliest first, then by priority)."""
        if self.next_run_monotonic != other.next_run_monotonic:
            return self.next_run_monotonic < other.next_run_monotonic
        return self.priority < other.priority


class Scheduler:
    """
    Background scheduler for utility commands.

    Manages a priority queue of tasks with precise timing using monotonic time.
    Supports pause/resume for timers, DND mode, and recurrence.
    """

    # A missed task is still fired if it is less than this late; anything
    # staler is cleaned up (or rolled forward, if recurring) on load.
    STALE_GRACE_S = 300

    def __init__(
        self,
        conn: Optional[sqlite3.Connection] = None,
        user_tz: str = "America/Chicago",
    ):
        self._conn = conn
        self._user_tz = user_tz

        # Rebuilds callbacks for tasks restored from persistence
        # (closures cannot be persisted). Set before start().
        self._callback_factory: Optional[
            Callable[["ScheduledTask"], Optional[Callable[[], "TaskResult"]]]
        ] = None

        # Task storage
        self._queue: List[ScheduledTask] = []  # heapq
        self._tasks: Dict[str, ScheduledTask] = {}  # id -> task

        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # DND
        self._dnd_until: Optional[datetime] = None

        # Callbacks for task execution events
        self._on_task_fire: Optional[Callable[[ScheduledTask, TaskResult], None]] = None
        self._on_task_suppress: Optional[Callable[[ScheduledTask], None]] = None

    # =========================================================================
    # Task Management
    # =========================================================================

    def add_task(self, task: ScheduledTask) -> str:
        """
        Add a task to the queue.

        Returns the task ID.
        """
        with self._lock:
            self._tasks[task.id] = task
            heapq.heappush(self._queue, task)

            # Persist if not a refresh task
            if self._conn and task.task_type != TaskType.REFRESH:
                self._persist_task(task)

        return task.id

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.

        Returns True if found and cancelled.
        """
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task is None:
                return False

            # Mark as cancelled (will be skipped when popped from queue)
            if self._conn and task.task_type != TaskType.REFRESH:
                self._delete_persisted_task(task_id)

            return True

    def cancel_by_type(self, task_type: TaskType) -> int:
        """
        Cancel all tasks of a type.

        Returns count cancelled.
        """
        with self._lock:
            cancelled = 0
            to_remove = []

            for task_id, task in self._tasks.items():
                if task.task_type == task_type:
                    to_remove.append(task_id)

            for task_id in to_remove:
                self._tasks.pop(task_id, None)
                if self._conn:
                    self._delete_persisted_task(task_id)
                cancelled += 1

            return cancelled

    def cancel_by_reference(self, reference_id: str) -> bool:
        """
        Cancel task by reference (alarm/timer/reminder ID).
        """
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task.reference_id == reference_id:
                    return self.cancel_task(task_id)
            return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_pending(self, task_type: Optional[TaskType] = None) -> List[ScheduledTask]:
        """
        List pending tasks, optionally filtered by type.

        Returns tasks sorted by next_run_monotonic.
        """
        with self._lock:
            tasks = list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return sorted(tasks, key=lambda t: t.next_run_monotonic)

    # =========================================================================
    # Timer-Specific
    # =========================================================================

    def get_timer_remaining(self, task_id: str) -> Optional[int]:
        """
        Get seconds remaining on a timer.

        Returns None if not found or not a timer.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.task_type != TaskType.TIMER:
                return None

            if task.paused_remaining is not None:
                return int(task.paused_remaining)

            remaining = task.next_run_monotonic - time.monotonic()
            return max(0, int(remaining))

    def pause_timer(self, task_id: str) -> bool:
        """
        Pause a running timer.

        Returns True if paused successfully.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.task_type != TaskType.TIMER:
                return False

            if task.paused_remaining is not None:
                return False  # Already paused

            # Calculate remaining time
            remaining = task.next_run_monotonic - time.monotonic()
            if remaining <= 0:
                return False  # Already expired

            task.paused_remaining = remaining

            # Update persistence
            if self._conn:
                self._persist_task(task)

            return True

    def resume_timer(self, task_id: str) -> bool:
        """
        Resume a paused timer.

        Returns True if resumed successfully.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.task_type != TaskType.TIMER:
                return False

            if task.paused_remaining is None:
                return False  # Not paused

            # Recalculate next run time
            remaining = task.paused_remaining
            task.next_run_monotonic = time.monotonic() + remaining
            task.next_run_wall = datetime.now(ZoneInfo(self._user_tz)) + timedelta(seconds=remaining)
            task.paused_remaining = None

            # Re-add to queue (it may have been removed)
            heapq.heappush(self._queue, task)

            # Update persistence
            if self._conn:
                self._persist_task(task)

            return True

    # =========================================================================
    # DND
    # =========================================================================

    def set_dnd(self, until: Optional[datetime]) -> None:
        """
        Enable DND until specified time.

        Pass None to disable DND.
        """
        with self._lock:
            self._dnd_until = until

    def is_dnd_active(self) -> bool:
        """Check if DND is currently active."""
        if self._dnd_until is None:
            return False

        now = datetime.now(ZoneInfo(self._user_tz))
        if self._dnd_until.tzinfo is None:
            # Assume same timezone
            dnd_until = self._dnd_until.replace(tzinfo=ZoneInfo(self._user_tz))
        else:
            dnd_until = self._dnd_until

        return now < dnd_until

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the scheduler thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True

        # Load persisted tasks
        if self._conn:
            self._load_persisted_tasks()

        self._thread = threading.Thread(
            target=self._run,
            name="utility-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler thread gracefully."""
        if not self._running:
            return

        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._running = False

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    # =========================================================================
    # Main Loop
    # =========================================================================

    def _run(self) -> None:
        """Main scheduler loop."""
        try:
            while not self._stop_event.is_set():
                try:
                    self._run_once()
                except Exception:
                    # A single bad task or DB hiccup must not kill the
                    # thread — every pending alarm/timer depends on it.
                    logger.exception("Scheduler loop error (continuing)")

                # Wait for next check (or until stop)
                self._stop_event.wait(timeout=0.5)
        finally:
            # Reflect reality if the thread exits for any reason,
            # so is_running() doesn't report a dead thread as alive.
            self._running = False

    def _run_once(self) -> None:
        """Pop all due tasks, then execute their callbacks outside the lock."""
        now_mono = time.monotonic()
        now_wall = datetime.now(ZoneInfo(self._user_tz))
        due: List[ScheduledTask] = []

        with self._lock:
            while self._queue:
                next_task = self._queue[0]

                # Check if task was cancelled
                if next_task.id not in self._tasks:
                    heapq.heappop(self._queue)
                    continue

                # Check if paused
                task = self._tasks[next_task.id]
                if task.paused_remaining is not None:
                    heapq.heappop(self._queue)
                    continue

                # Due by monotonic deadline, or by wall clock — monotonic
                # time does not advance during system sleep, so wall time
                # is authoritative after a wake.
                mono_due = task.next_run_monotonic - now_mono <= 0
                next_wall = task.next_run_wall
                if next_wall.tzinfo is None:
                    next_wall = next_wall.replace(tzinfo=ZoneInfo(self._user_tz))
                if not mono_due and next_wall > now_wall:
                    break  # Not ready yet

                heapq.heappop(self._queue)
                task = self._tasks.pop(next_task.id, None)
                if task is not None:
                    due.append(task)

        # Execute outside the lock so slow callbacks (audio, TTS, DB)
        # don't block add_task/cancel_task/list_pending on other threads.
        for task in due:
            self._execute_task(task)

    def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a task callback."""
        # DND check
        if self._should_suppress(task):
            self._handle_suppressed(task)
            return

        if task.callback is None:
            # No callback (e.g. restored task whose owner couldn't be
            # rebuilt). Still notify and clean up — otherwise the
            # persisted row would silently re-fire on every startup.
            self._handle_result(
                task, TaskResult(status=TaskStatus.COMPLETED, output=task.label)
            )
            return

        try:
            result = task.callback()
            self._handle_result(task, result)
        except Exception as e:
            self._handle_error(task, e)

    def _should_suppress(self, task: ScheduledTask) -> bool:
        """Check if task should be suppressed due to DND."""
        if not self.is_dnd_active():
            return False
        if task.dnd_override:
            return False
        if task.task_type == TaskType.REFRESH:
            return False  # Refresh is silent
        return True

    def _handle_suppressed(self, task: ScheduledTask) -> None:
        """Handle a DND-suppressed task."""
        if self._on_task_suppress:
            self._on_task_suppress(task)

        if task.task_type in (TaskType.ALARM, TaskType.REMINDER):
            # Defer to end of DND
            if self._dnd_until:
                task.next_run_wall = self._dnd_until
                task.next_run_monotonic = self._wall_to_monotonic(self._dnd_until)
                self.add_task(task)

        # Timers still "fire" silently (update status to expired)
        if task.task_type == TaskType.TIMER:
            if self._conn and task.reference_id:
                self._update_timer_status(task.reference_id, "expired")

    def _handle_result(self, task: ScheduledTask, result: TaskResult) -> None:
        """Handle successful task execution."""
        if self._on_task_fire:
            self._on_task_fire(task, result)

        # Delete persisted task
        if self._conn and task.task_type != TaskType.REFRESH:
            self._delete_persisted_task(task.id)

        # Handle recurrence. The task keeps its ID across occurrences so
        # external references (e.g. alarms.task_id) stay valid for cancel.
        if result.reschedule_at:
            task.next_run_wall = result.reschedule_at
            task.next_run_monotonic = self._wall_to_monotonic(result.reschedule_at)
            self.add_task(task)
        elif task.recurrence:
            now = datetime.now(ZoneInfo(self._user_tz))
            next_time = self._roll_forward(task, now)
            if next_time:
                task.next_run_wall = next_time
                task.next_run_monotonic = self._wall_to_monotonic(next_time)
                self.add_task(task)

        # Handle routine chaining
        if result.next_task:
            self.add_task(result.next_task)

    def _handle_error(self, task: ScheduledTask, error: Exception) -> None:
        """Handle task execution error."""
        if self._on_task_fire:
            result = TaskResult(
                status=TaskStatus.FAILED,
                error=str(error),
            )
            self._on_task_fire(task, result)

        # Delete persisted task
        if self._conn and task.task_type != TaskType.REFRESH:
            self._delete_persisted_task(task.id)

    # =========================================================================
    # Persistence
    # =========================================================================

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

    # =========================================================================
    # Time Utilities
    # =========================================================================

    def _wall_to_monotonic(self, wall_time: datetime) -> float:
        """Convert wall time to monotonic time."""
        now = datetime.now(ZoneInfo(self._user_tz))
        if wall_time.tzinfo is None:
            wall_time = wall_time.replace(tzinfo=ZoneInfo(self._user_tz))

        delta = (wall_time - now).total_seconds()
        return time.monotonic() + delta

    def _roll_forward(self, task: ScheduledTask, now: datetime) -> Optional[datetime]:
        """Advance a recurring task's wall time to its first future
        occurrence, skipping any occurrences missed while the process was
        down or the system was asleep (avoids a catch-up refire storm)."""
        original_wall = task.next_run_wall
        try:
            for _ in range(10000):
                next_time = self._compute_next_recurrence(task)
                if next_time is None:
                    return None
                if next_time > now:
                    return next_time
                task.next_run_wall = next_time
            return None
        finally:
            task.next_run_wall = original_wall

    def _compute_next_recurrence(self, task: ScheduledTask) -> Optional[datetime]:
        """Compute next occurrence for recurring task."""
        if task.recurrence is None:
            return None

        if isinstance(task.recurrence, int):
            # Interval in seconds
            return task.next_run_wall + timedelta(seconds=task.recurrence)

        if isinstance(task.recurrence, str):
            # RRULE - simplified handling
            # Full RRULE parsing would require python-dateutil
            # For now, just handle basic daily/weekly patterns
            rrule = task.recurrence.upper()

            if "FREQ=DAILY" in rrule:
                return task.next_run_wall + timedelta(days=1)
            elif "FREQ=WEEKLY" in rrule:
                return task.next_run_wall + timedelta(weeks=1)
            elif "FREQ=HOURLY" in rrule:
                return task.next_run_wall + timedelta(hours=1)

        return None


# =========================================================================
# Task Factory Functions
# =========================================================================

def create_timer_task(
    duration_s: int,
    label: Optional[str] = None,
    callback: Optional[Callable[[], TaskResult]] = None,
    reference_id: Optional[str] = None,
    user_tz: str = "America/Chicago",
) -> ScheduledTask:
    """Create a timer task."""
    now = datetime.now(ZoneInfo(user_tz))

    return ScheduledTask(
        id=str(uuid.uuid4()),
        task_type=TaskType.TIMER,
        priority=1,
        next_run_monotonic=time.monotonic() + duration_s,
        next_run_wall=now + timedelta(seconds=duration_s),
        created_at=now,
        callback=callback,
        reference_id=reference_id,
        label=label,
        duration_s=duration_s,
    )


def create_alarm_task(
    alarm_time: datetime,
    label: Optional[str] = None,
    callback: Optional[Callable[[], TaskResult]] = None,
    reference_id: Optional[str] = None,
    dnd_override: bool = False,
    recurrence: Optional[str] = None,
    user_tz: str = "America/Chicago",
) -> ScheduledTask:
    """Create an alarm task."""
    now = datetime.now(ZoneInfo(user_tz))

    # Ensure alarm_time has timezone
    if alarm_time.tzinfo is None:
        alarm_time = alarm_time.replace(tzinfo=ZoneInfo(user_tz))

    # Calculate monotonic time
    delta = (alarm_time - now).total_seconds()

    return ScheduledTask(
        id=str(uuid.uuid4()),
        task_type=TaskType.ALARM,
        priority=1,
        next_run_monotonic=time.monotonic() + delta,
        next_run_wall=alarm_time,
        created_at=now,
        callback=callback,
        reference_id=reference_id,
        label=label,
        dnd_override=dnd_override,
        recurrence=recurrence,
    )


def create_reminder_task(
    remind_time: datetime,
    text: str,
    callback: Optional[Callable[[], TaskResult]] = None,
    reference_id: Optional[str] = None,
    user_tz: str = "America/Chicago",
) -> ScheduledTask:
    """Create a reminder task."""
    now = datetime.now(ZoneInfo(user_tz))

    if remind_time.tzinfo is None:
        remind_time = remind_time.replace(tzinfo=ZoneInfo(user_tz))

    delta = (remind_time - now).total_seconds()

    return ScheduledTask(
        id=str(uuid.uuid4()),
        task_type=TaskType.REMINDER,
        priority=2,
        next_run_monotonic=time.monotonic() + delta,
        next_run_wall=remind_time,
        created_at=now,
        callback=callback,
        reference_id=reference_id,
        label=text,
    )
