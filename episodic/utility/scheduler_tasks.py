"""Scheduler task factory functions (timer / alarm / reminder).

Split out of scheduler.py and re-exported there.
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from episodic.utility.scheduler_types import ScheduledTask, TaskType, TaskResult


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
