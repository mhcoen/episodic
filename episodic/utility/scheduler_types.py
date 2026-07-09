"""Scheduler task types: enums and dataclasses.

Split out of scheduler.py. Re-exported there so external imports
(from episodic.utility.scheduler import TaskType, ScheduledTask, ...) are
unchanged.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Callable, List, Optional, Union


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
