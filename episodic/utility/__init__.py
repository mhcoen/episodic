"""
Utility Commands Module for Episodic.

Provides immediate-response commands (time, date, timer, weather, etc.)
that bypass the LLM for fast, deterministic responses.
"""

from .types import UtilityQuery, UtilityResult, ResultStatus
from .dispatcher import dispatch_utility, create_utility_query
from .db import init_utility_schema
from .scheduler import (
    Scheduler,
    ScheduledTask,
    TaskType,
    TaskStatus,
    TaskResult,
    create_timer_task,
    create_alarm_task,
    create_reminder_task,
)
from .audio import (
    AudioPlayer,
    AudioPlayerImpl,
    NullAudioPlayer,
    SoundType,
    SoundConfig,
    create_audio_player,
)

__all__ = [
    # Types
    "UtilityQuery",
    "UtilityResult",
    "ResultStatus",
    # Dispatcher
    "dispatch_utility",
    "create_utility_query",
    # Database
    "init_utility_schema",
    # Scheduler
    "Scheduler",
    "ScheduledTask",
    "TaskType",
    "TaskStatus",
    "TaskResult",
    "create_timer_task",
    "create_alarm_task",
    "create_reminder_task",
    # Audio
    "AudioPlayer",
    "AudioPlayerImpl",
    "NullAudioPlayer",
    "SoundType",
    "SoundConfig",
    "create_audio_player",
]
