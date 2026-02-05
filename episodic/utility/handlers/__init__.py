"""
Utility Command Handlers.

Each handler module provides functions to execute specific utility commands.
"""

from .time_date import handle_time, handle_date, dispatch_time_command
from .calculator import handle_calc, dispatch_calc_command
from .timer import (
    handle_timer_set,
    handle_timer_cancel,
    handle_timer_status,
    handle_timer_pause,
    handle_timer_resume,
    dispatch_timer_command,
)
from .alarm import (
    handle_alarm_set,
    handle_alarm_cancel,
    handle_alarm_list,
    handle_alarm_snooze,
    dispatch_alarm_command,
)
from .system import (
    handle_stop,
    handle_cancel,
    handle_undo,
    handle_repeat,
    handle_status,
    handle_dnd_on,
    handle_dnd_off,
    dispatch_system_command,
)
from .notes import (
    handle_note_add,
    handle_note_list,
    handle_note_search,
    handle_note_delete,
    dispatch_note_command,
)
from .reminders import (
    handle_remind_set,
    handle_remind_list,
    handle_remind_cancel,
    dispatch_reminder_command,
)
from .media import (
    handle_media_play,
    handle_media_pause,
    handle_media_resume,
    handle_media_stop,
    handle_media_status,
    handle_volume_up,
    handle_volume_down,
    handle_volume_mute,
    handle_volume_set,
    dispatch_media_command,
)
from .weather import (
    handle_weather_now,
    handle_weather_forecast,
    dispatch_weather_command,
)
from .news import (
    handle_news_headlines,
    handle_news_topic,
    dispatch_news_command,
)

__all__ = [
    # Time/Date
    "handle_time",
    "handle_date",
    "dispatch_time_command",
    # Calculator
    "handle_calc",
    "dispatch_calc_command",
    # Timer
    "handle_timer_set",
    "handle_timer_cancel",
    "handle_timer_status",
    "handle_timer_pause",
    "handle_timer_resume",
    "dispatch_timer_command",
    # Alarm
    "handle_alarm_set",
    "handle_alarm_cancel",
    "handle_alarm_list",
    "handle_alarm_snooze",
    "dispatch_alarm_command",
    # System
    "handle_stop",
    "handle_cancel",
    "handle_undo",
    "handle_repeat",
    "handle_status",
    "handle_dnd_on",
    "handle_dnd_off",
    "dispatch_system_command",
    # Notes
    "handle_note_add",
    "handle_note_list",
    "handle_note_search",
    "handle_note_delete",
    "dispatch_note_command",
    # Reminders
    "handle_remind_set",
    "handle_remind_list",
    "handle_remind_cancel",
    "dispatch_reminder_command",
    # Media
    "handle_media_play",
    "handle_media_pause",
    "handle_media_resume",
    "handle_media_stop",
    "handle_media_status",
    "handle_volume_up",
    "handle_volume_down",
    "handle_volume_mute",
    "handle_volume_set",
    "dispatch_media_command",
    # Weather
    "handle_weather_now",
    "handle_weather_forecast",
    "dispatch_weather_command",
    # News
    "handle_news_headlines",
    "handle_news_topic",
    "dispatch_news_command",
]
