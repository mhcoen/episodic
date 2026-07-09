"""Argument/time parsing helpers for utility commands.

Split out of cli_integration.py to reduce its size. Pure functions (no module
globals, no calls back into cli_integration); re-imported there so
handle_utility_command's bare-name calls resolve.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo


def _parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse duration string to seconds.

    Supports: 10s, 5m, 1h, 1h30m, 90m
    """
    duration_str = duration_str.lower().strip()

    # Handle combined formats like "1h30m"
    total_seconds = 0

    # Pattern: extract all (number, unit) pairs
    pattern = r'(\d+)\s*(h|hr|hour|hours|m|min|mins|minutes|s|sec|secs|seconds?)?'
    matches = re.findall(pattern, duration_str)

    if not matches:
        return None

    for value_str, unit in matches:
        value = int(value_str)

        if not unit or unit.startswith('s'):
            total_seconds += value
        elif unit.startswith('m'):
            total_seconds += value * 60
        elif unit.startswith('h'):
            total_seconds += value * 3600

    return total_seconds if total_seconds > 0 else None


_DURATION_UNITS = {
    's', 'sec', 'secs', 'second', 'seconds',
    'm', 'min', 'mins', 'minute', 'minutes',
    'h', 'hr', 'hour', 'hours',
}


def _parse_timer_args(args: list) -> tuple:
    """
    Split timer args into (duration_s, label).

    Greedily consumes tokens that are numbers or duration unit words,
    then treats the rest as label.

    "30 seconds"       → (30, None)
    "30 seconds pasta" → (30, "pasta")
    "5m eggs"          → (300, "eggs")
    "1h30m"            → (5400, None)
    "30"               → (30, None)
    """
    i = 0
    while i < len(args):
        token = args[i].lower()
        if re.match(r'^\d+', token):
            i += 1
            continue
        if token in _DURATION_UNITS and i > 0:
            i += 1
            continue
        break

    duration_str = " ".join(args[:i])
    duration_s = _parse_duration(duration_str) if duration_str else None
    label = " ".join(args[i:]) if i < len(args) else None
    return (duration_s, label)


def _parse_time(time_str: str, user_tz: str = "America/Chicago") -> Optional[datetime]:
    """
    Parse time string to datetime.

    Supports: 7am, 7:00am, 19:00, 7:00, 7:30pm
    """
    time_str = time_str.lower().strip()
    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    # Pattern for various time formats
    patterns = [
        # 7am, 7pm
        (r'^(\d{1,2})\s*(am|pm)$', lambda m: (int(m.group(1)), 0, m.group(2))),
        # 7:00am, 7:30pm
        (r'^(\d{1,2}):(\d{2})\s*(am|pm)$', lambda m: (int(m.group(1)), int(m.group(2)), m.group(3))),
        # 19:00, 7:00 (24-hour format)
        (r'^(\d{1,2}):(\d{2})$', lambda m: (int(m.group(1)), int(m.group(2)), None)),
    ]

    for pattern, extractor in patterns:
        match = re.match(pattern, time_str)
        if match:
            hour, minute, meridiem = extractor(match)

            # Convert to 24-hour format
            if meridiem:
                if meridiem == 'pm' and hour != 12:
                    hour += 12
                elif meridiem == 'am' and hour == 12:
                    hour = 0

            # Validate hour/minute ranges
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                return None

            # Create datetime for today
            alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # If time has passed today, schedule for tomorrow
            if alarm_time <= now:
                alarm_time += timedelta(days=1)

            return alarm_time

    return None


def _parse_remind_args(text: str, user_tz: str = "America/Chicago") -> Tuple[Optional[str], Optional[int], Optional[datetime]]:
    """
    Parse reminder text and time.

    Supports:
    - "call mom in 2h" → ("call mom", 7200, None)
    - "meeting at 3pm" → ("meeting", None, datetime)

    Returns (reminder_text, duration_seconds, alarm_datetime)
    """
    text = text.strip()

    # Try "X in Y" pattern
    in_match = re.match(r'^(.+?)\s+in\s+(.+)$', text, re.IGNORECASE)
    if in_match:
        reminder_text = in_match.group(1).strip()
        duration_str = in_match.group(2).strip()
        duration_s = _parse_duration(duration_str)
        if duration_s:
            return (reminder_text, duration_s, None)

    # Try "X at Y" pattern
    at_match = re.match(r'^(.+?)\s+at\s+(.+)$', text, re.IGNORECASE)
    if at_match:
        reminder_text = at_match.group(1).strip()
        time_str = at_match.group(2).strip()
        alarm_time = _parse_time(time_str, user_tz)
        if alarm_time:
            return (reminder_text, None, alarm_time)

    return (None, None, None)


