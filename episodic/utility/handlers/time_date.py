"""
Time and Date Handlers.

Zero-dependency handlers for time/date queries.
These respond immediately without LLM or external services.
"""

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from ..types import UtilityQuery, UtilityResult


def handle_time(query: UtilityQuery, user_tz: str = "America/Chicago") -> UtilityResult:
    """
    Handle time_now command.

    Returns current time in user's timezone.
    """
    try:
        tz = ZoneInfo(user_tz)
    except Exception:
        tz = ZoneInfo("America/Chicago")

    now = datetime.now(tz)

    # Format for display
    time_str = now.strftime("%I:%M %p").lstrip("0")  # "3:45 PM"
    date_str = now.strftime("%A, %B %d")  # "Tuesday, February 4"

    display = f"It's {time_str}"
    speech = f"It's {time_str}"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        _command="time_now",
        time=now.isoformat(),
        timezone=user_tz,
    )


def handle_date(query: UtilityQuery, user_tz: str = "America/Chicago") -> UtilityResult:
    """
    Handle date_today command.

    Returns current date in user's timezone.
    """
    try:
        tz = ZoneInfo(user_tz)
    except Exception:
        tz = ZoneInfo("America/Chicago")

    now = datetime.now(tz)

    # Format for display
    date_str = now.strftime("%A, %B %d, %Y")  # "Tuesday, February 4, 2026"
    day_of_week = now.strftime("%A")
    month_day = now.strftime("%B %d").replace(" 0", " ")  # Remove leading zero

    display = f"Today is {date_str}"
    speech = f"Today is {day_of_week}, {month_day}"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        _command="date_today",
        date=now.date().isoformat(),
        day_of_week=day_of_week,
        timezone=user_tz,
    )


def handle_day_of_week(query: UtilityQuery, user_tz: str = "America/Chicago") -> UtilityResult:
    """
    Handle day_of_week command for relative dates.

    Args in query:
        day_offset: 0=today, 1=tomorrow, -1=yesterday, etc.
        target_date: ISO date string (alternative to offset)
    """
    try:
        tz = ZoneInfo(user_tz)
    except Exception:
        tz = ZoneInfo("America/Chicago")

    today = datetime.now(tz).date()

    # Get target date from args
    if "target_date" in query.args:
        target = date.fromisoformat(query.args["target_date"])
    elif "day_offset" in query.args:
        offset = query.args["day_offset"]
        target = today + timedelta(days=offset)
    else:
        target = today

    day_name = target.strftime("%A")
    date_str = target.strftime("%B %d, %Y")

    # Determine relative description
    diff = (target - today).days
    if diff == 0:
        relative = "Today"
    elif diff == 1:
        relative = "Tomorrow"
    elif diff == -1:
        relative = "Yesterday"
    elif diff > 0:
        relative = f"In {diff} days"
    else:
        relative = f"{-diff} days ago"

    display = f"{relative} ({date_str}) is a {day_name}"
    speech = f"{relative} is {day_name}"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        day_of_week=day_name,
        date=target.isoformat(),
        relative=relative,
    )


# Command routing for time category
TIME_HANDLERS = {
    "time_now": handle_time,
    "date_today": handle_date,
    "day_of_week": handle_day_of_week,
}


def dispatch_time_command(query: UtilityQuery, user_tz: str = "America/Chicago") -> UtilityResult:
    """Dispatch a time category command to the appropriate handler."""
    handler = TIME_HANDLERS.get(query.command)
    if handler:
        return handler(query, user_tz)
    return UtilityResult.error("unknown_command", f"Unknown time command: {query.command}")
