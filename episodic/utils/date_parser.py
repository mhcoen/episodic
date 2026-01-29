"""
Natural language date parsing utilities.

Provides timezone-aware parsing of time expressions like:
- "since yesterday"
- "before last week"
- "between 10am and 2pm today"
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import dateparser


def parse_time_range(
    expression: str,
    user_tz: str = "America/Chicago",
    reference_time: Optional[datetime] = None
) -> Optional[Tuple[Optional[datetime], Optional[datetime]]]:
    """
    Parse a natural language time expression into a UTC datetime range.

    Supports:
    - "since X" → (parse(X), None)
    - "before X" → (None, parse(X))
    - "between X and Y" → (parse(X), parse(Y))
    - Plain time like "yesterday" → (start_of_day, end_of_day)

    Args:
        expression: Natural language time expression
        user_tz: User's timezone (for interpreting "today", "yesterday", etc.)
        reference_time: Reference datetime for relative parsing (default: now)

    Returns:
        Tuple of (start_utc, end_utc) where either can be None for open ranges,
        or None if parsing fails.
    """
    if not expression:
        return None

    expression = expression.strip().lower()
    tz = ZoneInfo(user_tz)

    # Get reference time in user's timezone
    if reference_time is None:
        reference_time = datetime.now(tz)
    elif reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=tz)
    else:
        reference_time = reference_time.astimezone(tz)

    # Pattern: "since X"
    since_match = re.match(r'^since\s+(.+)$', expression)
    if since_match:
        start = _parse_single(since_match.group(1), tz, reference_time)
        if start:
            return (start.astimezone(ZoneInfo("UTC")), None)
        return None

    # Pattern: "before X"
    before_match = re.match(r'^before\s+(.+)$', expression)
    if before_match:
        end = _parse_single(before_match.group(1), tz, reference_time)
        if end:
            return (None, end.astimezone(ZoneInfo("UTC")))
        return None

    # Pattern: "between X and Y"
    between_match = re.match(r'^between\s+(.+?)\s+and\s+(.+)$', expression)
    if between_match:
        start = _parse_single(between_match.group(1), tz, reference_time)
        end = _parse_single(between_match.group(2), tz, reference_time)
        if start and end:
            return (
                start.astimezone(ZoneInfo("UTC")),
                end.astimezone(ZoneInfo("UTC"))
            )
        return None

    # Plain time expression - interpret as day range
    parsed = _parse_single(expression, tz, reference_time)
    if parsed:
        # For day-based expressions, return the full day range
        if _is_day_expression(expression):
            start_of_day = parsed.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            return (
                start_of_day.astimezone(ZoneInfo("UTC")),
                end_of_day.astimezone(ZoneInfo("UTC"))
            )
        # For specific times, use as start with no end
        return (parsed.astimezone(ZoneInfo("UTC")), None)

    return None


def _parse_single(
    text: str,
    tz: ZoneInfo,
    reference_time: datetime
) -> Optional[datetime]:
    """
    Parse a single time expression.

    Args:
        text: The time text to parse
        tz: The timezone to interpret the time in
        reference_time: Reference datetime for relative parsing

    Returns:
        Parsed datetime in the given timezone, or None if parsing fails.
    """
    settings = {
        'TIMEZONE': str(tz),
        'RETURN_AS_TIMEZONE_AWARE': True,
        'RELATIVE_BASE': reference_time.replace(tzinfo=None),
        'PREFER_DATES_FROM': 'past',
    }

    result = dateparser.parse(text, settings=settings)
    if result:
        # Ensure it's in the user's timezone
        if result.tzinfo is None:
            result = result.replace(tzinfo=tz)
        return result

    return None


def _is_day_expression(text: str) -> bool:
    """Check if the expression refers to a whole day."""
    day_keywords = [
        'today', 'yesterday', 'tomorrow',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'last week', 'this week', 'next week',
        'last month', 'this month', 'next month',
        'ago',  # "3 days ago" should be a day range
    ]
    text_lower = text.lower()
    for keyword in day_keywords:
        if keyword in text_lower:
            return True

    # Check for date patterns like "january 15" or "jan 15 2024"
    if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d+', text_lower):
        return True

    return False


def format_time_range(
    start: Optional[datetime],
    end: Optional[datetime],
    user_tz: str = "America/Chicago"
) -> str:
    """
    Format a time range for display.

    Args:
        start: Start datetime (UTC)
        end: End datetime (UTC)
        user_tz: User's timezone for display

    Returns:
        Human-readable string describing the range
    """
    tz = ZoneInfo(user_tz)

    if start and end:
        start_local = start.astimezone(tz)
        end_local = end.astimezone(tz)
        return f"between {start_local.strftime('%Y-%m-%d %H:%M')} and {end_local.strftime('%Y-%m-%d %H:%M')}"
    elif start:
        start_local = start.astimezone(tz)
        return f"since {start_local.strftime('%Y-%m-%d %H:%M')}"
    elif end:
        end_local = end.astimezone(tz)
        return f"before {end_local.strftime('%Y-%m-%d %H:%M')}"
    else:
        return "all time"
