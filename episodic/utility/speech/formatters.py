"""
Value Formatters for Natural Speech.

Converts raw values (times, durations, temperatures) into
natural speech-friendly strings.
"""

from datetime import datetime
from typing import Any, Optional


# Number words for natural speech
_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
    21: "twenty-one", 22: "twenty-two", 23: "twenty-three",
    24: "twenty-four", 25: "twenty-five", 26: "twenty-six",
    27: "twenty-seven", 28: "twenty-eight", 29: "twenty-nine",
    30: "thirty", 31: "thirty-one", 32: "thirty-two",
    33: "thirty-three", 34: "thirty-four", 35: "thirty-five",
    36: "thirty-six", 37: "thirty-seven", 38: "thirty-eight",
    39: "thirty-nine", 40: "forty", 41: "forty-one",
    42: "forty-two", 43: "forty-three", 44: "forty-four",
    45: "forty-five", 46: "forty-six", 47: "forty-seven",
    48: "forty-eight", 49: "forty-nine", 50: "fifty",
    51: "fifty-one", 52: "fifty-two", 53: "fifty-three",
    54: "fifty-four", 55: "fifty-five", 56: "fifty-six",
    57: "fifty-seven", 58: "fifty-eight", 59: "fifty-nine",
}


def format_time_speech(dt: datetime) -> str:
    """
    Format datetime as natural speech.

    Examples:
        7:00 AM -> "seven AM" or "seven o'clock"
        7:30 PM -> "seven thirty PM"
        12:00 PM -> "noon"
        12:00 AM -> "midnight"
        3:15 PM -> "three fifteen PM"
    """
    hour = dt.hour
    minute = dt.minute

    # Handle special cases
    if hour == 0 and minute == 0:
        return "midnight"
    if hour == 12 and minute == 0:
        return "noon"

    # Convert to 12-hour format
    is_pm = hour >= 12
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12

    period = "PM" if is_pm else "AM"

    # Get hour word
    hour_word = _NUMBER_WORDS.get(hour_12, str(hour_12))

    # Format minutes
    if minute == 0:
        return f"{hour_word} {period}"
    elif minute < 10:
        # "seven oh five PM"
        minute_word = f"oh {_NUMBER_WORDS.get(minute, str(minute))}"
    else:
        minute_word = _NUMBER_WORDS.get(minute, str(minute))

    return f"{hour_word} {minute_word} {period}"


def format_duration_speech(seconds: int) -> str:
    """
    Format duration in seconds as natural speech.

    Examples:
        60 -> "one minute"
        300 -> "five minutes"
        3600 -> "one hour"
        5400 -> "one hour and thirty minutes"
        90 -> "one minute and thirty seconds"
    """
    if seconds < 60:
        if seconds == 1:
            return "one second"
        word = _NUMBER_WORDS.get(seconds, str(seconds))
        return f"{word} seconds"

    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    secs = remaining % 60

    parts = []

    if hours > 0:
        hour_word = _NUMBER_WORDS.get(hours, str(hours))
        parts.append(f"{hour_word} hour{'s' if hours != 1 else ''}")

    if minutes > 0:
        min_word = _NUMBER_WORDS.get(minutes, str(minutes))
        parts.append(f"{min_word} minute{'s' if minutes != 1 else ''}")

    if secs > 0 and hours == 0:  # Only include seconds if no hours
        sec_word = _NUMBER_WORDS.get(secs, str(secs))
        parts.append(f"{sec_word} second{'s' if secs != 1 else ''}")

    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return f"{parts[0]}, {parts[1]}, and {parts[2]}"


def format_temp_speech(temp: int, unit: str = "F") -> str:
    """
    Format temperature for speech.

    Examples:
        72 -> "72 degrees"
        -5 -> "negative 5 degrees"
        0 -> "zero degrees"
    """
    if temp < 0:
        return f"negative {abs(temp)} degrees"
    if temp == 0:
        return "zero degrees"
    return f"{temp} degrees"


def format_ordinal(n: int) -> str:
    """
    Format number as ordinal.

    Examples:
        1 -> "1st"
        2 -> "2nd"
        3 -> "3rd"
        4 -> "4th"
        11 -> "11th"
        21 -> "21st"
    """
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_date_speech(dt: datetime) -> str:
    """
    Format date for speech.

    Examples:
        Feb 4, 2026 -> "February fourth, twenty twenty-six"
    """
    month = dt.strftime("%B")
    day = format_ordinal(dt.day)
    year = dt.year

    return f"{month} {day}, {year}"


def format_day_of_week_speech(dt: datetime) -> str:
    """
    Format day of week for speech.

    Returns: "Monday", "Tuesday", etc.
    """
    return dt.strftime("%A")


def format_for_speech(key: str, value: Any) -> Optional[str]:
    """
    Route to appropriate formatter based on key name.

    Args:
        key: The field name (e.g., "time", "duration", "temp")
        value: The raw value

    Returns:
        Formatted string for speech, or None if no formatter applies
    """
    if key in ("time", "time_str") and isinstance(value, datetime):
        return format_time_speech(value)

    if key in ("duration", "duration_s") and isinstance(value, (int, float)):
        return format_duration_speech(int(value))

    if key in ("temp", "temperature") and isinstance(value, (int, float)):
        return format_temp_speech(int(value))

    if key == "date" and isinstance(value, datetime):
        return format_date_speech(value)

    if key == "day_of_week" and isinstance(value, datetime):
        return format_day_of_week_speech(value)

    # No special formatting needed
    return None
