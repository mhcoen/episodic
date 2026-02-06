"""
Speech Templates for Utility Commands.

Each command has multiple template variations to avoid repetitive output.
Templates use {placeholder} syntax for value substitution.
"""

from typing import Dict, List, Optional


# Emoji map for commands (weather emoji comes from result data)
EMOJI_MAP: Dict[str, str] = {
    "time_now": "\u23f0",      # Alarm clock
    "date_today": "\U0001f4c5", # Calendar
    "timer_set": "\u23f1\ufe0f", # Stopwatch
    "timer_fired": "\U0001f514", # Bell
    "alarm_set": "\u23f0",      # Alarm clock
    "alarm_fired": "\U0001f514", # Bell
    "reminder_set": "\U0001f4cc", # Pushpin
    "reminder_fired": "\U0001f514", # Bell
    "media_play": "\u25b6\ufe0f", # Play button
    "media_stop": "\u23f9\ufe0f", # Stop button
    "note_add": "\U0001f4dd",    # Memo
    "status": "\U0001f4ca",      # Chart
}


# Template variations for each command
# Each template is a tuple: (display_template, speech_template)
# If speech_template is None, display_template is used for both
TEMPLATES: Dict[str, List[tuple]] = {
    "time_now": [
        ("It's {time}.", None),
        ("The time is {time}.", None),
        ("{time}.", None),
        ("Right now it's {time}.", None),
    ],

    "date_today": [
        ("Today is {day_of_week}, {date}.", None),
        ("It's {day_of_week}, {date}.", None),
        ("{day_of_week}, {date}.", None),
    ],

    "timer_set": [
        ("Timer set for {duration}.", None),
        ("{duration} timer started.", None),
        ("Got it, {duration}.", "Got it, {duration} timer."),
        ("Starting {duration} timer.", None),
    ],

    "timer_set_labeled": [
        ("{label} timer set for {duration}.", None),
        ("Timer for {label}: {duration}.", "{label} timer, {duration}."),
        ("{duration} for {label}.", "{label}, {duration} timer set."),
        ("Got it, {label} in {duration}.", None),
    ],

    "timer_fired": [
        ("Timer's done!", None),
        ("Time's up!", None),
        ("Timer complete.", None),
        ("Ding! Timer done.", "Timer done."),
    ],

    "timer_fired_labeled": [
        ("{label} timer is done!", None),
        ("Time's up for {label}!", None),
        ("{label} complete.", None),
        ("Ding! {label} done.", "{label} done."),
    ],

    "alarm_set": [
        ("Alarm set for {time}.", None),
        ("I'll wake you at {time}.", "Alarm at {time}."),
        ("Alarm: {time}.", "Alarm set, {time}."),
        ("See you at {time}.", "Alarm for {time}."),
    ],

    "alarm_set_labeled": [
        ("{label} alarm set for {time}.", None),
        ("Alarm for {label} at {time}.", None),
        ("{label}: {time}.", "{label} alarm, {time}."),
    ],

    "alarm_fired": [
        ("Alarm!", None),
        ("Wake up!", None),
        ("Time to get up!", None),
        ("Rise and shine!", None),
    ],

    "alarm_fired_labeled": [
        ("{label}!", None),
        ("It's time for {label}!", None),
        ("{label} alarm!", None),
    ],

    "reminder_set": [
        ("I'll remind you.", "Reminder set."),
        ("Got it, I'll remind you.", "Reminder set."),
        ("Reminder set.", None),
        ("I'll let you know.", "Reminder set."),
    ],

    "reminder_fired": [
        ("Reminder: {text}", None),
        ("Don't forget: {text}", "Reminder, {text}"),
        ("Hey, {text}", "Reminder, {text}"),
        ("Just a reminder: {text}", None),
    ],

    "media_play": [
        ("Playing {station}.", None),
        ("Now playing: {station}.", "Playing {station}."),
        ("{station} is on.", "Playing {station}."),
        ("Here's {station}.", "Playing {station}."),
    ],

    "media_stop": [
        ("Stopped.", None),
        ("Music stopped.", "Stopped."),
        ("Playback stopped.", "Stopped."),
    ],

    "note_add": [
        ("Note saved.", None),
        ("Got it.", "Note saved."),
        ("Noted.", "Note saved."),
        ("I've saved that.", "Note saved."),
    ],

    # Status: no templates — handler builds display with actual timer/alarm data

    # Weather templates - emoji comes from result data
    # Note: {condition} should be capitalized by generator
    "weather_now": [
        ("Currently {temp} degrees in {location}. {condition}. High of {high}, low of {low}.", None),
        ("{temp} degrees and {condition} in {location}. High of {high}, low of {low}.", None),
        ("{location}: {temp} degrees, {condition}. High {high}, low {low}.", "{temp} degrees and {condition} in {location}. High of {high}, low of {low}."),
        ("It's {condition} in {location}. Currently {temp} degrees. High of {high}, low of {low}.", None),
    ],

    "weather_forecast": [
        ("Here's the forecast for {location}.", None),
        ("{location} forecast.", None),
    ],
}


# Random additions for weather to add variety
WEATHER_EXTENSIONS: Dict[str, List[str]] = {
    "clear": [
        " Perfect day outside.",
        " Great weather!",
        " Nice and sunny.",
        "",
    ],
    "cloudy": [
        " Pretty overcast.",
        " Gray skies today.",
        "",
    ],
    "partly cloudy": [
        " Some clouds around.",
        " Mix of sun and clouds.",
        "",
    ],
    "rain": [
        " Grab an umbrella.",
        " Rainy day.",
        " Wet out there.",
        "",
    ],
    "snow": [
        " Bundle up!",
        " Snowy out there.",
        " Winter wonderland.",
        "",
    ],
    "thunderstorm": [
        " Stay safe!",
        " Stormy weather.",
        "",
    ],
    "foggy": [
        " Drive carefully.",
        " Low visibility.",
        "",
    ],
}


def get_templates(command: str, has_label: bool = False) -> List[tuple]:
    """
    Get templates for a command, selecting labeled variant if applicable.

    Args:
        command: The command name (e.g., "timer_set")
        has_label: Whether the command has a label/name

    Returns:
        List of (display_template, speech_template) tuples
    """
    if has_label:
        labeled_key = f"{command}_labeled"
        if labeled_key in TEMPLATES:
            return TEMPLATES[labeled_key]

    return TEMPLATES.get(command, [])


def get_emoji(command: str, result_emoji: Optional[str] = None) -> str:
    """
    Get emoji for a command.

    Weather emoji comes from result data, others from EMOJI_MAP.
    """
    if result_emoji:
        return result_emoji
    return EMOJI_MAP.get(command, "")


def get_weather_extension(condition: str) -> str:
    """
    Get a random weather extension phrase.

    Returns empty string 25% of the time for variety.
    """
    import random

    condition_lower = condition.lower()
    extensions = WEATHER_EXTENSIONS.get(condition_lower, [""])

    return random.choice(extensions)
