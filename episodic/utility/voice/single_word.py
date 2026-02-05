"""
Single-Word Routing for Voice Grammar.

Quick routing for unambiguous single-word utterances.
"""

from typing import Optional, Tuple, Dict

from ..types import UtilityQuery


# Map single-word utterances to (category, command, confidence)
# None means ambiguous - do not route
SINGLE_WORD_ROUTING: Dict[str, Optional[Tuple[str, str, float]]] = {
    "time": ("time", "time_now", 0.90),
    "date": ("time", "date_today", 0.90),
    "weather": ("weather", "weather_now", 0.85),
    "temp": ("weather", "weather_now", 0.85),
    "temperature": ("weather", "weather_now", 0.85),
    "forecast": ("weather", "weather_forecast", 0.85),
    "news": ("news", "news_headlines", 0.85),
    "timers": ("timer", "timer_list", 0.85),
    "alarms": ("alarm", "alarm_list", 0.85),
    "reminders": ("reminder", "remind_list", 0.85),
    "notes": ("note", "note_list", 0.85),
    "status": ("system", "status", 0.85),

    # Ambiguous - do not route
    "tomorrow": None,   # date_query? forecast? alarm modifier?
    "today": ("time", "date_today", 0.85),  # Unambiguous as single word
    "stop": None,       # Handled by preempt
    "cancel": None,     # Needs target
    "play": None,       # Needs target
}


def route_single_word(word: str, raw_input: str = "") -> Optional[UtilityQuery]:
    """
    Route single-word utterances.

    Args:
        word: The single word to route
        raw_input: Original input string for the query

    Returns:
        UtilityQuery if routed, None if ambiguous or unknown
    """
    word_lower = word.lower().strip()

    if word_lower not in SINGLE_WORD_ROUTING:
        return None

    routing = SINGLE_WORD_ROUTING[word_lower]
    if routing is None:
        return None

    category, command, confidence = routing
    return UtilityQuery(
        category=category,
        command=command,
        args={},
        confidence=confidence,
        source="voice",
        raw_input=raw_input or word,
    )
