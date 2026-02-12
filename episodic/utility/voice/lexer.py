"""
Lexer for Voice Grammar.

Tokenizes normalized input using maximal munch (longest match) semantics.
Multi-word tokens match before single words.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Token:
    """A token produced by the lexer."""
    kind: str
    value: str
    position: int


# Core single-word tokens (no calendar/email — those come from plugins)
_CORE_TOKENS: Dict[str, str] = {
    # Query markers
    "what": "QUERY", "when": "QUERY", "where": "QUERY",
    "how": "QUERY", "which": "QUERY",
    "what's": "QUERY", "check": "QUERY", "show": "QUERY",
    "list": "QUERY", "get": "QUERY",

    # Action verbs
    "set": "ACTION_SET", "start": "ACTION_SET", "create": "ACTION_SET",
    "cancel": "ACTION_CANCEL", "delete": "ACTION_CANCEL", "remove": "ACTION_CANCEL",
    "play": "ACTION_PLAY", "pause": "ACTION_PAUSE", "resume": "ACTION_RESUME",
    "stop": "ACTION_STOP", "skip": "ACTION_SKIP",
    "add": "ACTION_ADD", "remind": "ACTION_REMIND",

    # Domain keywords
    "timer": "KW_TIMER", "timers": "KW_TIMER",
    "alarm": "KW_ALARM", "alarms": "KW_ALARM",
    "reminder": "KW_REMIND", "reminders": "KW_REMIND",
    "weather": "KW_WEATHER", "forecast": "KW_WEATHER", "temperature": "KW_WEATHER",
    "temp": "KW_WEATHER", "high": "KW_WEATHER_HIGH", "low": "KW_WEATHER_LOW",
    "news": "KW_NEWS", "headlines": "KW_NEWS",
    "note": "KW_NOTE", "notes": "KW_NOTE",
    "time": "KW_TIME", "clock": "KW_TIME",
    "date": "KW_DATE", "day": "KW_DATE",
    "radio": "KW_RADIO", "station": "KW_RADIO",
    "music": "KW_MUSIC", "song": "KW_MUSIC",

    # Time units
    "second": "TIME_UNIT", "seconds": "TIME_UNIT", "sec": "TIME_UNIT", "secs": "TIME_UNIT",
    "minute": "TIME_UNIT", "minutes": "TIME_UNIT", "min": "TIME_UNIT", "mins": "TIME_UNIT",
    "hour": "TIME_UNIT", "hours": "TIME_UNIT", "hr": "TIME_UNIT", "hrs": "TIME_UNIT",

    # Duration words
    "couple": "DURATION_WORD", "few": "DURATION_WORD",
    "half": "DURATION_WORD", "quarter": "DURATION_WORD",

    # Relative time
    "today": "RELATIVE_DAY", "tomorrow": "RELATIVE_DAY",
    "tonight": "RELATIVE_DAY", "morning": "RELATIVE_DAY",
    "afternoon": "RELATIVE_DAY", "evening": "RELATIVE_DAY",

    # AM/PM
    "am": "AMPM", "pm": "AMPM",

    # Prepositions
    "at": "PREP_AT", "in": "PREP_IN", "for": "PREP_FOR",
    "on": "PREP_ON", "to": "PREP_TO", "from": "PREP_FROM",

    # Articles and fillers
    "a": "ARTICLE", "an": "ARTICLE", "the": "ARTICLE",
    "please": "POLITENESS", "could": "POLITENESS", "would": "POLITENESS",
    "me": "PRONOUN", "my": "PRONOUN", "i": "PRONOUN",

    # Conjunctions
    "and": "CONJ", "or": "CONJ", "then": "CONJ", "also": "CONJ",

    # Named (for labels)
    "called": "NAMED", "named": "NAMED", "labeled": "NAMED",
}


def _build_single_word_map() -> Dict[str, str]:
    """Build the complete single-word map from core + plugin tokens."""
    word_map = dict(_CORE_TOKENS)

    try:
        from episodic.mcp.plugins import get_plugin_registry
        registry = get_plugin_registry()
        if not registry.initialized:
            registry.register_all()
        for reg in registry.registered():
            for td in reg.tokens:
                word = td.word.lower()
                # Plugin tokens can add or override non-core words
                word_map[word] = td.token_kind
    except ImportError:
        pass

    return word_map


class Lexer:
    """
    Lexer with REQUIRED maximal munch semantics.
    Multi-word tokens match before single words.
    """

    # Ordered by length descending for maximal munch
    MULTIWORD_TOKENS: List[Tuple[str, str]] = [
        # 4+ words
        ("do not disturb", "KW_DND"),
        ("quarter past", "TIME_QUARTER_PAST"),
        ("quarter to", "TIME_QUARTER_TO"),
        ("half past", "TIME_HALF_PAST"),

        # 3 words
        ("wake me up", "KW_ALARM"),
        ("in the morning", "AMPM"),
        ("in the afternoon", "AMPM"),
        ("in the evening", "AMPM"),
        ("at night", "AMPM"),

        # 2 words
        ("this week", "TIME_RANGE"),
        ("next week", "TIME_RANGE"),
        ("this morning", "TIME_RANGE"),
        ("this afternoon", "TIME_RANGE"),
        ("set up", "ACTION_SET"),
        ("look for", "ACTION_SEARCH"),
        ("look up", "ACTION_SEARCH"),
        ("draft a", "ACTION_DRAFT"),
        ("what is", "QUERY"),
        ("tell me", "QUERY"),
        ("show me", "QUERY"),
        ("give me", "QUERY"),
        ("how is", "QUERY"),
        ("turn on", "ACTION_ON"),
        ("turn off", "ACTION_OFF"),
        ("put on", "ACTION_ON"),
    ]

    def __init__(self) -> None:
        self.SINGLE_WORD_MAP: Dict[str, str] = _build_single_word_map()

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize with maximal munch."""
        tokens: List[Token] = []
        words = text.split()
        pos = 0

        while pos < len(words):
            matched = False

            # Try multiword tokens (longest first)
            for phrase, token_type in self.MULTIWORD_TOKENS:
                phrase_words = phrase.split()
                phrase_len = len(phrase_words)

                if pos + phrase_len <= len(words):
                    candidate = " ".join(words[pos:pos + phrase_len]).lower()
                    if candidate == phrase:
                        tokens.append(Token(token_type, candidate, pos))
                        pos += phrase_len
                        matched = True
                        break

            if not matched:
                word = words[pos]
                word_lower = word.lower()

                # Check if it's a number
                if word.isdigit():
                    token_type = "NUMBER"
                # Check single-word map
                elif word_lower in self.SINGLE_WORD_MAP:
                    token_type = self.SINGLE_WORD_MAP[word_lower]
                else:
                    token_type = "WORD"

                tokens.append(Token(token_type, word, pos))
                pos += 1

        return tokens

    def get_token_kinds(self, tokens: List[Token]) -> List[str]:
        """Extract just the kinds from a token list."""
        return [t.kind for t in tokens]

    def get_token_values(self, tokens: List[Token]) -> List[str]:
        """Extract just the values from a token list."""
        return [t.value for t in tokens]
