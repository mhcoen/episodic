"""Lexer token definitions contributed by the gsuite plugin.

These tokens are registered in the lexer's SINGLE_WORD_MAP via
the TokenRegistry, rather than being hardcoded in lexer.py.
"""

from episodic.mcp.plugins._protocol import TokenDefinition

GSUITE_TOKENS = [
    # Calendar domain
    TokenDefinition("calendar", "KW_CALENDAR"),
    TokenDefinition("meeting", "KW_MEETING"),
    TokenDefinition("agenda", "KW_CALENDAR"),
    TokenDefinition("schedule", "KW_CALENDAR"),
    TokenDefinition("event", "KW_CALENDAR"),
    TokenDefinition("events", "KW_CALENDAR"),
    TokenDefinition("busy", "KW_BUSY"),
    TokenDefinition("free", "KW_BUSY"),
    TokenDefinition("available", "KW_BUSY"),
    TokenDefinition("reschedule", "ACTION_RESCHEDULE"),
    TokenDefinition("postpone", "ACTION_RESCHEDULE"),
    TokenDefinition("book", "ACTION_SET"),
    TokenDefinition("calendars", "KW_CALENDARS"),

    # Email domain
    TokenDefinition("email", "KW_EMAIL"),
    TokenDefinition("mail", "KW_EMAIL"),
    TokenDefinition("inbox", "KW_EMAIL"),
    TokenDefinition("gmail", "KW_EMAIL"),
    TokenDefinition("message", "KW_EMAIL"),
    TokenDefinition("unread", "KW_UNREAD"),
    TokenDefinition("draft", "KW_DRAFT"),
    TokenDefinition("send", "ACTION_SEND"),
    TokenDefinition("reply", "ACTION_REPLY"),
    TokenDefinition("respond", "ACTION_REPLY"),
    TokenDefinition("forward", "ACTION_FORWARD"),
    TokenDefinition("compose", "ACTION_DRAFT"),
    TokenDefinition("search", "ACTION_SEARCH"),
    TokenDefinition("find", "ACTION_SEARCH"),
    TokenDefinition("about", "KW_ABOUT"),
    TokenDefinition("regarding", "KW_ABOUT"),
    TokenDefinition("subject", "KW_ABOUT"),
    TokenDefinition("yesterday", "RELATIVE_DAY"),
]
