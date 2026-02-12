"""
Calendar and Email Grammar Rules for Voice Grammar.

Produces GrammarRule lists for calendar and email commands.
Argument extractors for calendar/email-specific args.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .lexer import Token
from .confidence import ParseFeatures
from .grammar_types import GrammarRule


def build_calendar_rules() -> List[GrammarRule]:
    """Build grammar rules for calendar commands."""
    return [
        # calendar.list — "list my calendars", "show my calendars"
        GrammarRule(
            name="cal_list",
            category="calendar",
            command="calendar.list",
            patterns=[
                ["QUERY", "KW_CALENDARS"],             # "which calendars"
                ["ACTION_SEARCH", "KW_CALENDARS"],     # "list calendars"
            ],
            required_args=[],
            optional_args=[],
            is_exact_template=True,
        ),

        # calendar.query — "check my calendar", "what's on my calendar tomorrow"
        GrammarRule(
            name="cal_query",
            category="calendar",
            command="calendar.query",
            patterns=[
                ["QUERY", "KW_CALENDAR"],              # "what calendar" / "check calendar"
                ["QUERY", "KW_MEETING"],               # "what meetings"
                ["QUERY", "KW_CALENDAR", "TIME_RANGE"],  # "check calendar this week"
                ["QUERY", "KW_MEETING", "TIME_RANGE"],   # "check meetings this week"
                ["QUERY", "KW_CALENDAR", "RELATIVE_DAY"],  # "check calendar tomorrow"
                ["QUERY", "KW_MEETING", "RELATIVE_DAY"],   # "check meetings tomorrow"
            ],
            required_args=[],
            optional_args=["time_min", "time_max"],
            is_exact_template=True,
        ),

        # calendar.freebusy — "when am I free", "am I busy tomorrow"
        GrammarRule(
            name="cal_freebusy",
            category="calendar",
            command="calendar.freebusy",
            patterns=[
                ["QUERY", "KW_BUSY"],                  # "am I busy" / "when am I free"
                ["QUERY", "KW_BUSY", "TIME_RANGE"],    # "am I free this week"
                ["QUERY", "KW_BUSY", "RELATIVE_DAY"],  # "am I busy tomorrow"
            ],
            required_args=[],
            optional_args=["time_min", "time_max"],
            is_exact_template=True,
        ),

        # calendar.create — "schedule a meeting with Bob at 3pm"
        GrammarRule(
            name="cal_create",
            category="calendar",
            command="calendar.create",
            patterns=[
                ["ACTION_SET", "KW_MEETING"],          # "create meeting" / "book meeting"
                ["ACTION_SET", "KW_CALENDAR"],         # "create event"
                ["KW_CALENDAR", "KW_MEETING"],         # "schedule a meeting"
            ],
            required_args=[],
            optional_args=["summary", "start", "end", "attendees", "location"],
            is_exact_template=False,
        ),

        # calendar.delete — "cancel the meeting tomorrow"
        GrammarRule(
            name="cal_delete",
            category="calendar",
            command="calendar.delete",
            patterns=[
                ["ACTION_CANCEL", "KW_MEETING"],       # "cancel meeting"
                ["ACTION_CANCEL", "KW_CALENDAR"],      # "delete event"
            ],
            required_args=[],
            optional_args=["event_ref"],
            is_exact_template=False,
        ),

        # calendar.reschedule — "reschedule the meeting to 4pm"
        GrammarRule(
            name="cal_reschedule",
            category="calendar",
            command="calendar.reschedule",
            patterns=[
                ["ACTION_RESCHEDULE", "KW_MEETING"],   # "reschedule meeting"
                ["ACTION_RESCHEDULE", "KW_CALENDAR"],  # "reschedule event"
            ],
            required_args=[],
            optional_args=["event_ref", "new_start", "new_end"],
            is_exact_template=False,
        ),
    ]


def build_email_rules() -> List[GrammarRule]:
    """Build grammar rules for email commands."""
    return [
        # email.search — "check my email", "do I have new mail"
        GrammarRule(
            name="email_search",
            category="email",
            command="email.search",
            patterns=[
                ["QUERY", "KW_EMAIL"],                 # "check email" / "check inbox"
                ["QUERY", "KW_UNREAD", "KW_EMAIL"],    # "check unread email"
                ["QUERY", "KW_EMAIL", "KW_UNREAD"],    # "check email unread"
                ["ACTION_SEARCH", "KW_EMAIL"],         # "search email" / "find mail"
            ],
            required_args=[],
            optional_args=["query", "from_addr", "unread_only", "max_results"],
            is_exact_template=True,
        ),

        # email.search (from) — "email from Alice"
        GrammarRule(
            name="email_from",
            category="email",
            command="email.search",
            patterns=[
                ["KW_EMAIL", "PREP_FROM"],             # "email from ..."
                ["QUERY", "KW_EMAIL", "PREP_FROM"],    # "check email from ..."
            ],
            required_args=["from_addr"],
            optional_args=["query"],
            is_exact_template=False,
        ),

        # email.search (about) — "find email about budget"
        GrammarRule(
            name="email_about",
            category="email",
            command="email.search",
            patterns=[
                ["KW_EMAIL", "KW_ABOUT"],              # "email about ..."
                ["QUERY", "KW_EMAIL", "KW_ABOUT"],     # "check email about ..."
                ["ACTION_SEARCH", "KW_EMAIL", "KW_ABOUT"],  # "search email about ..."
            ],
            required_args=["query"],
            optional_args=[],
            is_exact_template=False,
        ),

        # email.create_draft — "draft an email to Bob"
        GrammarRule(
            name="email_draft",
            category="email",
            command="email.create_draft",
            patterns=[
                ["ACTION_DRAFT", "KW_EMAIL"],          # "draft email" / "compose email"
                ["ACTION_DRAFT", "KW_DRAFT"],          # "draft a draft"
                ["ACTION_SET", "KW_DRAFT"],            # "create draft"
                ["KW_DRAFT", "PREP_TO"],               # "draft to ..."
            ],
            required_args=[],
            optional_args=["to", "subject", "body"],
            is_exact_template=False,
        ),

        # email.reply — "reply to that email"
        GrammarRule(
            name="email_reply",
            category="email",
            command="email.reply",
            patterns=[
                ["ACTION_DRAFT", "ACTION_REPLY"],      # "draft a reply"
                ["ACTION_REPLY"],                      # "reply" / "respond"
                ["ACTION_REPLY", "PREP_TO"],           # "reply to ..."
                ["ACTION_SEND", "ACTION_REPLY"],       # "send reply"
            ],
            required_args=[],
            optional_args=["email_ref", "body", "send"],
            is_exact_template=False,
        ),

        # email.forward — "forward that email to Carol"
        GrammarRule(
            name="email_forward",
            category="email",
            command="email.forward",
            patterns=[
                ["ACTION_FORWARD"],                    # "forward"
                ["ACTION_FORWARD", "KW_EMAIL"],        # "forward email"
                ["ACTION_FORWARD", "PREP_TO"],         # "forward to ..."
            ],
            required_args=[],
            optional_args=["email_ref", "to"],
            is_exact_template=False,
        ),

        # email.delete_draft — "delete the draft"
        GrammarRule(
            name="email_del_draft",
            category="email",
            command="email.delete_draft",
            patterns=[
                ["ACTION_CANCEL", "KW_DRAFT"],         # "delete draft"
            ],
            required_args=[],
            optional_args=["draft_ref"],
            is_exact_template=False,
        ),
    ]


# --- Argument extractors ---

def extract_calendar_args(
    command: str,
    tokens: List[Token],
    words: List[str],
    consumed: int,
    user_tz: str = "America/Chicago",
) -> Dict[str, Any]:
    """Extract calendar command arguments from token stream."""
    args: Dict[str, Any] = {}

    if command == "calendar.list":
        return args

    if command in ("calendar.query", "calendar.freebusy"):
        time_min, time_max = _extract_time_range(tokens, user_tz)
        if time_min:
            args["time_min"] = time_min.isoformat()
        if time_max:
            args["time_max"] = time_max.isoformat()
        return args

    if command == "calendar.create":
        remaining_words = words[consumed:]
        summary = _extract_text_after(tokens, consumed, stop_kinds={
            "PREP_AT", "PREP_ON", "PREP_FOR", "PREP_IN",
        })
        if summary:
            args["summary"] = summary
        person = _extract_person(tokens, consumed)
        if person:
            args["attendees"] = [person]
        return args

    if command == "calendar.delete":
        ref = _extract_text_after(tokens, consumed, stop_kinds=set())
        if ref:
            args["event_ref"] = ref
        return args

    if command == "calendar.reschedule":
        ref = _extract_text_after(tokens, consumed, stop_kinds={"PREP_TO"})
        if ref:
            args["event_ref"] = ref
        return args

    return args


def extract_email_args(
    command: str,
    tokens: List[Token],
    words: List[str],
    consumed: int,
) -> Dict[str, Any]:
    """Extract email command arguments from token stream."""
    args: Dict[str, Any] = {}

    if command == "email.search":
        # Check for from_addr pattern — search ALL tokens (from is in pattern)
        person = _extract_person(tokens, 0)
        if person:
            args["from_addr"] = person

        # Check for about/query pattern — search ALL tokens
        query = _extract_query_text(tokens, 0)
        if query:
            args["query"] = query

        # Check for unread
        for t in tokens:
            if t.kind == "KW_UNREAD":
                args["unread_only"] = True
                break

        return args

    if command == "email.create_draft":
        person = _extract_person(tokens, consumed)
        if person:
            args["to"] = person
        query = _extract_query_text(tokens, consumed)
        if query:
            args["subject"] = query
        return args

    if command == "email.reply":
        # Default send=True for "reply to", send=False for "draft a reply"
        args["send"] = True
        for t in tokens:
            if t.kind in ("ACTION_DRAFT", "KW_DRAFT"):
                args["send"] = False
                break
            # Also check value for "compose"
            if t.value.lower() in ("compose", "draft a"):
                args["send"] = False
                break

        body = _extract_body_text(tokens, consumed)
        if body:
            args["body"] = body

        ref = _extract_email_ref(tokens, consumed)
        if ref:
            args["email_ref"] = ref
        return args

    if command == "email.forward":
        person = _extract_person(tokens, consumed)
        if person:
            args["to"] = person
        ref = _extract_email_ref(tokens, consumed)
        if ref:
            args["email_ref"] = ref
        return args

    if command == "email.delete_draft":
        ref = _extract_text_after(tokens, consumed, stop_kinds=set())
        if ref:
            args["draft_ref"] = ref
        return args

    return args


# --- Helper extractors ---

def _extract_time_range(
    tokens: List[Token],
    user_tz: str = "America/Chicago",
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Extract time range from tokens."""
    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    for t in tokens:
        if t.kind == "TIME_RANGE":
            val = t.value.lower()
            if val == "this week":
                # Monday to Sunday
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                start -= timedelta(days=start.weekday())
                end = start + timedelta(days=7)
                return start, end
            elif val == "next week":
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                start -= timedelta(days=start.weekday())
                start += timedelta(days=7)
                end = start + timedelta(days=7)
                return start, end
            elif val == "this morning":
                start = now.replace(hour=6, minute=0, second=0, microsecond=0)
                end = now.replace(hour=12, minute=0, second=0, microsecond=0)
                return start, end
            elif val == "this afternoon":
                start = now.replace(hour=12, minute=0, second=0, microsecond=0)
                end = now.replace(hour=18, minute=0, second=0, microsecond=0)
                return start, end

        elif t.kind == "RELATIVE_DAY":
            val = t.value.lower()
            if val == "today":
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=1)
                return start, end
            elif val == "tomorrow":
                start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                end = start + timedelta(days=1)
                return start, end
            elif val == "yesterday":
                start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                end = start + timedelta(days=1)
                return start, end

    return None, None


def _extract_person(tokens: List[Token], start: int) -> Optional[str]:
    """Extract person name or email after 'from'/'to'/'with' preposition."""
    for i in range(start, len(tokens)):
        if tokens[i].kind in ("PREP_FROM", "PREP_TO") and i + 1 < len(tokens):
            # Collect subsequent WORD tokens as name
            name_parts = []
            for j in range(i + 1, len(tokens)):
                if tokens[j].kind in ("WORD", "KW_EMAIL"):
                    name_parts.append(tokens[j].value)
                elif tokens[j].kind == "ARTICLE":
                    continue  # skip "the"
                else:
                    break
            if name_parts:
                return " ".join(name_parts)
    return None


def _extract_text_after(
    tokens: List[Token],
    start: int,
    stop_kinds: set,
) -> Optional[str]:
    """Extract free text from tokens, stopping at stop_kinds."""
    parts = []
    for i in range(start, len(tokens)):
        if tokens[i].kind in stop_kinds:
            break
        if tokens[i].kind not in ("ARTICLE", "POLITENESS", "PRONOUN"):
            parts.append(tokens[i].value)
    return " ".join(parts) if parts else None


def _extract_query_text(tokens: List[Token], start: int) -> Optional[str]:
    """Extract query text after 'about'/'regarding' keyword."""
    for i in range(start, len(tokens)):
        if tokens[i].kind == "KW_ABOUT" and i + 1 < len(tokens):
            parts = [t.value for t in tokens[i + 1:]]
            return " ".join(parts) if parts else None
    return None


def _extract_body_text(tokens: List[Token], start: int) -> Optional[str]:
    """Extract body text after 'saying'/'with' keyword."""
    for i in range(start, len(tokens)):
        val = tokens[i].value.lower()
        if val in ("saying", "with") and i + 1 < len(tokens):
            parts = [t.value for t in tokens[i + 1:]]
            return " ".join(parts) if parts else None
    return None


def _extract_email_ref(tokens: List[Token], start: int) -> Optional[str]:
    """Extract email reference (anaphoric or descriptive)."""
    for i in range(start, len(tokens)):
        val = tokens[i].value.lower()
        if val in ("that", "this", "the") and i + 1 < len(tokens):
            next_kind = tokens[i + 1].kind
            if next_kind == "KW_EMAIL":
                return "last"  # Anaphoric reference
    return None
