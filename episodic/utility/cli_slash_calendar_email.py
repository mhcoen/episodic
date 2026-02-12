"""
Slash Command Parsers for Calendar and Email.

Parses /cal, /calendar, /calendars, /schedule, /email, /mail,
/inbox, /draft, /reply, /forward into UtilityQuery objects.
"""

from __future__ import annotations

from .dispatcher import create_utility_query
from .types import UtilityQuery


def parse_cal_args(args_str: str) -> UtilityQuery:
    """Parse /cal <time_range> → calendar.query."""
    args = {}
    if args_str:
        args["query"] = args_str.strip()
    return create_utility_query(
        "calendar", "calendar.query",
        args=args,
        source="cli",
        raw_input=f"/cal {args_str}" if args_str else "/cal",
    )


def parse_schedule_args(args_str: str) -> UtilityQuery:
    """Parse /schedule <description> → calendar.create."""
    args = {}
    if args_str:
        args["summary"] = args_str.strip()
    return create_utility_query(
        "calendar", "calendar.create",
        args=args,
        source="cli",
        raw_input=f"/schedule {args_str}" if args_str else "/schedule",
    )


def parse_email_args(args_str: str) -> UtilityQuery:
    """Parse /email <filter> → email.search."""
    args = {}
    if args_str:
        text = args_str.strip().lower()
        if text.startswith("from "):
            args["from_addr"] = text[5:].strip()
        elif text.startswith("about "):
            args["query"] = text[6:].strip()
        elif text == "unread":
            args["unread_only"] = True
        else:
            args["query"] = args_str.strip()
    return create_utility_query(
        "email", "email.search",
        args=args,
        source="cli",
        raw_input=f"/email {args_str}" if args_str else "/email",
    )


def parse_draft_args(args_str: str) -> UtilityQuery:
    """Parse /draft to <person> about <subject> → email.create_draft."""
    args = {}
    if args_str:
        text = args_str.strip()
        # Parse "to X about Y" or "to X re: Y"
        if " about " in text.lower():
            parts = text.lower().split(" about ", 1)
            to_part = parts[0].strip()
            if to_part.startswith("to "):
                args["to"] = to_part[3:].strip()
            args["subject"] = parts[1].strip()
        elif text.lower().startswith("to "):
            args["to"] = text[3:].strip()
        else:
            args["subject"] = text
    return create_utility_query(
        "email", "email.create_draft",
        args=args,
        source="cli",
        raw_input=f"/draft {args_str}" if args_str else "/draft",
    )


def parse_reply_args(args_str: str) -> UtilityQuery:
    """Parse /reply <body> → email.reply."""
    args = {"send": True}
    if args_str:
        text = args_str.strip()
        if text.lower().startswith("saying "):
            args["body"] = text[7:].strip()
        else:
            args["body"] = text
    return create_utility_query(
        "email", "email.reply",
        args=args,
        source="cli",
        raw_input=f"/reply {args_str}" if args_str else "/reply",
    )


def parse_forward_args(args_str: str) -> UtilityQuery:
    """Parse /forward to <person> → email.forward."""
    args = {}
    if args_str:
        text = args_str.strip()
        if text.lower().startswith("to "):
            args["to"] = text[3:].strip()
        else:
            args["to"] = text
    return create_utility_query(
        "email", "email.forward",
        args=args,
        source="cli",
        raw_input=f"/forward {args_str}" if args_str else "/forward",
    )
