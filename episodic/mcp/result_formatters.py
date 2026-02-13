"""
MCP Result Formatters.

Converts raw MCP tool call results into human-readable display text.

Raw results from mcp-gsuite arrive as::

    {"content": ["[{json ...}]"], "is_error": False}

The ``content`` list contains JSON strings that need parsing.  Each
formatter extracts structured data and produces display + speech text.
"""

from __future__ import annotations

import html
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_content(raw_result: Any) -> List[Any]:
    """Extract structured data from raw MCP result.

    Handles: ``{"content": ["[{...}]"]}`` → parsed list of dicts.
    Falls back to returning content strings as-is if JSON parsing fails.
    """
    if not isinstance(raw_result, dict):
        return []
    content = raw_result.get("content", [])
    if not isinstance(content, list):
        return []

    items: List[Any] = []
    for piece in content:
        if isinstance(piece, str):
            piece = piece.strip()
            if not piece:
                continue
            try:
                parsed = json.loads(piece)
                if isinstance(parsed, list):
                    items.extend(parsed)
                else:
                    items.append(parsed)
            except (json.JSONDecodeError, ValueError):
                items.append(piece)
        elif isinstance(piece, dict):
            items.append(piece)
    return items


def format_result(
    command: str,
    raw_result: Any,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format an MCP tool call result for display.

    Returns ``(display_text, speech_text, structured_items)``.
    ``structured_items`` are parsed dicts for context tracking.
    """
    items = parse_content(raw_result)

    if command in ("email.search", "email.get"):
        return _format_emails(items)
    elif command in ("calendar.query", "calendar.freebusy"):
        return _format_events(items)
    elif command == "calendar.list":
        return _format_calendars(items)
    elif command == "email.create_draft":
        return _format_draft_created(items)
    elif command == "email.reply":
        return _format_reply_sent(items)
    elif command == "email.forward":
        return _format_forward_created(items)
    elif command in ("calendar.create",):
        return _format_event_created(items)

    # Fallback: show raw content strings
    return _format_generic(items)


# ── Email formatters ──────────────────────────────────────────────


def _format_emails(items: List[Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format email list results."""
    emails = [i for i in items if isinstance(i, dict) and "subject" in i]
    if not emails:
        display = "No emails found."
        return display, display, []

    n = len(emails)
    lines = [f"Found {n} email{'s' if n != 1 else ''}:\n"]
    for i, em in enumerate(emails, 1):
        subj = html.unescape(em.get("subject", "(no subject)"))
        sender = _short_sender(em.get("from", ""))
        date = _short_date(em.get("date", ""))
        snippet = html.unescape(em.get("snippet", ""))
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."

        lines.append(f"  {i}. Subject: {subj}")
        lines.append(f"     From: {sender}  {date}")
        if snippet:
            lines.append(f"     {snippet}")

    display = "\n".join(lines)
    speech = f"You have {n} email{'s' if n != 1 else ''}."
    if n <= 3:
        subjects = [html.unescape(e.get("subject", "no subject")) for e in emails]
        speech += " " + ". ".join(subjects) + "."
    return display, speech, emails


def _short_sender(from_str: str) -> str:
    """Extract display name from 'Name <email>' format."""
    if "<" in from_str:
        return from_str.split("<")[0].strip().strip('"')
    return from_str


def _short_date(date_str: str) -> str:
    """Shorten date to just the essentials."""
    if not date_str:
        return ""
    # Try to extract 'Mon, DD Mon YYYY' or similar
    parts = date_str.split(",")
    if len(parts) >= 2:
        # "Thu, 12 Feb 2026 04:20:32 GMT" → "Feb 12"
        rest = parts[1].strip().split()
        if len(rest) >= 2:
            return f"{rest[1]} {rest[0]}"
    return date_str[:16]


# ── Calendar formatters ───────────────────────────────────────────


def _format_events(items: List[Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format calendar event results."""
    events = [i for i in items if isinstance(i, dict)]
    if not events:
        display = "No events found."
        return display, display, []

    n = len(events)
    lines = [f"{n} event{'s' if n != 1 else ''}:\n"]
    for i, ev in enumerate(events, 1):
        summary = ev.get("summary", "(no title)")
        start = ev.get("start", "")
        end = ev.get("end", "")
        location = ev.get("location", "")

        time_str = _format_event_time(start, end)
        line = f"  {i}. {summary}"
        if time_str:
            line += f" — {time_str}"
        lines.append(line)
        if location:
            lines.append(f"     Location: {location}")

    display = "\n".join(lines)
    speech = f"You have {n} event{'s' if n != 1 else ''}."
    return display, speech, events


def _format_event_time(start: Any, end: Any) -> str:
    """Format event start/end times."""
    if isinstance(start, dict):
        start = start.get("dateTime") or start.get("date", "")
    if isinstance(end, dict):
        end = end.get("dateTime") or end.get("date", "")

    if not start:
        return ""

    # Extract just time portion from ISO datetime
    start_time = _extract_time(str(start))
    end_time = _extract_time(str(end)) if end else ""

    if start_time and end_time:
        return f"{start_time} - {end_time}"
    return start_time or str(start)[:16]


def _extract_time(iso_str: str) -> str:
    """Extract HH:MM from ISO datetime string."""
    if "T" in iso_str:
        time_part = iso_str.split("T")[1][:5]
        # Convert 24h to 12h
        try:
            h, m = int(time_part[:2]), time_part[3:5]
            suffix = "AM" if h < 12 else "PM"
            h = h % 12 or 12
            return f"{h}:{m} {suffix}"
        except (ValueError, IndexError):
            return time_part
    return iso_str[:10]  # Just the date


def _format_calendars(items: List[Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format calendar list results."""
    cals = [i for i in items if isinstance(i, dict)]
    if not cals:
        display = "No calendars found."
        return display, display, []

    n = len(cals)
    lines = [f"{n} calendar{'s' if n != 1 else ''}:\n"]
    for i, cal in enumerate(cals, 1):
        name = cal.get("summary", cal.get("name", "(unnamed)"))
        cal_id = cal.get("id", "")
        primary = " (primary)" if cal.get("primary") else ""
        lines.append(f"  {i}. {name}{primary}")
        if cal_id and cal_id != name:
            lines.append(f"     {cal_id}")

    display = "\n".join(lines)
    speech = f"You have {n} calendar{'s' if n != 1 else ''}."
    return display, speech, cals


# ── Write operation formatters ────────────────────────────────────


def _format_draft_created(
    items: List[Any],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format draft creation result."""
    display = "Draft created."
    speech = "Draft created."
    drafts = [i for i in items if isinstance(i, dict)]
    if drafts:
        d = drafts[0]
        to = d.get("to", "")
        subj = d.get("subject", "")
        if to:
            display = f"Draft created to {to}"
            if subj:
                display += f": {subj}"
            speech = display
    return display, speech, drafts


def _format_reply_sent(
    items: List[Any],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format reply result."""
    display = "Reply sent."
    return display, display, []


def _format_forward_created(
    items: List[Any],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format forward result."""
    display = "Forward draft created."
    return display, display, []


def _format_event_created(
    items: List[Any],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Format event creation result."""
    display = "Event created."
    events = [i for i in items if isinstance(i, dict)]
    if events:
        ev = events[0]
        summary = ev.get("summary", "")
        if summary:
            display = f"Event created: {summary}"
    return display, display, events


# ── Generic fallback ──────────────────────────────────────────────


def _format_generic(items: List[Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Fallback: show content items as text."""
    if not items:
        return "Done.", "Done.", []

    text_parts = []
    dicts = []
    for item in items:
        if isinstance(item, dict):
            dicts.append(item)
            text_parts.append(json.dumps(item, indent=2))
        else:
            text_parts.append(str(item))

    display = "\n".join(text_parts)
    speech = "Done." if dicts else display[:200]
    return display, speech, dicts
