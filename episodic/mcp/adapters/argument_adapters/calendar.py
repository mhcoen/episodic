"""
Calendar Argument Adapters.

Translate CFG-normalized calendar args to mcp-gsuite tool argument dicts.
All mcp-gsuite tools require __user_id__ (account email).
Calendar tools accept optional __calendar_id__ (defaults to "primary").
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from episodic.mcp.adapters.base import ArgumentAdapter
from episodic.mcp.dispatch_types import MCPResolution


class CalendarListAdapter(ArgumentAdapter):
    """
    Adapts calendar.list → list_calendars.

    CFG produces:    {}
    mcp-gsuite expects: {__user_id__: str}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "__user_id__": self._resolve_account(query_args, config),
        }


class CalendarQueryAdapter(ArgumentAdapter):
    """
    Adapts calendar.query → get_calendar_events.

    CFG produces:    {time_min: str?, time_max: str?, calendar_id: str?}
    mcp-gsuite expects: {__user_id__: str, time_min: str?, time_max: str?,
                         __calendar_id__: str?}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "__user_id__": self._resolve_account(query_args, config),
        }
        if query_args.get("time_min"):
            result["time_min"] = query_args["time_min"]
        if query_args.get("time_max"):
            result["time_max"] = query_args["time_max"]
        if query_args.get("calendar_id"):
            result["__calendar_id__"] = query_args["calendar_id"]
        return result


class CalendarFreeBusyAdapter(ArgumentAdapter):
    """
    Adapts calendar.freebusy → get_calendar_events.

    Same tool as CalendarQueryAdapter; adapter interprets gaps as free slots.

    CFG produces:    {time_min: str?, time_max: str?, duration_s: int?}
    mcp-gsuite expects: {__user_id__: str, time_min: str?, time_max: str?}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "__user_id__": self._resolve_account(query_args, config),
        }
        if query_args.get("time_min"):
            result["time_min"] = query_args["time_min"]
        if query_args.get("time_max"):
            result["time_max"] = query_args["time_max"]
        return result


class CalendarCreateAdapter(ArgumentAdapter):
    """
    Adapts calendar.create → create_calendar_event.

    CFG produces:    {summary: str, start: str?, end: str?,
                      attendees: list?, location: str?, description: str?}
    mcp-gsuite expects: {__user_id__: str, summary: str, start: str?,
                         end: str?, attendees: list?, location: str?,
                         description: str?, __calendar_id__: str?}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "__user_id__": self._resolve_account(query_args, config),
        }
        for key in ("summary", "start", "end", "attendees",
                     "location", "description"):
            if query_args.get(key):
                result[key] = query_args[key]
        if query_args.get("calendar_id"):
            result["__calendar_id__"] = query_args["calendar_id"]
        return result


class CalendarDeleteAdapter(ArgumentAdapter):
    """
    Adapts calendar.delete → delete_calendar_event.

    CFG produces:    {event_ref: str} or {event_id: str}
    mcp-gsuite expects: {__user_id__: str, event_id: str,
                         __calendar_id__: str?}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "__user_id__": self._resolve_account(query_args, config),
        }
        # event_id may come from decomposition step, event_ref from CFG
        event_id = query_args.get("event_id") or query_args.get("event_ref", "")
        if event_id:
            result["event_id"] = event_id
        if query_args.get("calendar_id"):
            result["__calendar_id__"] = query_args["calendar_id"]
        return result


class CalendarRescheduleAdapter(ArgumentAdapter):
    """
    Adapts calendar.reschedule (decomposed to delete + create).

    This adapter is a no-op — the MCPDecomposer handles decomposition.
    Included for registry completeness.
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Decomposition is handled by MCPDecomposer
        return {
            "__user_id__": self._resolve_account(query_args, config),
            "event_ref": query_args.get("event_ref", ""),
            "new_start": query_args.get("new_start"),
            "new_end": query_args.get("new_end"),
        }
