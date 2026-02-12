"""
Calendar Command Handlers.

Thin shim that delegates calendar commands to the MCP dispatch layer.
"""

from __future__ import annotations

from typing import Any, Optional

from ..types import UtilityQuery, UtilityResult


# Handler dispatch table for calendar commands
CALENDAR_HANDLERS = {
    "calendar.list",
    "calendar.query",
    "calendar.freebusy",
    "calendar.create",
    "calendar.delete",
    "calendar.reschedule",
}


async def dispatch_calendar_command(
    query: UtilityQuery,
    mcp_client: Any = None,
    pipeline: Any = None,
) -> UtilityResult:
    """
    Dispatch a calendar command via MCP.

    Falls back to error if MCP client is not connected.
    """
    if query.command not in CALENDAR_HANDLERS:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown calendar command: {query.command}",
        )

    from episodic.mcp.dispatch import MCPResolver, dispatch_mcp
    resolver = MCPResolver()
    resolution = resolver.resolve(query.command)

    if resolution is None:
        return UtilityResult.error(
            "unmapped_intent",
            f"No MCP tool mapped for {query.command}",
        )

    result = await dispatch_mcp(
        query=query,
        resolution=resolution,
        user_message=query.raw_input,
        pipeline=pipeline,
        mcp_client=mcp_client,
    )

    if result.success:
        return UtilityResult.ok(
            display=result.display_text,
            speech=result.speech_text,
            _command=query.command,
        )
    return UtilityResult.error(
        result.error_type or "mcp_error",
        result.error_message or result.display_text,
    )
