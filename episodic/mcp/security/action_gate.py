"""Part 3: Action gate — user confirmation for write operations.

Presents proposed actions to the user and requires explicit
confirmation before execution. The gate is the last line of defense
after all deterministic checks pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from episodic.mcp.security.types import SecurityContext


@dataclass
class ActionProposal:
    """A proposed action awaiting user confirmation."""

    tool_name: str
    args: Dict[str, Any]
    summary: str                   # Human-readable action summary
    context: Dict[str, Any]        # Additional context for display
    heightened_scrutiny: bool = False  # True if class_hiding_possible


class ActionGate:
    """User confirmation gate for write operations.

    See spec Part 3 for full requirements:
    - All write-capable tools require confirmation
    - Read-only tools skip the gate
    - Heightened scrutiny when class_hiding_possible
    - Cancelled actions are never executed
    """

    async def confirm(
        self,
        proposal: ActionProposal,
        ctx: SecurityContext,
    ) -> bool:
        """Present action to user and await confirmation.

        Returns True if user confirms, False if cancelled.
        Delegates to ctx.confirmation_handler if available;
        otherwise denies by default (safe fallback).
        """
        if ctx.confirmation_handler is not None:
            context_dict: Dict[str, Any] = {
                **proposal.context,
                "summary": proposal.summary,
                "heightened_scrutiny": proposal.heightened_scrutiny,
            }
            return await ctx.confirmation_handler.confirm(
                proposal.tool_name,
                proposal.args,
                context_dict,
            )

        # No handler configured — deny by default
        return False
