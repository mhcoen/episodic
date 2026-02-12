"""
MCP Result Context.

Tracks most recent MCP tool results for anaphoric reference resolution
(e.g., "reply to *that* email", "cancel *the* meeting").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MCPResultContext:
    """Tracks most recent MCP tool results for reference resolution."""
    last_emails: List[Dict[str, Any]] = field(default_factory=list)
    last_events: List[Dict[str, Any]] = field(default_factory=list)
    last_drafts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0
    ttl: float = 300.0  # 5 minutes

    def is_stale(self) -> bool:
        """Check if context has expired."""
        if self.timestamp == 0.0:
            return True
        return (time.time() - self.timestamp) > self.ttl

    def update_emails(self, emails: List[Dict[str, Any]]) -> None:
        """Update email result context."""
        self.last_emails = emails
        self.timestamp = time.time()

    def update_events(self, events: List[Dict[str, Any]]) -> None:
        """Update calendar event result context."""
        self.last_events = events
        self.timestamp = time.time()

    def update_drafts(self, drafts: List[Dict[str, Any]]) -> None:
        """Update draft result context."""
        self.last_drafts = drafts
        self.timestamp = time.time()

    def clear(self) -> None:
        """Clear all context."""
        self.last_emails = []
        self.last_events = []
        self.last_drafts = []
        self.timestamp = 0.0


def resolve_anaphoric_ref(
    ref: Optional[str],
    category: str,
    context: MCPResultContext,
) -> Optional[str]:
    """
    Resolve an anaphoric reference ("that", "the", "last") to an ID.

    Returns the ID string if resolution succeeds, None otherwise.
    """
    if ref is None or ref != "last":
        return ref  # Not anaphoric, pass through

    if context.is_stale():
        return None  # Context expired

    if category == "email":
        if context.last_emails:
            first = context.last_emails[0]
            return first.get("id") or first.get("message_id")
    elif category == "calendar":
        if context.last_events:
            first = context.last_events[0]
            return first.get("id") or first.get("event_id")
    elif category == "draft":
        if context.last_drafts:
            first = context.last_drafts[0]
            return first.get("id") or first.get("draft_id")

    return None
