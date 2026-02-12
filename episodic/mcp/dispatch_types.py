"""
MCP Dispatch Type Definitions.

Core dataclasses for the CFG-to-MCP dispatch pipeline:
MCPResolution, MCPStep, DispatchResult, and sensitivity constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MCPResolution:
    """Result of resolving a UtilityQuery command to an MCP tool."""
    server_id: str
    tool_name: str
    sensitivity: str          # "read", "draft", "write", "destructive"
    requires_auth_event: bool  # True for draft/write/destructive
    schema_fingerprint: str = ""  # From tool discovery, for allowlist lookup


@dataclass(frozen=True)
class MCPStep:
    """A single step in a (possibly multi-step) MCP dispatch."""
    intent: str               # e.g. "email.search", "calendar.delete"
    tool_name: str
    tool_args: Dict[str, Any]
    sensitivity: str
    requires_auth_event: bool
    description: str = ""     # Human-readable summary of this step


@dataclass
class DispatchResult:
    """Result of an MCP dispatch operation."""
    success: bool
    speech_text: str
    display_text: str
    payload: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None     # "forbidden", "cancelled", "tool_error", etc.
    error_message: Optional[str] = None
    latency_us: int = 0
    logged: bool = False
    steps_completed: int = 0
    steps_total: int = 1


# Sensitivity levels
SENSITIVITY_READ = "read"
SENSITIVITY_DRAFT = "draft"
SENSITIVITY_WRITE = "write"
SENSITIVITY_DESTRUCTIVE = "destructive"

# Intents that require authorization events
WRITE_INTENTS: frozenset = frozenset({
    "email.create_draft",
    "email.reply",
    "email.forward",
    "email.delete_draft",
    "calendar.create",
    "calendar.delete",
    "calendar.reschedule",
})

# Default intent-to-tool mapping (populated from config at discovery time)
DEFAULT_INTENT_MAPPING: Dict[str, Dict[str, Any]] = {
    "calendar.list":       {"tool": "list_calendars",        "sensitivity": "read"},
    "calendar.query":      {"tool": "get_calendar_events",   "sensitivity": "read"},
    "calendar.freebusy":   {"tool": "get_calendar_events",   "sensitivity": "read"},
    "calendar.create":     {"tool": "create_calendar_event", "sensitivity": "write"},
    "calendar.delete":     {"tool": "delete_calendar_event", "sensitivity": "destructive"},
    "calendar.reschedule": {"tool": None,                    "sensitivity": "write"},
    "email.search":        {"tool": "query_gmail_emails",    "sensitivity": "read"},
    "email.get":           {"tool": "get_gmail_email",       "sensitivity": "read"},
    "email.get_attachments": {"tool": "get_gmail_attachment", "sensitivity": "read"},
    "email.create_draft":  {"tool": "create_gmail_draft",    "sensitivity": "draft"},
    "email.reply":         {"tool": "reply_gmail_email",     "sensitivity": "write"},
    "email.forward":       {"tool": "create_gmail_draft",    "sensitivity": "write"},
    "email.delete_draft":  {"tool": "delete_gmail_draft",    "sensitivity": "destructive"},
}
