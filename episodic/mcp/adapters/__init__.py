"""
MCP Adapters.

Argument adapters translate CFG-normalized args to MCP tool argument dicts.
The CFGDirectiveAdapter produces AuthorizationEvents from UtilityQuery objects.
"""

from .argument_adapters.calendar import (
    CalendarListAdapter,
    CalendarQueryAdapter,
    CalendarFreeBusyAdapter,
    CalendarCreateAdapter,
    CalendarDeleteAdapter,
    CalendarRescheduleAdapter,
)
from .argument_adapters.email import (
    EmailSearchAdapter,
    EmailGetAdapter,
    EmailAttachmentAdapter,
    EmailDraftAdapter,
    EmailReplyAdapter,
    EmailForwardAdapter,
    EmailDeleteDraftAdapter,
)

# Registry: intent name → adapter class
ARGUMENT_ADAPTERS = {
    "calendar.list":         CalendarListAdapter,
    "calendar.query":        CalendarQueryAdapter,
    "calendar.freebusy":     CalendarFreeBusyAdapter,
    "calendar.create":       CalendarCreateAdapter,
    "calendar.delete":       CalendarDeleteAdapter,
    "calendar.reschedule":   CalendarRescheduleAdapter,
    "email.search":          EmailSearchAdapter,
    "email.get":             EmailGetAdapter,
    "email.get_attachments": EmailAttachmentAdapter,
    "email.create_draft":    EmailDraftAdapter,
    "email.reply":           EmailReplyAdapter,
    "email.forward":         EmailForwardAdapter,
    "email.delete_draft":    EmailDeleteDraftAdapter,
}

__all__ = ["ARGUMENT_ADAPTERS"]
