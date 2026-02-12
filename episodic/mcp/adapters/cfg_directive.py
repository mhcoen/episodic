"""
CFG Directive Adapter.

Produces AuthorizationEvents from CFG UtilityQuery objects.
This is the CFG parser's analog of the RegexDirectiveParser.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional, TYPE_CHECKING

from episodic.mcp.dispatch_types import WRITE_INTENTS

if TYPE_CHECKING:
    from episodic.utility.types import UtilityQuery
    from episodic.mcp.security.types import AuthorizationEvent


class CFGDirectiveAdapter:
    """
    Produces AuthorizationEvents from CFG UtilityQuery objects.

    Calendar and email commands parsed by the CFG grammar
    produce authorization events for write/draft/destructive operations.
    Read operations return None (no auth event needed).
    """

    def produce(
        self, query: "UtilityQuery", user_message: str,
    ) -> Optional["AuthorizationEvent"]:
        """
        Create an AuthorizationEvent for write intents, None for reads.

        The action field is set to the intent name (e.g., 'email.reply').
        The dispatch layer translates this to the tool name before
        calling PolicyEngine.check_tool_execution().
        """
        if query.command not in WRITE_INTENTS:
            return None

        from episodic.mcp.security.types import AuthorizationEvent as AE

        return AE(
            action=query.command,
            scope=self._extract_scope(query),
            message_hash=hashlib.sha256(user_message.encode()).hexdigest(),
            timestamp=time.time(),
            source="cfg_parser" if query.source == "cli" else "slash_command",
            session_id=None,
        )

    def _extract_scope(self, query: "UtilityQuery") -> dict:
        """
        Extract scope constraints from CFG args.

        Scope is the security-relevant subset of args that the
        policy engine verifies against the actual tool call args.
        """
        scope: dict = {}
        args = query.args

        # Email: scope on recipient
        if "to" in args and args["to"]:
            scope["recipient"] = args["to"]
        if "email_ref" in args and args["email_ref"]:
            scope["email_ref"] = args["email_ref"]

        # Calendar: scope on attendees
        if "attendees" in args and args["attendees"]:
            scope["attendees"] = args["attendees"]

        # General: scope on the action verb
        scope["action"] = query.command

        return scope
