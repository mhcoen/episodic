"""
Email Argument Adapters.

Translate CFG-normalized email args to mcp-gsuite tool argument dicts.
All mcp-gsuite tools require __user_id__ (account email).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from episodic.mcp.adapters.base import ArgumentAdapter
from episodic.mcp.dispatch_types import MCPResolution


class EmailSearchAdapter(ArgumentAdapter):
    """
    Adapts email.search → query_gmail_emails.

    CFG produces:    {query: str?, from_addr: str?, unread_only: bool?,
                      max_results: int?}
    mcp-gsuite expects: {__user_id__: str, query: str, max_results: int?}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Build Gmail query string from structured args
        parts = []
        raw_query = query_args.get("query", "")
        if raw_query:
            # Expand shorthand: bare "unread" → Gmail syntax "is:unread"
            normalized = raw_query.strip().lower()
            if normalized == "unread" or normalized == "is:unread":
                parts.append("is:unread")
            else:
                parts.append(raw_query)
        if query_args.get("from_addr"):
            parts.append(f"from:{query_args['from_addr']}")
        if query_args.get("unread_only"):
            parts.append("is:unread")

        result: Dict[str, Any] = {
            "__user_id__": self._resolve_account(query_args, config),
            "query": " ".join(parts) if parts else "is:unread",
        }
        if query_args.get("max_results"):
            result["max_results"] = query_args["max_results"]
        return result


class EmailGetAdapter(ArgumentAdapter):
    """
    Adapts email.get → get_gmail_email.

    CFG produces:    {email_ref: str} or {email_id: str}
    mcp-gsuite expects: {__user_id__: str, email_id: str}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        email_id = query_args.get("email_id") or query_args.get("email_ref", "")
        return {
            "__user_id__": self._resolve_account(query_args, config),
            "email_id": email_id,
        }


class EmailAttachmentAdapter(ArgumentAdapter):
    """
    Adapts email.get_attachments → get_gmail_attachment.

    CFG produces:    {email_ref: str}
    mcp-gsuite expects: {__user_id__: str, email_id: str,
                         attachment_id: str, save_to_disk: str?}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        email_id = query_args.get("email_id") or query_args.get("email_ref", "")
        result: Dict[str, Any] = {
            "__user_id__": self._resolve_account(query_args, config),
            "email_id": email_id,
        }
        if query_args.get("attachment_id"):
            result["attachment_id"] = query_args["attachment_id"]
        return result


class EmailDraftAdapter(ArgumentAdapter):
    """
    Adapts email.create_draft → create_gmail_draft.

    CFG produces:    {to: str?, subject: str?, body: str?}
    mcp-gsuite expects: {__user_id__: str, to: str?, subject: str?,
                         body: str?}
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
        for key in ("to", "subject", "body"):
            if query_args.get(key):
                result[key] = query_args[key]
        return result


class EmailReplyAdapter(ArgumentAdapter):
    """
    Adapts email.reply → reply_gmail_email.

    Dual-mode: send=true for "reply to", send=false for "draft a reply".

    CFG produces:    {email_ref: str, body: str, send: bool}
    mcp-gsuite expects: {__user_id__: str, email_id: str, body: str?,
                         send: bool}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        email_id = query_args.get("email_id") or query_args.get("email_ref", "")
        result: Dict[str, Any] = {
            "__user_id__": self._resolve_account(query_args, config),
            "email_id": email_id,
            "send": query_args.get("send", True),
        }
        if query_args.get("body"):
            result["body"] = query_args["body"]
        return result


class EmailForwardAdapter(ArgumentAdapter):
    """
    Adapts email.forward → create_gmail_draft (composed forward).

    No native forward tool in mcp-gsuite. Adapter composes a draft with
    [Fwd] subject prefix and quoted original body.

    CFG produces:    {email_ref: str, to: str}
    mcp-gsuite expects: {__user_id__: str, to: str, subject: str, body: str}

    Requires prior email.get step to populate original_subject and original_body.
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
        if query_args.get("to"):
            result["to"] = query_args["to"]

        # If original email content is available (from prior step)
        original_subject = query_args.get("original_subject", "")
        original_body = query_args.get("original_body", "")

        if original_subject or original_body:
            result["subject"] = f"[Fwd] {original_subject}"
            result["body"] = (
                f"---------- Forwarded message ----------\n"
                f"{original_body}"
            )
        else:
            # Placeholder — will be populated by multi-step dispatch
            if query_args.get("subject"):
                result["subject"] = f"[Fwd] {query_args['subject']}"
            if query_args.get("body"):
                result["body"] = query_args["body"]

        return result


class EmailDeleteDraftAdapter(ArgumentAdapter):
    """
    Adapts email.delete_draft → delete_gmail_draft.

    CFG produces:    {draft_ref: str}
    mcp-gsuite expects: {__user_id__: str, draft_id: str}
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        draft_id = query_args.get("draft_id") or query_args.get("draft_ref", "")
        return {
            "__user_id__": self._resolve_account(query_args, config),
            "draft_id": draft_id,
        }
