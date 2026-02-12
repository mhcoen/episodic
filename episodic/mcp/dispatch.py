"""
MCP Dispatch Layer.

Resolves abstract CFG intents to concrete MCP tool invocations,
handles multi-step decomposition, and wires through the security pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .dispatch_types import (
    MCPResolution,
    MCPStep,
    DispatchResult,
    WRITE_INTENTS,
    DEFAULT_INTENT_MAPPING,
    SENSITIVITY_READ,
)
from .result_context import MCPResultContext, resolve_anaphoric_ref

if TYPE_CHECKING:
    from episodic.utility.types import UtilityQuery
    from episodic.mcp.security.pipeline import SecurityPipeline

logger = logging.getLogger(__name__)


class MCPResolver:
    """
    Resolves abstract intent names to concrete MCP tool invocations.

    Does NOT hardcode tool names. The mapping is populated at
    tool-discovery time from the MCP tools/list response and
    stored in config.
    """

    def __init__(self, mapping: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        raw = mapping or DEFAULT_INTENT_MAPPING
        self._mapping: Dict[str, MCPResolution] = {}
        for intent, info in raw.items():
            tool = info.get("tool")
            if tool is None:
                continue  # Decomposed intent, no single tool
            sensitivity = info.get("sensitivity", "read")
            self._mapping[intent] = MCPResolution(
                server_id=info.get("server_id", "gsuite"),
                tool_name=tool,
                sensitivity=sensitivity,
                requires_auth_event=sensitivity != "read",
                schema_fingerprint=info.get("schema_fingerprint", ""),
            )

    def resolve(self, command: str) -> Optional[MCPResolution]:
        """
        Look up the MCP tool for a command string.

        Returns None if no MCP tool is mapped for this intent,
        signaling fallthrough to built-in handlers.
        """
        return self._mapping.get(command)

    def has_mapping(self, command: str) -> bool:
        """Check if a command has any mapping (including decomposed)."""
        return command in self._mapping or command in DEFAULT_INTENT_MAPPING


class MCPDecomposer:
    """
    Decomposes multi-step commands into ordered MCP call sequences.

    Each step independently traverses the security pipeline.
    Write steps require their own AuthorizationEvent.
    """

    def __init__(self, resolver: MCPResolver) -> None:
        self._resolver = resolver

    def decompose(
        self,
        command: str,
        args: Dict[str, Any],
        result_context: MCPResultContext,
    ) -> List[MCPStep]:
        """
        Returns ordered list of MCP steps.
        Single-step commands return a list of length 1.
        """
        # Multi-step: email.reply with descriptive reference
        if command == "email.reply" and args.get("email_ref") and args["email_ref"] != "last":
            return self._decompose_reply(args)

        # Multi-step: email.forward
        if command == "email.forward":
            return self._decompose_forward(args, result_context)

        # Multi-step: calendar.reschedule
        if command == "calendar.reschedule":
            return self._decompose_reschedule(args, result_context)

        # Single-step: resolve directly
        resolution = self._resolver.resolve(command)
        if resolution is None:
            return []

        return [MCPStep(
            intent=command,
            tool_name=resolution.tool_name,
            tool_args=args,
            sensitivity=resolution.sensitivity,
            requires_auth_event=resolution.requires_auth_event,
            description=f"Execute {command}",
        )]

    def _decompose_reply(self, args: Dict[str, Any]) -> List[MCPStep]:
        """Decompose email.reply with descriptive ref → search + reply."""
        search_resolution = self._resolver.resolve("email.search")
        reply_resolution = self._resolver.resolve("email.reply")
        if not search_resolution or not reply_resolution:
            return []

        ref = args.get("email_ref", "")
        steps = [
            MCPStep(
                intent="email.search",
                tool_name=search_resolution.tool_name,
                tool_args={"query": ref, "max_results": 1},
                sensitivity="read",
                requires_auth_event=False,
                description=f"Search for email: {ref}",
            ),
            MCPStep(
                intent="email.reply",
                tool_name=reply_resolution.tool_name,
                tool_args={k: v for k, v in args.items() if k != "email_ref"},
                sensitivity=reply_resolution.sensitivity,
                requires_auth_event=True,
                description="Reply to found email",
            ),
        ]
        return steps

    def _decompose_forward(
        self, args: Dict[str, Any], context: MCPResultContext,
    ) -> List[MCPStep]:
        """Decompose email.forward → search + get + create_draft."""
        search_res = self._resolver.resolve("email.search")
        get_res = self._resolver.resolve("email.get")
        draft_res = self._resolver.resolve("email.forward")
        if not search_res or not get_res or not draft_res:
            return []

        ref = args.get("email_ref", "")
        steps = []

        # Anaphoric resolution: only "last" can resolve from context
        resolved_id = None
        if ref == "last":
            resolved_id = resolve_anaphoric_ref(ref, "email", context)

        # Step 1: Search (unless we already have a resolved ID)
        if not resolved_id:
            steps.append(MCPStep(
                intent="email.search",
                tool_name=search_res.tool_name,
                tool_args={"query": ref or "recent", "max_results": 1},
                sensitivity="read",
                requires_auth_event=False,
                description=f"Search for email: {ref}",
            ))

        # Step 2: Get email content
        steps.append(MCPStep(
            intent="email.get",
            tool_name=get_res.tool_name,
            tool_args={"email_id": resolved_id or "__STEP_RESULT__"},
            sensitivity="read",
            requires_auth_event=False,
            description="Get email content for forwarding",
        ))

        # Step 3: Create forward draft
        steps.append(MCPStep(
            intent="email.forward",
            tool_name=draft_res.tool_name,
            tool_args={"to": args.get("to", ""), "__forward__": True},
            sensitivity="write",
            requires_auth_event=True,
            description=f"Forward email to {args.get('to', 'recipient')}",
        ))

        return steps

    def _decompose_reschedule(
        self, args: Dict[str, Any], context: MCPResultContext,
    ) -> List[MCPStep]:
        """Decompose calendar.reschedule → query + delete + create."""
        query_res = self._resolver.resolve("calendar.query")
        delete_res = self._resolver.resolve("calendar.delete")
        create_res = self._resolver.resolve("calendar.create")
        if not query_res or not delete_res or not create_res:
            return []

        ref = args.get("event_ref", "")
        steps = []

        # Anaphoric resolution: only "last" can resolve from context
        resolved_id = None
        if ref == "last":
            resolved_id = resolve_anaphoric_ref(ref, "calendar", context)

        # Step 1: Find the event (unless we already have a resolved ID)
        if not resolved_id:
            steps.append(MCPStep(
                intent="calendar.query",
                tool_name=query_res.tool_name,
                tool_args={"query": ref},
                sensitivity="read",
                requires_auth_event=False,
                description=f"Find event: {ref}",
            ))

        # Step 2: Delete old event
        steps.append(MCPStep(
            intent="calendar.delete",
            tool_name=delete_res.tool_name,
            tool_args={"event_id": resolved_id or "__STEP_RESULT__"},
            sensitivity="destructive",
            requires_auth_event=True,
            description="Delete existing event",
        ))

        # Step 3: Create new event
        create_args = {k: v for k, v in args.items()
                       if k not in ("event_ref", "new_start", "new_end")}
        if "new_start" in args:
            create_args["start"] = args["new_start"]
        if "new_end" in args:
            create_args["end"] = args["new_end"]
        steps.append(MCPStep(
            intent="calendar.create",
            tool_name=create_res.tool_name,
            tool_args=create_args,
            sensitivity="write",
            requires_auth_event=True,
            description="Create rescheduled event",
        ))

        return steps


# Module-level result context (singleton per session)
_result_context = MCPResultContext()


def get_result_context() -> MCPResultContext:
    """Get the module-level result context."""
    return _result_context


def update_result_context(command: str, raw_result: Any) -> None:
    """Update result context from a tool call result."""
    ctx = get_result_context()
    if command.startswith("email.search") or command == "email.get":
        emails = _extract_list(raw_result, "emails", "messages")
        if emails:
            ctx.update_emails(emails)
    elif command.startswith("calendar."):
        events = _extract_list(raw_result, "events", "items")
        if events:
            ctx.update_events(events)
    elif command == "email.create_draft":
        drafts = _extract_list(raw_result, "drafts", "draft")
        if drafts:
            ctx.update_drafts(drafts)


def _extract_list(result: Any, *keys: str) -> List[Dict[str, Any]]:
    """Try to extract a list from a result object."""
    if isinstance(result, dict):
        for key in keys:
            if key in result and isinstance(result[key], list):
                return result[key]
        # Single item? Wrap in list.
        if "id" in result:
            return [result]
    elif isinstance(result, list):
        return result
    return []


async def dispatch_mcp(
    query: "UtilityQuery",
    resolution: MCPResolution,
    user_message: str,
    pipeline: Optional["SecurityPipeline"] = None,
    mcp_client: Any = None,
    confirm_handler: Any = None,
) -> DispatchResult:
    """
    Main async dispatch function for MCP tool calls.

    Steps:
    1. Adapt arguments (done by caller)
    2. Produce authorization event for write intents
    3. Security pipeline check
    4. Action gate (confirmation for writes)
    5. Execute via MCP client
    6. Process inbound response (security)
    7. Update result context
    8. Format result
    """
    from .adapters.cfg_directive import CFGDirectiveAdapter

    start = time.monotonic_ns()

    # 1. Produce authorization event
    auth_adapter = CFGDirectiveAdapter()
    auth_event = auth_adapter.produce(query, user_message)

    # 2. Security check (if pipeline available)
    if pipeline is not None:
        from .security.types import SecurityContext, TrustLevel, ContentType, PolicyConfig

        ctx = SecurityContext(
            mode="client",
            source_type="mcp_server",
            source_id=resolution.server_id,
            trust_level=TrustLevel.UNTRUSTED,
            content_type=ContentType.PLAINTEXT,
            policy=PolicyConfig(),
            confirmation_handler=confirm_handler,
        )

        # Translate intent action to tool name for policy engine
        effective_event = None
        if auth_event is not None:
            from .security.types import AuthorizationEvent
            effective_event = AuthorizationEvent(
                action=resolution.tool_name,
                scope=auth_event.scope,
                message_hash=auth_event.message_hash,
                timestamp=auth_event.timestamp,
                source=auth_event.source,
                session_id=auth_event.session_id,
            )

        gate_result = pipeline.check_tool_execution(
            tool=resolution.tool_name,
            args={},
            ctx=ctx,
            auth_event=effective_event,
        )

        if not gate_result.allowed:
            return DispatchResult(
                success=False,
                speech_text=f"I can't do that: {gate_result.reason}",
                display_text=gate_result.reason,
                error_type="forbidden",
                error_message=gate_result.reason,
                latency_us=(time.monotonic_ns() - start) // 1000,
                logged=True,
            )

    # 3. Execute via MCP client
    if mcp_client is None:
        return DispatchResult(
            success=False,
            speech_text="MCP client not available.",
            display_text="MCP client not connected.",
            error_type="not_connected",
            error_message="No MCP client",
            latency_us=(time.monotonic_ns() - start) // 1000,
        )

    try:
        raw_result = await mcp_client.call_tool(
            resolution.server_id, resolution.tool_name, {}
        )
    except Exception as e:
        from .security.error_sanitizer import sanitize_error
        sanitized = sanitize_error(e)
        return DispatchResult(
            success=False,
            speech_text="There was a problem with that request.",
            display_text=sanitized["error"]["message"],
            error_type="tool_error",
            error_message=str(e),
            latency_us=(time.monotonic_ns() - start) // 1000,
            logged=True,
        )

    # 4. Process inbound (if pipeline available)
    if pipeline is not None:
        processed = pipeline.process_inbound(
            content=str(raw_result),
            ctx=ctx,
        )
        display_content = processed.content
    else:
        display_content = str(raw_result)

    # 5. Update result context
    update_result_context(query.command, raw_result)

    # 6. Format result
    latency_us = (time.monotonic_ns() - start) // 1000
    return DispatchResult(
        success=True,
        speech_text=f"Done with {query.command}.",
        display_text=display_content,
        payload=raw_result if isinstance(raw_result, dict) else {},
        latency_us=latency_us,
        logged=True,
    )
