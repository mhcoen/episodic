"""
Input Processor - Single entry point for all input processing.

Both CLI and TestSession call process_input().
"""

from typing import Optional, Tuple

from .events import Event, EventStream, EventKind, EventLevel
from .runtime import RuntimeState


def process_input(text: str, runtime: RuntimeState) -> EventStream:
    """
    Single entry point for all input processing.

    Routes input to appropriate handler based on content:
    - Commands (starting with /) -> command handler
    - Utility phrases (voice grammar) -> utility handler
    - Memory queries -> retrieval system
    - Everything else -> LLM

    Args:
        text: Raw user input
        runtime: Injected dependencies

    Returns:
        EventStream containing all events from processing
    """
    stream = EventStream()
    text = text.strip()

    if not text:
        return stream

    timestamp = runtime.clock.monotonic()

    # Route based on input type
    if text.startswith("/"):
        # CLI command
        return _process_command(text, runtime, stream)
    else:
        # Natural language input
        return _process_utterance(text, runtime, stream)


def _process_command(text: str, runtime: RuntimeState, stream: EventStream) -> EventStream:
    """
    Process a CLI command.

    Commands start with / and are routed to specific handlers.
    """
    # Parse command and args
    parts = text[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    args_str = parts[1] if len(parts) > 1 else ""

    timestamp = runtime.clock.monotonic()

    # Emit routing decision debug event
    runtime.emit_debug(
        kind=EventKind.ROUTER_DECISION.value,
        channel="router",
        fields={
            "input": text,
            "target": "COMMAND",
            "command": cmd,
            "confidence": 1.0,
            "reason": "slash_prefix",
        },
        stream=stream,
    )

    # Check for exit commands
    if cmd in ("exit", "quit", "q", "bye"):
        stream.add_user_event(Event.user(
            kind=EventKind.COMMAND_RESULT.value,
            fields={"command": cmd, "action": "exit", "message": "Goodbye!"},
            timestamp=timestamp,
        ))
        return stream

    # Check for utility commands
    utility_commands = {
        "time", "timer", "alarm", "remind", "weather", "forecast",
        "news", "calc", "note", "play", "pause", "cancel", "undo",
        "dnd", "status", "stop",
    }

    if cmd in utility_commands:
        return _process_utility_command(cmd, args_str, runtime, stream)

    # Other commands - delegate to command registry
    # For now, emit a placeholder event
    stream.add_user_event(Event.user(
        kind=EventKind.COMMAND_RESULT.value,
        fields={
            "command": cmd,
            "args": args_str,
            "status": "delegated",
            "message": f"Command /{cmd} processed",
        },
        timestamp=timestamp,
    ))

    return stream


def _process_utility_command(
    cmd: str,
    args_str: str,
    runtime: RuntimeState,
    stream: EventStream,
) -> EventStream:
    """Process a utility command (timer, weather, etc.)."""
    timestamp = runtime.clock.monotonic()

    # Emit provider call debug event
    runtime.emit_debug(
        kind=EventKind.PROVIDER_CALL.value,
        channel="providers",
        fields={
            "provider": cmd,
            "method": "command",
            "args": args_str,
            "source": "slash_command",
        },
        stream=stream,
    )

    # For now, emit placeholder utility result
    # Real implementation will call handle_utility_command
    stream.add_user_event(Event.user(
        kind=EventKind.UTILITY_RESULT.value,
        fields={
            "command": cmd,
            "args": args_str,
            "status": "ok",
            "display": f"Utility command /{cmd} {args_str}",
        },
        timestamp=timestamp,
    ))

    return stream


def _process_utterance(text: str, runtime: RuntimeState, stream: EventStream) -> EventStream:
    """
    Process natural language input.

    Routes through:
    1. Voice grammar (utility commands)
    2. Memory query detection
    3. LLM fallback
    """
    timestamp = runtime.clock.monotonic()

    # Step 1: Try voice grammar for utility commands
    utility_result = _try_voice_grammar(text, runtime, stream)
    if utility_result is not None:
        return utility_result

    # Step 2: Check for memory queries
    if _is_memory_query(text, runtime, stream):
        return _process_memory_query(text, runtime, stream)

    # Step 3: Route to LLM
    return _process_llm(text, runtime, stream)


def _try_voice_grammar(
    text: str,
    runtime: RuntimeState,
    stream: EventStream,
) -> Optional[EventStream]:
    """
    Try to parse input as utility command via voice grammar.

    Returns EventStream if handled, None to continue routing.
    """
    timestamp = runtime.clock.monotonic()

    # Import voice grammar (lazy to avoid circular imports)
    try:
        from ..routing import route, RouteTarget
        from ..routing.router import RuntimeState as VoiceRuntimeState

        # Create voice runtime state
        voice_state = VoiceRuntimeState(timezone=runtime.timezone)

        # Route the utterance
        result = route(text, voice_state, user_tz=runtime.timezone)

        # Emit parse attempt debug event
        runtime.emit_debug(
            kind=EventKind.PARSE_ATTEMPT.value,
            channel="grammar",
            fields={
                "input": text,
                "target": result.target.name if result.target else "UNKNOWN",
                "confidence": result.confidence,
                "accepted": result.target == RouteTarget.UTILITY,
            },
            stream=stream,
        )

        if result.target == RouteTarget.PREEMPT:
            # Preempt commands (stop, cancel) - handle immediately
            runtime.emit_debug(
                kind=EventKind.ROUTER_DECISION.value,
                channel="router",
                fields={
                    "input": text,
                    "target": "PREEMPT",
                    "confidence": result.confidence,
                    "reason": "preempt_pattern",
                },
                stream=stream,
            )
            # Execute preempt command
            if result.utility_query:
                stream.add_user_event(Event.user(
                    kind=EventKind.UTILITY_EXECUTED.value,
                    fields={
                        "command": result.utility_query.command,
                        "category": result.utility_query.category,
                        "mutating": result.utility_query.command in ("timer_cancel", "alarm_cancel"),
                        "status": "ok",
                    },
                    timestamp=timestamp,
                ))
            return stream

        elif result.target == RouteTarget.UTILITY:
            # Utility command - execute
            runtime.emit_debug(
                kind=EventKind.ROUTER_DECISION.value,
                channel="router",
                fields={
                    "input": text,
                    "target": "UTILITY",
                    "confidence": result.confidence,
                    "reason": "voice_grammar_match",
                },
                stream=stream,
            )
            if result.utility_query:
                stream.add_user_event(Event.user(
                    kind=EventKind.UTILITY_EXECUTED.value,
                    fields={
                        "command": result.utility_query.command,
                        "category": result.utility_query.category,
                        "args": result.utility_query.args,
                        "mutating": result.utility_query.command.endswith("_set"),
                        "status": "ok",
                    },
                    timestamp=timestamp,
                ))
            return stream

        elif result.target == RouteTarget.MQL:
            # Memory query - handled separately
            runtime.emit_debug(
                kind=EventKind.ROUTER_DECISION.value,
                channel="router",
                fields={
                    "input": text,
                    "target": "MQL",
                    "confidence": result.confidence,
                    "reason": "mql_pattern",
                },
                stream=stream,
            )
            return _process_memory_query(text, runtime, stream)

    except ImportError:
        # Voice grammar not available
        pass

    # Not handled by voice grammar
    return None


def _is_memory_query(text: str, runtime: RuntimeState, stream: EventStream) -> bool:
    """Check if input is a memory query."""
    # Simple heuristic for now
    memory_markers = [
        "did we", "did i", "have we", "have i",
        "when did", "what did", "where did",
        "remember when", "recall", "our conversation",
        "last time", "earlier", "before",
    ]
    text_lower = text.lower()
    return any(marker in text_lower for marker in memory_markers)


def _process_memory_query(
    text: str,
    runtime: RuntimeState,
    stream: EventStream,
) -> EventStream:
    """Process a memory/retrieval query."""
    timestamp = runtime.clock.monotonic()

    runtime.emit_debug(
        kind=EventKind.ROUTER_DECISION.value,
        channel="router",
        fields={
            "input": text,
            "target": "MQL",
            "confidence": 0.8,
            "reason": "memory_markers",
        },
        stream=stream,
    )

    # Placeholder - real implementation will call retrieval system
    stream.add_user_event(Event.user(
        kind=EventKind.ASSISTANT_RESPONSE.value,
        fields={
            "source": "memory",
            "text": f"[Memory query: {text}]",
        },
        timestamp=timestamp,
    ))

    return stream


def _process_llm(text: str, runtime: RuntimeState, stream: EventStream) -> EventStream:
    """Process input through LLM."""
    timestamp = runtime.clock.monotonic()

    runtime.emit_debug(
        kind=EventKind.ROUTER_DECISION.value,
        channel="router",
        fields={
            "input": text,
            "target": "LLM",
            "confidence": 0.5,
            "reason": "fallback",
        },
        stream=stream,
    )

    # Check if LLM client is available
    if runtime.llm is None:
        stream.add_user_event(Event.error(
            message="LLM client not configured",
            timestamp=timestamp,
        ))
        return stream

    # Build LLM request (simplified)
    from .runtime import LLMRequest

    request = LLMRequest(
        messages=[{"role": "user", "content": text}],
        model=runtime.get_config("model", "gpt-3.5-turbo"),
        temperature=runtime.get_config("temperature", 0.7),
    )

    # Emit context planning debug event
    runtime.emit_debug(
        kind=EventKind.CONTEXT_PLAN.value,
        channel="context",
        fields={
            "budget": runtime.get_config("max_tokens", 4096),
            "message_count": 1,
            "planned_usage": len(text) // 4,  # rough token estimate
        },
        stream=stream,
    )

    # Call LLM
    try:
        response = runtime.llm.complete(request)

        # Emit LLM metadata debug event
        runtime.emit_debug(
            kind=EventKind.LLM_REQUEST_META.value,
            channel="llm",
            fields={
                "model": request.model,
                "input_tokens": len(text) // 4,
                "output_tokens": response.tokens_used,
            },
            stream=stream,
        )

        # Add response event
        stream.add_user_event(Event.user(
            kind=EventKind.ASSISTANT_RESPONSE.value,
            fields={
                "source": "llm",
                "text": response.content,
                "model": request.model,
            },
            timestamp=runtime.clock.monotonic(),
        ))

    except Exception as e:
        stream.add_user_event(Event.error(
            message=f"LLM error: {str(e)}",
            timestamp=runtime.clock.monotonic(),
        ))

    return stream
