"""
CLI Integration for Utility Commands.

Wires up slash commands to the utility dispatcher.
Handles argument parsing and output display.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from ..config import config
from .types import UtilityQuery, UtilityResult, ResultStatus
from .dispatcher import dispatch_utility, create_utility_query
from .scheduler import Scheduler
from .adapters.base import AdapterRegistry
from .adapters.radio import RadioAdapter, NullRadioAdapter
from .audio import AudioPlayerImpl, create_audio_player
from .data_refresh import DataRefreshScheduler, get_data_refresh_scheduler


from episodic.utility.service_lifecycle import (  # noqa: F401  (re-exported)
    get_scheduler, get_audio_player, get_adapter_registry,
    start_data_refresh_scheduler, shutdown_utility_services,
    _ensure_utility_schema, _get_mcp_client_manager, _get_security_pipeline,
    _restore_task_callback, _handle_task_fire,
)

_last_result: Optional[UtilityResult] = None


from episodic.utility.arg_parsing import (  # noqa: F401  (re-exported)
    _parse_duration,
    _parse_timer_args,
    _parse_time,
    _parse_remind_args,
)
def _handle_plugin_slash_command(sc, args_str: str, cmd: str) -> Optional[UtilityQuery]:
    """Build a UtilityQuery for a plugin slash command.

    Uses the extraction pipeline's matched_domains + domain scoping
    to create a query that will be dispatched via async MCP path.

    Since the user explicitly typed a slash command, we always produce
    a query — extraction refines it, but on failure or null intent we
    fall back to a simple passthrough query for the domain.
    """
    default_command = (
        f"{sc.domain}.query" if sc.domain == "calendar"
        else f"{sc.domain}.search"
    )

    if not args_str.strip():
        # No arguments — default to a basic query for the domain
        default_args: dict = {}
        if sc.domain == "email":
            default_args = {"unread_only": True}
        return create_utility_query(
            sc.domain, default_command,
            args=default_args, source="cli",
            raw_input=f"/{cmd}",
        )

    # Use extraction to parse the natural language args
    try:
        import asyncio
        from episodic.mcp.extraction import (
            extract_intent,
            check_dispatchability,
        )
        from episodic.mcp.extraction.prompt import get_intents_for_domains

        # Scope extraction to the slash command's domain
        domains = {sc.domain}
        intents = get_intents_for_domains(domains)

        async def _extract():
            return await extract_intent(
                args_str,
                matched_domains=domains,
                contacts={},
            )

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _extract())
                result = future.result()
        except RuntimeError:
            result = asyncio.run(_extract())

        verdict = check_dispatchability(result, intents)

        if verdict.dispatchable and verdict.intent:
            return create_utility_query(
                sc.domain, verdict.intent,
                args=verdict.args,
                source="cli",
                confidence=1.0,
                raw_input=f"/{cmd} {args_str}",
            )
        elif verdict.missing_required_args:
            missing = ", ".join(verdict.missing_required_args)
            from ..color_utils import secho_color
            secho_color(f"Missing required info: {missing}", fg="yellow")
            return None
        elif verdict.is_unknown_command:
            from ..color_utils import secho_color
            secho_color(
                f"Not sure what you mean by that. "
                f"Try: /{cmd} {', '.join(sc.completions[:3])}",
                fg="yellow",
            )
            return None
        # Null intent or other non-dispatchable: fall through to default
    except Exception:
        pass  # Extraction failure: fall through to default

    # Fallback: user typed /cmd <text>, pass text as raw query
    return create_utility_query(
        sc.domain, default_command,
        args={"query": args_str},
        source="cli",
        raw_input=f"/{cmd} {args_str}",
    )


def handle_utility_command(cmd: str, args_str: str) -> Optional[UtilityResult]:
    """
    Handle a utility slash command.

    Args:
        cmd: Command name without slash (e.g., "timer", "alarm")
        args_str: Arguments as a string

    Returns:
        UtilityResult if handled, None if not a utility command
    """
    global _last_result

    user_tz = config.get("timezone", "America/Chicago")

    # First, validate arguments and create query without accessing scheduler
    # This allows validation errors to be returned quickly
    query = None

    # Parse command and create query
    if cmd == "stop":
        query = create_utility_query("system", "stop", source="cli")

    elif cmd == "timer":
        args = args_str.split() if args_str else []
        if not args:
            # Show timer status
            query = create_utility_query("timer", "timer_status", source="cli")
        else:
            duration_s, label = _parse_timer_args(args)
            if duration_s is None:
                return UtilityResult.error("invalid_duration", f"Could not parse duration: {args_str}")

            query = create_utility_query(
                "timer", "timer_set",
                args={"duration_s": duration_s, "label": label},
                source="cli",
                raw_input=f"timer {args_str}",
            )

    elif cmd == "alarm":
        args = args_str.split() if args_str else []
        if not args:
            # List alarms
            query = create_utility_query("alarm", "alarm_list", source="cli")
        else:
            # Validate time before creating query (avoids hitting scheduler/DB)
            alarm_time = _parse_time(args[0], user_tz)
            if alarm_time is None:
                return UtilityResult.error("invalid_time", f"Could not parse time: {args[0]}")
            # First arg is time, rest is label
            label = " ".join(args[1:]) if len(args) > 1 else None
            query = create_utility_query(
                "alarm", "alarm_set",
                args={"time": args[0], "label": label},
                source="cli",
                raw_input=f"alarm {args_str}",
            )

    elif cmd == "time":
        query = create_utility_query("time", "time_now", source="cli")

    elif cmd == "calc":
        if not args_str:
            return UtilityResult.error("missing_expression", "Usage: /calc <expression>")
        query = create_utility_query(
            "calc", "calc_expr",
            args={"expr": args_str},
            source="cli",
            raw_input=f"calc {args_str}",
        )

    elif cmd == "note":
        if not args_str:
            # List notes
            query = create_utility_query("note", "note_list", source="cli")
        else:
            query = create_utility_query(
                "note", "note_add",
                args={"text": args_str},
                source="cli",
                raw_input=f"note {args_str}",
            )

    elif cmd == "remind":
        if not args_str:
            # List reminders
            query = create_utility_query("reminder", "remind_list", source="cli")
        else:
            reminder_text, duration_s, alarm_time = _parse_remind_args(args_str, user_tz)
            if reminder_text is None:
                return UtilityResult.error(
                    "invalid_format",
                    "Usage: /remind <text> in <duration> or /remind <text> at <time>"
                )

            args_dict = {"text": reminder_text}
            if duration_s:
                args_dict["minutes"] = duration_s // 60
            elif alarm_time:
                args_dict["at_time"] = alarm_time.isoformat()

            query = create_utility_query(
                "reminder", "remind_set",
                args=args_dict,
                source="cli",
                raw_input=f"remind {args_str}",
            )

    elif cmd == "play":
        if not args_str:
            query = create_utility_query("media", "media_status", source="cli")
        else:
            query = create_utility_query(
            "media", "media_play",
            args={"query": args_str, "source": "radio"},
            source="cli",
            raw_input=f"play {args_str}",
        )

    elif cmd == "pause":
        query = create_utility_query("media", "media_pause", source="cli")

    elif cmd == "cancel":
        args = args_str.split() if args_str else []
        if not args:
            # Cancel most recent timer/alarm
            query = create_utility_query("system", "cancel", source="cli")
        elif args[0].lower() == "timer":
            query = create_utility_query("timer", "timer_cancel", source="cli")
        elif args[0].lower() == "alarm":
            query = create_utility_query("alarm", "alarm_cancel", source="cli")
        else:
            query = create_utility_query("system", "cancel", source="cli")

    elif cmd == "undo":
        query = create_utility_query("system", "undo", source="cli")

    elif cmd == "dnd":
        args = args_str.split() if args_str else []
        if not args:
            # Toggle DND
            query = create_utility_query("system", "dnd_on", source="cli")
        elif args[0].lower() == "on":
            query = create_utility_query("system", "dnd_on", source="cli")
        elif args[0].lower() == "off":
            query = create_utility_query("system", "dnd_off", source="cli")
        else:
            # Duration specified
            duration_s = _parse_duration(args[0])
            if duration_s:
                query = create_utility_query(
                    "system", "dnd_on",
                    args={"duration_minutes": duration_s // 60},
                    source="cli",
                )
            else:
                return UtilityResult.error("invalid_duration", f"Could not parse DND duration: {args[0]}")

    elif cmd == "status":
        query = create_utility_query("system", "status", source="cli")

    elif cmd == "weather":
        # /weather [location]
        place = args_str.strip() if args_str else "current"
        query = create_utility_query(
            "weather", "weather_now",
            args={"place": place},
            source="cli",
            raw_input=f"weather {args_str}" if args_str else "weather",
        )

    elif cmd == "forecast":
        # /forecast [location]
        place = args_str.strip() if args_str else "current"
        query = create_utility_query(
            "weather", "weather_forecast",
            args={"place": place},
            source="cli",
            raw_input=f"forecast {args_str}" if args_str else "forecast",
        )

    elif cmd == "news":
        # /news [category]
        category = args_str.strip().lower() if args_str else "general"
        query = create_utility_query(
            "news", "news_headlines",
            args={"category": category},
            source="cli",
            raw_input=f"news {args_str}" if args_str else "news",
        )

    else:
        # Check plugin registry for slash commands
        from episodic.mcp.plugins import get_plugin_registry
        registry = get_plugin_registry()
        if not registry.initialized:
            registry.register_all()

        slash_cmd = f"/{cmd}"
        sc = registry.get_slash_command(slash_cmd)
        if sc is not None:
            # Plugin slash command — use extraction pipeline
            query = _handle_plugin_slash_command(sc, args_str or "", cmd)
        else:
            # Not a utility command
            return None

    if query is None:
        return None

    # Calendar/Email: delegate to async MCP dispatch
    if query.category in ("calendar", "email"):
        return _execute_async_utility_query(query)

    # Initialize only the services needed for this command category
    scheduler = None
    adapter_registry = None
    audio_player = None

    # Categories that need scheduler
    if query.category in ("timer", "alarm", "reminder"):
        scheduler = get_scheduler()
        audio_player = get_audio_player()

    # Categories that need adapter registry
    if query.category == "media":
        adapter_registry = get_adapter_registry()

    # System commands: stop needs adapters, others may need scheduler
    if query.category == "system":
        if query.command in ("stop",):
            adapter_registry = get_adapter_registry()
            audio_player = get_audio_player()
        elif query.command in ("cancel", "status"):
            scheduler = get_scheduler()
            adapter_registry = get_adapter_registry()
            audio_player = get_audio_player()

    # Ensure utility schema exists (for event logging)
    _ensure_utility_schema()

    # Get database connection
    from ..db_connection import get_connection

    # Dispatch the query
    with get_connection() as conn:
        result = dispatch_utility(
            query,
            conn=conn,
            user_tz=user_tz,
            scheduler=scheduler,
            audio_player=audio_player,
            adapter_registry=adapter_registry,
            last_result=_last_result,
        )

    # Store for undo/repeat
    _last_result = result

    return result


def display_utility_result(result: UtilityResult) -> None:
    """Display utility result directly (no word tokenizer)."""
    from ..color_utils import secho_color
    from ..configuration import get_system_color

    if result.status == ResultStatus.OK:
        # Generate varied speech from result data
        from .speech import SpeechGenerator

        generator = SpeechGenerator.get_instance()
        command = result.data.get("_command", "")

        if command and result.data:
            # Include display/speech text so the generator fallback
            # path can use them for commands without templates
            values = dict(result.data)
            values.setdefault("display_text", result.display_text)
            values.setdefault("speech_text", result.speech_text)
            display_text, speech_text = generator.generate(command, values)
        else:
            display_text = result.display_text or "Done"
            speech_text = result.speech_text or display_text

        # Print directly — utility text is pre-formatted and short,
        # the streaming word tokenizer destroys emoji spacing
        color = get_system_color()
        secho_color(display_text, fg=color)

        # TTS if voice mode enabled
        if config.get("voice_mode") and config.get("voice_tts_enabled", True):
            from ..voice import get_voice_manager

            voice_manager = get_voice_manager()
            if voice_manager.is_active:
                voice_manager.speak(speech_text)
    elif result.status == ResultStatus.ERROR:
        error_msg = result.error_message or "An error occurred"
        secho_color(f"Error: {error_msg}", fg="red")
    elif result.status == ResultStatus.CONFIRM:
        text = result.display_text or "Confirm?"
        secho_color(text, fg="yellow")
    elif result.status == ResultStatus.FALLBACK:
        secho_color("Command not understood", fg="yellow")


def is_utility_command(cmd: str) -> bool:
    """Check if a command is a utility command."""
    utility_commands = {
        "stop", "timer", "alarm", "time", "calc", "note",
        "remind", "play", "pause", "cancel", "undo", "dnd", "status",
        "weather", "forecast", "news",
    }
    if cmd in utility_commands:
        return True

    # Check plugin registry for slash commands
    try:
        from episodic.mcp.plugins import get_plugin_registry
        registry = get_plugin_registry()
        if not registry.initialized:
            registry.register_all()
        if registry.has_slash_command(f"/{cmd}"):
            return True
    except ImportError:
        pass

    return False


def handle_voice_utterance(text: str) -> Optional[UtilityResult]:
    """
    Handle a natural language utterance that may be a utility command.

    This is the integration point for voice/typed input that should be
    checked for utility commands before falling through to the LLM.

    Order:
    1. Extraction keyword gate (sub-millisecond). If plugin domains match,
       try the LLM extraction pipeline before grammar parse.
    2. Standard route() for core commands (timer, alarm, etc.) and MQL.

    Args:
        text: Raw user input

    Returns:
        UtilityResult if handled as utility, None to fall through to chat
    """
    global _last_result

    # Try extraction for plugin domains BEFORE grammar parse
    extraction_result = _try_extraction_for_voice(text)
    if extraction_result is not None:
        return extraction_result

    from ..routing import route, RouteTarget
    from ..routing.router import RuntimeState

    user_tz = config.get("timezone", "America/Chicago")

    # Create runtime state
    # TODO: Track actual media/TTS state from adapters
    state = RuntimeState(timezone=user_tz)

    # Route the utterance
    result = route(text, state, user_tz=user_tz)

    # Handle based on routing decision
    if result.target == RouteTarget.PREEMPT:
        # Execute preempt command
        return _execute_utility_query(result.utility_query)

    elif result.target == RouteTarget.UTILITY:
        # Execute utility command
        utility_result = _execute_utility_query(result.utility_query)
        # If dispatcher rejected (low confidence fallback), let it fall through to LLM
        if utility_result and utility_result.status == ResultStatus.FALLBACK:
            return None
        return utility_result

    elif result.target == RouteTarget.MQL:
        # Fall through to chat - MQL will be handled by chat
        return None

    else:  # LLM fallback
        return None


def _try_extraction_for_voice(text: str) -> Optional[UtilityResult]:
    """Try extraction pipeline for plugin domains on a voice utterance.

    Returns UtilityResult if extraction produced a dispatchable intent,
    None to fall through to grammar parse.
    """
    try:
        from episodic.mcp.extraction.gate import matched_domains
        domains = matched_domains(text)
        if not domains:
            return None

        import asyncio
        from episodic.mcp.extraction import extract_intent, check_dispatchability
        from episodic.mcp.extraction.prompt import get_intents_for_domains

        intents = get_intents_for_domains(domains)

        async def _extract():
            return await extract_intent(
                text,
                matched_domains=domains,
                contacts={},
            )

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _extract())
                result = future.result()
        except RuntimeError:
            result = asyncio.run(_extract())

        verdict = check_dispatchability(result, intents)

        if verdict.dispatchable and verdict.intent:
            domain = verdict.intent.split(".")[0] if "." in verdict.intent else "unknown"
            query = create_utility_query(
                domain, verdict.intent,
                args=verdict.args,
                source="voice",
                confidence=1.0,
                raw_input=text,
            )
            return _execute_utility_query(query)
    except Exception:
        pass  # Extraction failed — fall through to grammar parse

    return None


def _execute_utility_query(query: UtilityQuery) -> Optional[UtilityResult]:
    """Execute a UtilityQuery and return the result."""
    global _last_result

    # MCP-backed categories: delegate to async dispatch
    _mcp_categories = {"calendar", "email"}
    try:
        from episodic.mcp.plugins import get_plugin_registry
        registry = get_plugin_registry()
        for reg in registry.registered():
            for sc in reg.slash_commands:
                if sc.domain:
                    _mcp_categories.add(sc.domain)
    except ImportError:
        pass

    if query.category in _mcp_categories:
        return _execute_async_utility_query(query)

    user_tz = config.get("timezone", "America/Chicago")

    # Initialize only the services needed for this command category
    scheduler = None
    adapter_registry = None
    audio_player = None

    # Categories that need scheduler
    if query.category in ("timer", "alarm", "reminder"):
        scheduler = get_scheduler()
        audio_player = get_audio_player()

    # Categories that need adapter registry
    if query.category == "media":
        adapter_registry = get_adapter_registry()

    # System commands may need both
    if query.category == "system":
        if query.command in ("stop", "stop_tts"):
            adapter_registry = get_adapter_registry()
            audio_player = get_audio_player()
        elif query.command in ("cancel", "status"):
            scheduler = get_scheduler()
            adapter_registry = get_adapter_registry()
            audio_player = get_audio_player()

    # Ensure utility schema exists
    _ensure_utility_schema()

    # Get database connection
    from ..db_connection import get_connection

    # Dispatch the query
    with get_connection() as conn:
        utility_result = dispatch_utility(
            query,
            conn=conn,
            user_tz=user_tz,
            scheduler=scheduler,
            audio_player=audio_player,
            adapter_registry=adapter_registry,
            last_result=_last_result,
        )

    # Store for undo/repeat
    _last_result = utility_result

    return utility_result


def _execute_async_utility_query(query: UtilityQuery) -> Optional[UtilityResult]:
    """Execute a calendar/email UtilityQuery via async MCP dispatch."""
    import asyncio
    from .dispatcher import async_dispatch_utility

    global _last_result

    user_tz = config.get("timezone", "America/Chicago")
    _ensure_utility_schema()

    mcp_client = _get_mcp_client_manager()

    from ..db_connection import get_connection

    async def _run():
        try:
            with get_connection() as conn:
                return await async_dispatch_utility(
                    query,
                    conn=conn,
                    user_tz=user_tz,
                    mcp_client=mcp_client,
                    pipeline=_get_security_pipeline(),
                )
        finally:
            # Disconnect within same async context to avoid
            # cancel-scope errors during event loop shutdown
            await mcp_client.disconnect_all()

    # Run in event loop (create one if not running)
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — run in a thread to avoid nested loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run())
            result = future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        result = asyncio.run(_run())

    _last_result = result
    return result


