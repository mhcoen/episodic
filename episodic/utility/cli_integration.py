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


# Global instances (initialized on first use)
_scheduler: Optional[Scheduler] = None
_adapter_registry: Optional[AdapterRegistry] = None
_audio_player: Optional[AudioPlayerImpl] = None
_data_refresh_scheduler: Optional[DataRefreshScheduler] = None
_last_result: Optional[UtilityResult] = None
_mcp_client_manager = None
_security_pipeline = None
_schema_initialized: bool = False


def _get_mcp_client_manager():
    """Get or create the shared MCPClientManager singleton."""
    global _mcp_client_manager
    if _mcp_client_manager is None:
        from episodic.mcp.client_manager import MCPClientManager
        _mcp_client_manager = MCPClientManager()
    return _mcp_client_manager


def _get_security_pipeline():
    """Get or create a shared MCP SecurityPipeline for async utility dispatch."""
    global _security_pipeline
    if _security_pipeline is None:
        from episodic.mcp.security.audit import AuditLogger
        from episodic.mcp.security.pipeline import SecurityPipeline
        _security_pipeline = SecurityPipeline(audit_logger=AuditLogger())
    return _security_pipeline


def _ensure_utility_schema() -> None:
    """Ensure utility database schema exists."""
    global _schema_initialized
    if _schema_initialized:
        return

    from ..db_connection import get_connection
    from .db import init_utility_schema

    with get_connection() as conn:
        init_utility_schema(conn)

    _schema_initialized = True


def get_scheduler() -> Scheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        import sqlite3
        from ..db_connection import get_db_path

        # Ensure schema exists first
        _ensure_utility_schema()

        user_tz = config.get("timezone", "America/Chicago")

        # Create a dedicated connection for the scheduler (not from pool)
        # The scheduler needs a persistent connection for its background thread
        db_path = get_db_path()
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        _scheduler = Scheduler(conn=conn, user_tz=user_tz)
        _scheduler._on_task_fire = _handle_task_fire
        _scheduler.start()
    return _scheduler


def _handle_task_fire(task, result) -> None:
    """Handle a timer/alarm/reminder firing — display output and play sound."""
    from ..color_utils import secho_color
    from ..configuration import get_system_color
    from .speech import SpeechGenerator

    generator = SpeechGenerator.get_instance()

    # Determine command name for speech templates
    task_type_str = task.task_type.name.lower()  # "timer", "alarm", "reminder"

    if task_type_str == "timer":
        command = "timer_fired"
    elif task_type_str == "alarm":
        command = "alarm_fired"
    elif task_type_str == "reminder":
        command = "reminder_fired"
    else:
        command = f"{task_type_str}_fired"

    values = {"_command": command, "label": task.label or ""}
    if task_type_str == "reminder" and task.label:
        values["text"] = task.label

    display_text, speech_text = generator.generate(command, values)

    # Print to terminal directly (not through word tokenizer)
    secho_color(display_text, fg=get_system_color())

    # TTS if voice mode enabled
    if config.get("voice_mode") and config.get("voice_tts_enabled", True):
        try:
            from ..voice import get_voice_manager
            voice_manager = get_voice_manager()
            if voice_manager.is_active:
                voice_manager.speak(speech_text)
        except Exception:
            pass


def get_audio_player() -> AudioPlayerImpl:
    """Get or create the global audio player."""
    global _audio_player
    if _audio_player is None:
        _audio_player = create_audio_player()
    return _audio_player


def get_adapter_registry() -> AdapterRegistry:
    """Get or create the global adapter registry."""
    global _adapter_registry
    if _adapter_registry is None:
        _adapter_registry = AdapterRegistry()
        # Register radio adapter (uses NullRadioAdapter if VLC not available)
        try:
            radio = RadioAdapter()
            _adapter_registry.register(radio)
        except Exception:
            # Fall back to null adapter for testing
            _adapter_registry.register(NullRadioAdapter())
    return _adapter_registry


def start_data_refresh_scheduler() -> DataRefreshScheduler:
    """
    Start the data refresh scheduler for background provider updates.

    Registers news and weather providers for pre-fetching.
    """
    global _data_refresh_scheduler

    if _data_refresh_scheduler is not None and _data_refresh_scheduler.is_running():
        return _data_refresh_scheduler

    _data_refresh_scheduler = get_data_refresh_scheduler()

    # Register news provider for background refresh (25 min interval)
    from .handlers.news import get_news_provider
    news_provider = get_news_provider()
    _data_refresh_scheduler.register(
        "news_general",
        news_provider,
        refresh_interval_s=1500,  # 25 minutes
        args={"category": "general"},
    )

    # Register weather provider for background refresh (10 min interval)
    from .handlers.weather import get_weather_provider, _configure_provider
    weather_provider = get_weather_provider()

    # Configure provider so it has API key and location preferences
    _ensure_utility_schema()
    from ..db_connection import get_connection
    with get_connection() as conn:
        _configure_provider(weather_provider, conn)

    _data_refresh_scheduler.register(
        "weather_default",
        weather_provider,
        refresh_interval_s=600,  # 10 minutes
        args={},
    )

    _data_refresh_scheduler.start()
    return _data_refresh_scheduler


def _parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse duration string to seconds.

    Supports: 10s, 5m, 1h, 1h30m, 90m
    """
    duration_str = duration_str.lower().strip()

    # Handle combined formats like "1h30m"
    total_seconds = 0

    # Pattern: extract all (number, unit) pairs
    pattern = r'(\d+)\s*(h|hr|hour|hours|m|min|mins|minutes|s|sec|secs|seconds?)?'
    matches = re.findall(pattern, duration_str)

    if not matches:
        return None

    for value_str, unit in matches:
        value = int(value_str)

        if not unit or unit.startswith('s'):
            total_seconds += value
        elif unit.startswith('m'):
            total_seconds += value * 60
        elif unit.startswith('h'):
            total_seconds += value * 3600

    return total_seconds if total_seconds > 0 else None


_DURATION_UNITS = {
    's', 'sec', 'secs', 'second', 'seconds',
    'm', 'min', 'mins', 'minute', 'minutes',
    'h', 'hr', 'hour', 'hours',
}


def _parse_timer_args(args: list) -> tuple:
    """
    Split timer args into (duration_s, label).

    Greedily consumes tokens that are numbers or duration unit words,
    then treats the rest as label.

    "30 seconds"       → (30, None)
    "30 seconds pasta" → (30, "pasta")
    "5m eggs"          → (300, "eggs")
    "1h30m"            → (5400, None)
    "30"               → (30, None)
    """
    i = 0
    while i < len(args):
        token = args[i].lower()
        if re.match(r'^\d+', token):
            i += 1
            continue
        if token in _DURATION_UNITS and i > 0:
            i += 1
            continue
        break

    duration_str = " ".join(args[:i])
    duration_s = _parse_duration(duration_str) if duration_str else None
    label = " ".join(args[i:]) if i < len(args) else None
    return (duration_s, label)


def _parse_time(time_str: str, user_tz: str = "America/Chicago") -> Optional[datetime]:
    """
    Parse time string to datetime.

    Supports: 7am, 7:00am, 19:00, 7:00, 7:30pm
    """
    time_str = time_str.lower().strip()
    tz = ZoneInfo(user_tz)
    now = datetime.now(tz)

    # Pattern for various time formats
    patterns = [
        # 7am, 7pm
        (r'^(\d{1,2})\s*(am|pm)$', lambda m: (int(m.group(1)), 0, m.group(2))),
        # 7:00am, 7:30pm
        (r'^(\d{1,2}):(\d{2})\s*(am|pm)$', lambda m: (int(m.group(1)), int(m.group(2)), m.group(3))),
        # 19:00, 7:00 (24-hour format)
        (r'^(\d{1,2}):(\d{2})$', lambda m: (int(m.group(1)), int(m.group(2)), None)),
    ]

    for pattern, extractor in patterns:
        match = re.match(pattern, time_str)
        if match:
            hour, minute, meridiem = extractor(match)

            # Convert to 24-hour format
            if meridiem:
                if meridiem == 'pm' and hour != 12:
                    hour += 12
                elif meridiem == 'am' and hour == 12:
                    hour = 0

            # Validate hour/minute ranges
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                return None

            # Create datetime for today
            alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # If time has passed today, schedule for tomorrow
            if alarm_time <= now:
                alarm_time += timedelta(days=1)

            return alarm_time

    return None


def _parse_remind_args(text: str, user_tz: str = "America/Chicago") -> Tuple[Optional[str], Optional[int], Optional[datetime]]:
    """
    Parse reminder text and time.

    Supports:
    - "call mom in 2h" → ("call mom", 7200, None)
    - "meeting at 3pm" → ("meeting", None, datetime)

    Returns (reminder_text, duration_seconds, alarm_datetime)
    """
    text = text.strip()

    # Try "X in Y" pattern
    in_match = re.match(r'^(.+?)\s+in\s+(.+)$', text, re.IGNORECASE)
    if in_match:
        reminder_text = in_match.group(1).strip()
        duration_str = in_match.group(2).strip()
        duration_s = _parse_duration(duration_str)
        if duration_s:
            return (reminder_text, duration_s, None)

    # Try "X at Y" pattern
    at_match = re.match(r'^(.+?)\s+at\s+(.+)$', text, re.IGNORECASE)
    if at_match:
        reminder_text = at_match.group(1).strip()
        time_str = at_match.group(2).strip()
        alarm_time = _parse_time(time_str, user_tz)
        if alarm_time:
            return (reminder_text, None, alarm_time)

    return (None, None, None)


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
            # First arg is time, rest is label
            # Pass the raw time string - handler will parse it
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
            return UtilityResult.error("missing_query", "Usage: /play <station>")
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


def shutdown_utility_services() -> None:
    """Shutdown utility services (scheduler, adapters, audio, data refresh)."""
    global _scheduler, _adapter_registry, _audio_player, _data_refresh_scheduler

    if _data_refresh_scheduler is not None:
        _data_refresh_scheduler.stop()
        _data_refresh_scheduler = None

    if _scheduler is not None:
        # Close the scheduler's dedicated connection
        if _scheduler._conn is not None:
            try:
                _scheduler._conn.close()
            except Exception:
                pass
        _scheduler.stop()
        _scheduler = None

    if _audio_player is not None:
        try:
            _audio_player.stop()
        except Exception:
            pass
        _audio_player = None

    if _adapter_registry is not None:
        # Stop any playing adapters
        for adapter in _adapter_registry.list_adapters():
            try:
                adapter.stop()
            except Exception:
                pass
        _adapter_registry = None
