"""
Utility Command Dispatcher.

Routes UtilityQuery objects to the appropriate handlers based on category.
Implements safety gate (confidence checking) and event logging.
"""

import json
import time
import sqlite3
from typing import Optional, Callable

from .types import UtilityQuery, UtilityResult, ResultStatus, MUTATING_COMMANDS
from .handlers.time_date import dispatch_time_command
from .handlers.calculator import dispatch_calc_command
from .handlers.timer import dispatch_timer_command
from .handlers.alarm import dispatch_alarm_command
from .handlers.system import dispatch_system_command
from .handlers.notes import dispatch_note_command
from .handlers.reminders import dispatch_reminder_command
from .handlers.media import dispatch_media_command


# Handler registry by category
CATEGORY_DISPATCHERS = {
    "time": dispatch_time_command,
    "calc": dispatch_calc_command,
    "timer": dispatch_timer_command,
    "alarm": dispatch_alarm_command,
    "system": dispatch_system_command,
    "note": dispatch_note_command,
    "reminder": dispatch_reminder_command,
    "media": dispatch_media_command,
    # Future categories:
    # "weather": dispatch_weather_command,
    # "routine": dispatch_routine_command,
}


def should_execute(query: UtilityQuery, confirm_mutations: bool = False) -> tuple[bool, Optional[str]]:
    """
    Safety gate: determine if command should execute.

    Returns:
        (should_execute, reason_or_prompt)
        - If True: reason why it's allowed
        - If False: confirmation prompt or rejection reason
    """
    is_mutating = query.is_mutating()

    # High confidence: execute
    if query.confidence >= 0.9:
        return True, "high confidence"

    # Medium confidence, read-only: execute
    if query.confidence >= 0.7 and not is_mutating:
        return True, "read-only, acceptable confidence"

    # Medium confidence, mutating: need confirmation if setting enabled
    if query.confidence >= 0.7 and is_mutating:
        if confirm_mutations:
            prompt = f"Confirm: {query.command}"
            if query.args:
                args_str = ", ".join(f"{k}={v}" for k, v in query.args.items() if v is not None)
                prompt += f" ({args_str})"
            prompt += "?"
            return False, prompt
        return True, "mutations allowed without confirmation"

    # Low confidence: reject
    return False, "low confidence, falling back to LLM"


def log_utility_event(
    conn: Optional[sqlite3.Connection],
    query: UtilityQuery,
    result: UtilityResult,
    latency_us: int
) -> Optional[int]:
    """
    Log utility command execution to event log.

    Returns event_id if logged, None if no connection.
    """
    if conn is None:
        return None

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO utility_event_log
        (ts, source, category, command, args_json, result_status,
         result_payload_json, error_type, error_message, latency_us, side_effects_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(time.time()),
        query.source,
        query.category,
        query.command,
        json.dumps(query.args) if query.args else None,
        result.status.value,
        json.dumps(result.data) if result.data else None,
        result.error_type,
        result.error_message,
        latency_us,
        json.dumps(result.side_effects) if result.side_effects else None,
    ))
    conn.commit()
    return cursor.lastrowid


def dispatch_utility(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
    confirm_mutations: bool = False,
    scheduler=None,
    audio_player=None,
    tts_engine=None,
    media_adapters=None,
    last_result: Optional[UtilityResult] = None,
    adapter_registry=None,
) -> UtilityResult:
    """
    Dispatch a utility command to the appropriate handler.

    Args:
        query: UtilityQuery to execute
        conn: SQLite connection for persistence (optional)
        user_tz: User timezone (IANA format)
        confirm_mutations: Whether to require confirmation for mutating commands
        scheduler: Scheduler instance for timer/alarm commands (optional)
        audio_player: AudioPlayer instance for timer/alarm sounds (optional)
        tts_engine: TTS engine for speech output (optional)
        media_adapters: Media adapter dict for system controls (optional)
        last_result: Last result for repeat command (optional)
        adapter_registry: AdapterRegistry for media commands (optional)

    Returns:
        UtilityResult from handler
    """
    start_time = time.perf_counter_ns()

    # Safety gate
    should_exec, reason = should_execute(query, confirm_mutations)

    if not should_exec:
        if "confirm" in reason.lower():
            return UtilityResult.confirm(reason)
        return UtilityResult.fallback()

    # Get dispatcher for category
    dispatcher = CATEGORY_DISPATCHERS.get(query.category)

    if dispatcher is None:
        result = UtilityResult.error(
            "unknown_category",
            f"Unknown utility category: {query.category}"
        )
    else:
        # Execute handler
        try:
            # Route based on category with appropriate arguments
            if query.category == "time":
                result = dispatcher(query, user_tz)
            elif query.category == "calc":
                result = dispatcher(query)
            elif query.category == "timer":
                if scheduler is None:
                    result = UtilityResult.error(
                        "scheduler_required",
                        "Timer commands require a scheduler"
                    )
                else:
                    result = dispatcher(query, scheduler, conn, user_tz, audio_player)
            elif query.category == "alarm":
                if scheduler is None:
                    result = UtilityResult.error(
                        "scheduler_required",
                        "Alarm commands require a scheduler"
                    )
                else:
                    result = dispatcher(query, scheduler, conn, user_tz, audio_player)
            elif query.category == "system":
                # Create a recursive dispatcher for undo
                def recursive_dispatch(q):
                    return dispatch_utility(
                        q, conn, user_tz, confirm_mutations, scheduler,
                        audio_player, tts_engine, media_adapters, last_result,
                        adapter_registry
                    )
                result = dispatcher(
                    query, scheduler, conn, audio_player, tts_engine,
                    media_adapters, last_result, recursive_dispatch, user_tz
                )
            elif query.category == "note":
                result = dispatcher(query, conn, user_tz)
            elif query.category == "reminder":
                if scheduler is None:
                    result = UtilityResult.error(
                        "scheduler_required",
                        "Reminder commands require a scheduler"
                    )
                else:
                    result = dispatcher(query, scheduler, conn, user_tz, tts_engine)
            elif query.category == "media":
                if adapter_registry is None:
                    result = UtilityResult.error(
                        "adapter_registry_required",
                        "Media commands require an adapter registry"
                    )
                else:
                    result = dispatcher(query, adapter_registry, conn, audio_player)
            else:
                result = dispatcher(query)
        except Exception as e:
            result = UtilityResult.error(
                "handler_error",
                f"Handler failed: {e}"
            )

    # Calculate latency
    end_time = time.perf_counter_ns()
    latency_us = (end_time - start_time) // 1000

    # Log event
    if conn is not None:
        log_utility_event(conn, query, result, latency_us)

    return result


def create_utility_query(
    category: str,
    command: str,
    args: Optional[dict] = None,
    source: str = "cli",
    confidence: float = 1.0,
    raw_input: str = "",
) -> UtilityQuery:
    """
    Helper to create UtilityQuery objects.

    Useful for slash commands and programmatic access.
    """
    return UtilityQuery(
        category=category,
        command=command,
        args=args or {},
        confidence=confidence,
        source=source,
        raw_input=raw_input,
    )
