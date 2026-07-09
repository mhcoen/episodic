"""Utility service lifecycle: singletons, factories, and shutdown.

Split out of cli_integration.py. Owns the scheduler / audio / adapter /
data-refresh / MCP-client singletons and their teardown. Re-imported into
cli_integration so its execution functions and external callers resolve
unchanged.
"""

from typing import Optional

from ..config import config
from .types import UtilityResult
from .scheduler import Scheduler
from .adapters.base import AdapterRegistry
from .adapters.radio import RadioAdapter, NullRadioAdapter
from .audio import AudioPlayerImpl, create_audio_player
from .data_refresh import DataRefreshScheduler, get_data_refresh_scheduler


_scheduler: Optional[Scheduler] = None
_adapter_registry: Optional[AdapterRegistry] = None
_audio_player: Optional[AudioPlayerImpl] = None
_data_refresh_scheduler: Optional[DataRefreshScheduler] = None
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
        # Must be set before start(): start() loads persisted tasks, and
        # restored tasks need their callbacks rebuilt to ring/update DB rows
        _scheduler._callback_factory = _restore_task_callback
        _scheduler.start()
    return _scheduler


def _restore_task_callback(task):
    """Rebuild the callback for a task restored from persistence.

    Closures can't be persisted, so tasks loaded on startup arrive without
    callbacks. Rebuild the same behavior the original handler closures
    provide: play the sound and update the owning DB row. Display and TTS
    are handled by _handle_task_fire via the scheduler's _on_task_fire hook.
    """
    from .scheduler import TaskType, TaskStatus, TaskResult

    if task.task_type == TaskType.ALARM:
        def alarm_callback() -> TaskResult:
            get_audio_player().play_alarm(task.label)
            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=task.label or "Alarm",
                side_effects=["alarm_fired", task.reference_id],
            )
        return alarm_callback

    if task.task_type == TaskType.TIMER:
        def timer_callback() -> TaskResult:
            get_audio_player().play_timer(task.label)
            if task.reference_id:
                try:
                    from ..db_connection import get_connection
                    from .handlers.timer import _update_timer_status
                    with get_connection() as fresh_conn:
                        _update_timer_status(fresh_conn, task.reference_id, "expired")
                except Exception:
                    pass
            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=f"{task.label} timer done" if task.label else "Timer done",
                side_effects=["timer_expired", task.reference_id],
            )
        return timer_callback

    if task.task_type == TaskType.REMINDER:
        def reminder_callback() -> TaskResult:
            if task.reference_id:
                try:
                    from ..db_connection import get_connection
                    from .handlers.reminders import _disable_reminder
                    with get_connection() as fresh_conn:
                        _disable_reminder(fresh_conn, task.reference_id)
                except Exception:
                    pass
            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=f"Reminder: {task.label}" if task.label else "Reminder",
                side_effects=["reminder_fired", task.reference_id],
            )
        return reminder_callback

    return None


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

def shutdown_utility_services() -> None:
    """Shutdown utility services (scheduler, adapters, audio, data refresh)."""
    global _scheduler, _adapter_registry, _audio_player, _data_refresh_scheduler

    if _data_refresh_scheduler is not None:
        _data_refresh_scheduler.stop()
        _data_refresh_scheduler = None

    if _scheduler is not None:
        # Stop (join) the scheduler thread BEFORE closing its dedicated
        # connection — a task firing after the close would hit a closed
        # database and kill the thread mid-flight.
        _scheduler.stop()
        if _scheduler._conn is not None:
            try:
                _scheduler._conn.close()
            except Exception:
                pass
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
