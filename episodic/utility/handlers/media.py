"""
Media Handler.

Handles media playback utility commands:
- media_play: Play radio, music, etc.
- media_pause: Pause current playback
- media_resume: Resume playback
- media_stop: Stop playback
- media_status: What's playing
- volume_up/down/mute/set: Volume controls
"""

import sqlite3
from typing import Optional

from ..types import UtilityQuery, UtilityResult
from ..adapters.base import AdapterRegistry, AdapterResult, AdapterStatus


def _adapter_result_to_utility_result(result: AdapterResult) -> UtilityResult:
    """Convert AdapterResult to UtilityResult."""
    if result.status == "ok":
        return UtilityResult.ok(
            display=result.display_text,
            speech=result.speech_text,
            **result.payload,
        )
    else:
        return UtilityResult.error(
            result.error_type or "adapter_error",
            result.error_message or "Unknown error",
        )


def handle_media_play(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle media_play command.

    Args in query:
        source: "radio", "spotify", etc. (optional, will infer)
        station: Station name for radio
        query: Search query for spotify
        type: Track/album/artist/playlist for spotify
    """
    source = query.args.get("source", "").lower()
    station = query.args.get("station")
    search_query = query.args.get("query")

    # Infer source if not specified
    if not source:
        if station:
            source = "radio"
        elif search_query:
            # Check if query matches a radio station before defaulting to spotify
            radio_adapter = adapter_registry.get_adapter("radio")
            if radio_adapter and hasattr(radio_adapter, '_stations'):
                if search_query.lower() in radio_adapter._stations:
                    source = "radio"
                    station = search_query
            if not source:
                source = "spotify"
        else:
            # Try to find any available media adapter
            for adapter in adapter_registry.list_adapters():
                if adapter.status() == AdapterStatus.READY:
                    source = adapter.name
                    break

    if not source:
        return UtilityResult.error(
            "no_source",
            "No media source specified and no adapters available"
        )

    # Get adapter
    adapter = adapter_registry.get_adapter(source)
    if adapter is None:
        available = [a.name for a in adapter_registry.list_adapters()]
        return UtilityResult.error(
            "adapter_not_found",
            f"Media source '{source}' not found. Available: {', '.join(available) or 'none'}"
        )

    # Check adapter status
    status = adapter.status()
    if status == AdapterStatus.NOT_CONFIGURED:
        return UtilityResult.error(
            "not_configured",
            f"{adapter.display_name} is not configured"
        )
    elif status == AdapterStatus.NOT_AUTHENTICATED:
        return UtilityResult.error(
            "not_authenticated",
            f"{adapter.display_name} requires authentication"
        )
    elif status == AdapterStatus.UNAVAILABLE:
        # Provide more helpful error messages for known adapters
        if source == "radio":
            return UtilityResult.error(
                "unavailable",
                "Radio requires VLC. Install with: brew install vlc && pip install python-vlc"
            )
        return UtilityResult.error(
            "unavailable",
            f"{adapter.display_name} is unavailable"
        )

    # Build args for adapter
    adapter_args = {}

    if source == "radio":
        # Radio play - station required
        if station:
            adapter_args["station"] = station
        elif search_query:
            adapter_args["station"] = search_query
        else:
            return UtilityResult.error(
                "missing_station",
                "Which station would you like to play?"
            )

    elif source == "spotify":
        # Spotify play - query required
        if search_query:
            adapter_args["query"] = search_query
            adapter_args["type"] = query.args.get("type", "track")
        else:
            return UtilityResult.error(
                "missing_query",
                "What would you like to play?"
            )

    # Execute
    result = adapter.execute("play", adapter_args)
    return _adapter_result_to_utility_result(result)


def handle_media_pause(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
) -> UtilityResult:
    """
    Handle media_pause command.

    Pauses the currently playing adapter.
    """
    # Find playing adapter
    playing = adapter_registry.get_playing_adapters()

    if not playing:
        return UtilityResult.ok(
            display="Nothing playing",
            speech="Nothing playing",
        )

    # Pause first playing adapter
    adapter = playing[0]

    if "pause" in adapter.commands:
        result = adapter.execute("pause", {})
        return _adapter_result_to_utility_result(result)
    elif "stop" in adapter.commands:
        # Fallback to stop if no pause
        result = adapter.execute("stop", {})
        return _adapter_result_to_utility_result(result)

    return UtilityResult.error(
        "no_pause",
        f"{adapter.display_name} does not support pause"
    )


def handle_media_resume(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
) -> UtilityResult:
    """
    Handle media_resume command.

    Resumes paused playback.
    """
    # Find adapter with resume capability
    for adapter in adapter_registry.list_adapters():
        if "resume" in adapter.commands:
            result = adapter.execute("resume", {})
            if result.status == "ok":
                return _adapter_result_to_utility_result(result)

    return UtilityResult.ok(
        display="Nothing to resume",
        speech="Nothing to resume",
    )


def handle_media_stop(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
) -> UtilityResult:
    """
    Handle media_stop command.

    Stops all media playback.
    """
    # Stop all playing adapters
    playing = adapter_registry.get_playing_adapters()

    if not playing:
        return UtilityResult.ok(
            display="Nothing playing",
            speech="Nothing playing",
        )

    stopped = []
    for adapter in playing:
        try:
            adapter.stop()
            stopped.append(adapter.display_name)
        except Exception:
            pass

    if stopped:
        return UtilityResult.ok(
            display=f"Stopped: {', '.join(stopped)}",
            speech="Stopped",
            _command="media_stop",
            stopped=stopped,
        )

    return UtilityResult.ok(
        display="Media stopped",
        speech="Stopped",
        _command="media_stop",
    )


def handle_media_status(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
) -> UtilityResult:
    """
    Handle media_status command.

    Shows what's currently playing.
    """
    playing = adapter_registry.get_playing_adapters()

    if not playing:
        return UtilityResult.ok(
            display="Nothing playing",
            speech="Nothing playing",
            playing=False,
        )

    # Get status from each playing adapter
    status_parts = []
    data = {"playing": True, "adapters": []}

    for adapter in playing:
        if "status" in adapter.commands:
            result = adapter.execute("status", {})
            status_parts.append(result.display_text)
            data["adapters"].append({
                "name": adapter.name,
                "display_name": adapter.display_name,
                "status": result.payload,
            })
        else:
            status_parts.append(f"{adapter.display_name}: playing")
            data["adapters"].append({
                "name": adapter.name,
                "display_name": adapter.display_name,
            })

    display = "\n".join(status_parts)

    if len(status_parts) == 1:
        speech = status_parts[0]
    else:
        speech = f"{len(playing)} sources playing"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        **data,
    )


def handle_volume_up(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
    audio_player=None,
) -> UtilityResult:
    """
    Handle volume_up command.

    Increases volume by delta (default 10).
    """
    delta = query.args.get("delta", 10)

    # Try to adjust volume on playing adapters
    for adapter in adapter_registry.get_playing_adapters():
        if hasattr(adapter, "volume_up"):
            volume = adapter.volume_up(delta)
            return UtilityResult.ok(
                display=f"Volume: {volume}%",
                speech=f"Volume {volume} percent",
                volume=volume,
            )

    # Fall back to system audio player
    if audio_player and hasattr(audio_player, "set_volume"):
        # Assume current volume and increase
        current = getattr(audio_player, "_volume", 0.8) if hasattr(audio_player, "_volume") else 0.8
        new_volume = min(1.0, current + delta / 100)
        audio_player.set_volume(new_volume)
        return UtilityResult.ok(
            display=f"Volume: {int(new_volume * 100)}%",
            speech=f"Volume {int(new_volume * 100)} percent",
            volume=int(new_volume * 100),
        )

    return UtilityResult.ok(
        display="Volume up",
        speech="Volume up",
    )


def handle_volume_down(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
    audio_player=None,
) -> UtilityResult:
    """
    Handle volume_down command.

    Decreases volume by delta (default 10).
    """
    delta = query.args.get("delta", 10)

    # Try to adjust volume on playing adapters
    for adapter in adapter_registry.get_playing_adapters():
        if hasattr(adapter, "volume_down"):
            volume = adapter.volume_down(delta)
            return UtilityResult.ok(
                display=f"Volume: {volume}%",
                speech=f"Volume {volume} percent",
                volume=volume,
            )

    # Fall back to system audio player
    if audio_player and hasattr(audio_player, "set_volume"):
        current = getattr(audio_player, "_volume", 0.8) if hasattr(audio_player, "_volume") else 0.8
        new_volume = max(0.0, current - delta / 100)
        audio_player.set_volume(new_volume)
        return UtilityResult.ok(
            display=f"Volume: {int(new_volume * 100)}%",
            speech=f"Volume {int(new_volume * 100)} percent",
            volume=int(new_volume * 100),
        )

    return UtilityResult.ok(
        display="Volume down",
        speech="Volume down",
    )


def handle_volume_mute(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
    audio_player=None,
) -> UtilityResult:
    """
    Handle volume_mute command.

    Mutes audio.
    """
    # Mute all playing adapters
    for adapter in adapter_registry.get_playing_adapters():
        if hasattr(adapter, "set_volume"):
            adapter.set_volume(0)

    # Mute system audio
    if audio_player and hasattr(audio_player, "set_volume"):
        audio_player.set_volume(0)

    return UtilityResult.ok(
        display="Muted",
        speech="Muted",
        muted=True,
    )


def handle_volume_set(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
    audio_player=None,
) -> UtilityResult:
    """
    Handle volume_set command.

    Sets volume to specific level.
    """
    level = query.args.get("level")

    if level is None:
        return UtilityResult.error(
            "missing_level",
            "What volume level?"
        )

    # Normalize to 0-100
    level = max(0, min(100, int(level)))

    # Set on playing adapters
    for adapter in adapter_registry.get_playing_adapters():
        if hasattr(adapter, "set_volume"):
            adapter.set_volume(level)

    # Set on system audio
    if audio_player and hasattr(audio_player, "set_volume"):
        audio_player.set_volume(level / 100)

    return UtilityResult.ok(
        display=f"Volume: {level}%",
        speech=f"Volume set to {level} percent",
        volume=level,
    )


def handle_radio_list(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
) -> UtilityResult:
    """
    Handle radio_list command.

    Lists available radio stations.
    """
    adapter = adapter_registry.get_adapter("radio")

    if adapter is None:
        return UtilityResult.error(
            "adapter_not_found",
            "Radio adapter not available"
        )

    genre = query.args.get("genre")
    result = adapter.execute("list", {"genre": genre} if genre else {})
    return _adapter_result_to_utility_result(result)


# Command routing for media category
MEDIA_HANDLERS = {
    "media_play": handle_media_play,
    "media_pause": handle_media_pause,
    "media_resume": handle_media_resume,
    "media_stop": handle_media_stop,
    "media_status": handle_media_status,
    "volume_up": handle_volume_up,
    "volume_down": handle_volume_down,
    "volume_mute": handle_volume_mute,
    "volume_set": handle_volume_set,
    "radio_list": handle_radio_list,
}


def dispatch_media_command(
    query: UtilityQuery,
    adapter_registry: AdapterRegistry,
    conn: Optional[sqlite3.Connection] = None,
    audio_player=None,
) -> UtilityResult:
    """Dispatch a media category command to the appropriate handler."""
    handler = MEDIA_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown media command: {query.command}"
        )

    # Route based on command requirements
    if query.command in ("media_play",):
        return handler(query, adapter_registry, conn)
    elif query.command in ("volume_up", "volume_down", "volume_mute", "volume_set"):
        return handler(query, adapter_registry, audio_player)
    else:
        return handler(query, adapter_registry)
