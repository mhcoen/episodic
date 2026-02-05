"""
Radio Adapter.

Streams internet radio using VLC (python-vlc).
Requires VLC to be installed on the system.
"""

from typing import Dict, Any, List, Optional

from .base import (
    ServiceAdapter,
    AdapterStatus,
    AdapterResult,
    CommandSchema,
)


# Default station registry
DEFAULT_STATIONS = {
    "wbez": {
        "name": "WBEZ Chicago",
        "url": "https://stream.wbez.org/wbez128.mp3",
        "genre": "news",
    },
    "wxrt": {
        "name": "WXRT Chicago",
        "url": "https://live.amperwave.net/direct/audacy-wxrtfmaac-imc",
        "genre": "rock",
    },
    "wfmt": {
        "name": "WFMT Classical",
        "url": "https://wfmt.streamguys1.com/main-mp3",
        "genre": "classical",
    },
    "bbc4": {
        "name": "BBC Radio 4",
        "url": "http://stream.live.vc.bbcmedia.co.uk/bbc_radio_fourfm",
        "genre": "news",
    },
    "npr": {
        "name": "NPR News",
        "url": "https://npr-ice.streamguys1.com/live.mp3",
        "genre": "news",
    },
    "kexp": {
        "name": "KEXP Seattle",
        "url": "https://kexp-mp3-128.streamguys1.com/kexp128.mp3",
        "genre": "indie",
    },
    "classical": {
        "name": "Classical KUSC",
        "url": "https://kusc.streamguys1.com/kusc-128k.mp3",
        "genre": "classical",
    },
    "jazz": {
        "name": "WBGO Jazz",
        "url": "https://wbgo.streamguys1.com/wbgo128",
        "genre": "jazz",
    },
}


class RadioAdapter:
    """
    Radio streaming adapter using VLC.

    Provides play/stop/list commands for internet radio stations.
    Supports fuzzy matching by station key, name, or genre.
    """

    name = "radio"
    display_name = "Radio"
    commands = ["play", "stop", "list", "status"]

    def __init__(self):
        self._player = None
        self._current_station: Optional[Dict[str, Any]] = None
        self._stations = DEFAULT_STATIONS.copy()
        self._vlc_available: Optional[bool] = None
        self._volume = 80  # 0-100

    def describe(self) -> Dict[str, CommandSchema]:
        """Return schema for all commands."""
        return {
            "play": CommandSchema(
                name="play",
                description="Play a radio station",
                args={"station": "str"},
                required_args=["station"],
                mutating=True,
                requires_auth=False,
            ),
            "stop": CommandSchema(
                name="stop",
                description="Stop radio playback",
                args={},
                required_args=[],
                mutating=True,
                requires_auth=False,
            ),
            "list": CommandSchema(
                name="list",
                description="List available stations",
                args={"genre": "str"},
                required_args=[],
                mutating=False,
                requires_auth=False,
            ),
            "status": CommandSchema(
                name="status",
                description="Get current playback status",
                args={},
                required_args=[],
                mutating=False,
                requires_auth=False,
            ),
        }

    def status(self) -> AdapterStatus:
        """Check if VLC is available."""
        if self._vlc_available is None:
            self._vlc_available = self._check_vlc()

        if self._vlc_available:
            return AdapterStatus.READY
        return AdapterStatus.UNAVAILABLE

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration."""
        # Load custom stations
        if "stations" in config:
            self._stations.update(config["stations"])

        # Set volume
        if "volume" in config:
            self._volume = max(0, min(100, config["volume"]))

    def authenticate(self) -> bool:
        """No authentication needed."""
        return True

    def execute(self, command: str, args: Dict[str, Any]) -> AdapterResult:
        """Execute a radio command."""
        if command == "play":
            return self._play(args.get("station", ""))
        elif command == "stop":
            return self._stop()
        elif command == "list":
            return self._list(args.get("genre"))
        elif command == "status":
            return self._status()
        else:
            return AdapterResult.error(
                "unknown_command",
                f"Unknown radio command: {command}"
            )

    def is_playing(self) -> bool:
        """Check if currently playing."""
        if self._player is None:
            return False

        try:
            import vlc
            state = self._player.get_state()
            return state in (vlc.State.Playing, vlc.State.Buffering)
        except Exception:
            return False

    def stop(self) -> None:
        """Stop playback (for system stop)."""
        self._stop_player()

    # =========================================================================
    # Command Implementations
    # =========================================================================

    def _play(self, station_query: str) -> AdapterResult:
        """Play a radio station."""
        # Check VLC availability
        if not self._check_vlc():
            return AdapterResult.error(
                "vlc_not_available",
                "VLC is not installed. Please install VLC to use radio."
            )

        # Find station
        station = self._find_station(station_query)
        if station is None:
            # Try genre match
            genre_match = self._find_by_genre(station_query)
            if genre_match:
                station = genre_match
            else:
                available = ", ".join(self._stations.keys())
                return AdapterResult.error(
                    "station_not_found",
                    f"Station '{station_query}' not found. Available: {available}"
                )

        # Stop current playback
        self._stop_player()

        # Start new stream
        try:
            import vlc

            instance = vlc.Instance("--no-xlib", "--quiet")
            self._player = instance.media_player_new()

            media = instance.media_new(station["url"])
            self._player.set_media(media)
            self._player.audio_set_volume(self._volume)
            self._player.play()

            self._current_station = station

            return AdapterResult.ok(
                display=f"Playing {station['name']}",
                speech=f"Playing {station['name']}",
                station=station["name"],
                url=station["url"],
                genre=station.get("genre"),
                side_effects=["radio_started", station["name"]],
            )

        except Exception as e:
            return AdapterResult.error(
                "playback_error",
                f"Failed to play radio: {e}"
            )

    def _stop(self) -> AdapterResult:
        """Stop radio playback."""
        was_playing = self._current_station is not None
        station_name = self._current_station["name"] if self._current_station else None

        self._stop_player()

        if was_playing:
            return AdapterResult.ok(
                display=f"Stopped {station_name}",
                speech="Radio stopped",
                side_effects=["radio_stopped"],
            )
        else:
            return AdapterResult.ok(
                display="Radio not playing",
                speech="Radio not playing",
            )

    def _list(self, genre: Optional[str] = None) -> AdapterResult:
        """List available stations."""
        if genre:
            # Filter by genre
            filtered = {
                k: v for k, v in self._stations.items()
                if v.get("genre", "").lower() == genre.lower()
            }
            if not filtered:
                return AdapterResult.ok(
                    display=f"No stations found for genre: {genre}",
                    speech=f"No {genre} stations available",
                    stations=[],
                )
            stations_to_list = filtered
        else:
            stations_to_list = self._stations

        # Build display
        lines = []
        for key, station in stations_to_list.items():
            genre_str = f" [{station.get('genre', '')}]" if station.get("genre") else ""
            lines.append(f"  {key}: {station['name']}{genre_str}")

        display = "Radio Stations:\n" + "\n".join(lines)

        station_names = [s["name"] for s in stations_to_list.values()]
        if len(station_names) <= 3:
            speech = f"Available stations: {', '.join(station_names)}"
        else:
            speech = f"{len(station_names)} stations available"

        return AdapterResult.ok(
            display=display,
            speech=speech,
            stations=list(stations_to_list.keys()),
        )

    def _status(self) -> AdapterResult:
        """Get current playback status."""
        if not self.is_playing():
            return AdapterResult.ok(
                display="Radio not playing",
                speech="Radio not playing",
                playing=False,
            )

        station = self._current_station
        return AdapterResult.ok(
            display=f"Playing: {station['name']}",
            speech=f"Playing {station['name']}",
            playing=True,
            station=station["name"],
            genre=station.get("genre"),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _check_vlc(self) -> bool:
        """Check if VLC/python-vlc is available."""
        try:
            import vlc
            return True
        except ImportError:
            return False
        except Exception:
            # VLC installed but can't initialize
            return False

    def _find_station(self, query: str) -> Optional[Dict[str, Any]]:
        """Find station by key or name."""
        query = query.lower().strip()

        if not query:
            return None

        # Exact key match
        if query in self._stations:
            return self._stations[query]

        # Partial key match
        for key, station in self._stations.items():
            if query in key:
                return station

        # Name match
        for key, station in self._stations.items():
            if query in station["name"].lower():
                return station

        return None

    def _find_by_genre(self, genre: str) -> Optional[Dict[str, Any]]:
        """Find first station matching genre."""
        genre = genre.lower().strip()

        for station in self._stations.values():
            if station.get("genre", "").lower() == genre:
                return station

        # Partial genre match
        for station in self._stations.values():
            if genre in station.get("genre", "").lower():
                return station

        return None

    def _stop_player(self) -> None:
        """Stop the VLC player."""
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
            self._player = None
        self._current_station = None

    # =========================================================================
    # Volume Control
    # =========================================================================

    def set_volume(self, volume: int) -> None:
        """Set volume (0-100)."""
        self._volume = max(0, min(100, volume))

        if self._player:
            try:
                self._player.audio_set_volume(self._volume)
            except Exception:
                pass

    def get_volume(self) -> int:
        """Get current volume."""
        return self._volume

    def volume_up(self, delta: int = 10) -> int:
        """Increase volume."""
        self.set_volume(self._volume + delta)
        return self._volume

    def volume_down(self, delta: int = 10) -> int:
        """Decrease volume."""
        self.set_volume(self._volume - delta)
        return self._volume


class NullRadioAdapter:
    """
    No-op radio adapter for testing.

    Implements same interface but doesn't actually play audio.
    """

    name = "radio"
    display_name = "Radio (Test)"
    commands = ["play", "stop", "list", "status"]

    def __init__(self):
        self._current_station: Optional[Dict[str, Any]] = None
        self._stations = DEFAULT_STATIONS.copy()
        self._playing = False

    def describe(self) -> Dict[str, CommandSchema]:
        return RadioAdapter().__class__.describe(self)

    def status(self) -> AdapterStatus:
        return AdapterStatus.READY

    def configure(self, config: Dict[str, Any]) -> None:
        if "stations" in config:
            self._stations.update(config["stations"])

    def authenticate(self) -> bool:
        return True

    def execute(self, command: str, args: Dict[str, Any]) -> AdapterResult:
        if command == "play":
            station_query = args.get("station", "")
            station = self._find_station(station_query)
            if station:
                self._current_station = station
                self._playing = True
                return AdapterResult.ok(
                    display=f"Playing {station['name']}",
                    speech=f"Playing {station['name']}",
                    station=station["name"],
                )
            return AdapterResult.error("station_not_found", f"Station not found: {station_query}")

        elif command == "stop":
            self._playing = False
            self._current_station = None
            return AdapterResult.ok(display="Radio stopped", speech="Radio stopped")

        elif command == "list":
            return AdapterResult.ok(
                display="Stations: " + ", ".join(self._stations.keys()),
                speech=f"{len(self._stations)} stations available",
                stations=list(self._stations.keys()),
            )

        elif command == "status":
            if self._playing and self._current_station:
                return AdapterResult.ok(
                    display=f"Playing: {self._current_station['name']}",
                    speech=f"Playing {self._current_station['name']}",
                    playing=True,
                )
            return AdapterResult.ok(display="Not playing", speech="Not playing", playing=False)

        return AdapterResult.error("unknown_command", f"Unknown: {command}")

    def is_playing(self) -> bool:
        return self._playing

    def stop(self) -> None:
        self._playing = False
        self._current_station = None

    def _find_station(self, query: str) -> Optional[Dict[str, Any]]:
        query = query.lower().strip()
        if query in self._stations:
            return self._stations[query]
        for key, station in self._stations.items():
            if query in key or query in station["name"].lower():
                return station
        return None
