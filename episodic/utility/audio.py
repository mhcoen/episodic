"""
Audio Player for Utility Commands.

Handles non-TTS audio: alarm sounds, notification chimes, timer alerts.
Uses pygame if available, falls back to system audio commands.
"""

import subprocess
import platform
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol


class SoundType(Enum):
    """Types of sounds the player can produce."""
    ALARM = "alarm"
    TIMER = "timer"
    NOTIFICATION = "notification"
    REMINDER = "reminder"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class SoundConfig:
    """Configuration for audio player."""
    sound_dir: Path
    default_alarm: str = "alarm_default.wav"
    default_timer: str = "timer_default.wav"
    default_notification: str = "notification.wav"
    volume: float = 0.8


class AudioPlayer(Protocol):
    """Protocol for audio playback."""

    def configure(self, config: SoundConfig) -> None:
        """Apply configuration."""
        ...

    def play_sound(self, sound_type: SoundType, label: Optional[str] = None) -> None:
        """Play a sound by type."""
        ...

    def play_file(self, path: Path, loop: bool = False) -> None:
        """Play a specific audio file."""
        ...

    def play_alarm(self, label: Optional[str] = None) -> None:
        """Play alarm sound (loops until stopped)."""
        ...

    def play_timer(self, label: Optional[str] = None) -> None:
        """Play timer completion sound."""
        ...

    def play_notification(self) -> None:
        """Play notification chime."""
        ...

    def stop(self) -> None:
        """Stop all audio playback."""
        ...

    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 to 1.0)."""
        ...

    def is_playing(self) -> bool:
        """Check if currently playing."""
        ...


class AudioPlayerImpl:
    """
    Cross-platform audio player.

    Uses pygame mixer if available, falls back to system commands
    (afplay on macOS, aplay on Linux, winsound on Windows).
    """

    def __init__(self):
        self._config: Optional[SoundConfig] = None
        self._playing = False
        self._loop = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mixer_initialized = False
        self._pygame_available = False
        self._current_sound_type: Optional[SoundType] = None
        self._current_label: Optional[str] = None

    def configure(self, config: SoundConfig) -> None:
        """Apply configuration."""
        self._config = config
        self._init_mixer()

    def _init_mixer(self) -> None:
        """Initialize pygame mixer if available."""
        if self._mixer_initialized:
            return

        try:
            import pygame
            pygame.mixer.init()
            if self._config:
                pygame.mixer.music.set_volume(self._config.volume)
            self._mixer_initialized = True
            self._pygame_available = True
        except ImportError:
            self._pygame_available = False
        except Exception:
            # pygame installed but mixer init failed (no audio device, etc.)
            self._pygame_available = False

    def play_sound(self, sound_type: SoundType, label: Optional[str] = None) -> None:
        """Play sound by type."""
        if self._config is None:
            return

        sound_map = {
            SoundType.ALARM: self._config.default_alarm,
            SoundType.TIMER: self._config.default_timer,
            SoundType.NOTIFICATION: self._config.default_notification,
            SoundType.REMINDER: self._config.default_notification,
            SoundType.ERROR: "error.wav",
            SoundType.SUCCESS: "success.wav",
        }

        filename = sound_map.get(sound_type, "notification.wav")

        # Check for label-specific sound
        if label:
            label_file = self._config.sound_dir / f"{label.lower().replace(' ', '_')}.wav"
            if label_file.exists():
                filename = label_file.name

        # Track what's playing
        self._current_sound_type = sound_type
        self._current_label = label

        path = self._config.sound_dir / filename
        loop = sound_type in (SoundType.ALARM, SoundType.TIMER)
        self.play_file(path, loop=loop)

    def play_file(self, path: Path, loop: bool = False) -> None:
        """Play audio file."""
        if not path.exists():
            return

        self.stop()  # Stop any current playback

        self._loop = loop
        self._stop_event.clear()
        self._playing = True

        self._thread = threading.Thread(
            target=self._play_thread,
            args=(path,),
            daemon=True,
        )
        self._thread.start()

    def _play_thread(self, path: Path) -> None:
        """Background thread for audio playback."""
        if self._pygame_available:
            self._play_pygame(path)
        else:
            self._play_system(path)

        self._playing = False

    def _play_pygame(self, path: Path) -> None:
        """Play using pygame mixer."""
        try:
            import pygame

            pygame.mixer.music.load(str(path))

            if self._loop:
                pygame.mixer.music.play(loops=-1)
            else:
                pygame.mixer.music.play()

            # Wait for completion or stop
            while pygame.mixer.music.get_busy() and not self._stop_event.is_set():
                pygame.time.Clock().tick(10)

        except Exception:
            # Fallback to system audio on error
            self._play_system(path)

    def _play_system(self, path: Path) -> None:
        """Play using system audio commands."""
        system = platform.system()

        try:
            if system == "Darwin":  # macOS
                # afplay supports looping with -1 flag, but we'll handle looping ourselves
                while True:
                    if self._stop_event.is_set():
                        break
                    proc = subprocess.Popen(
                        ["afplay", str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    # Wait for completion or stop
                    while proc.poll() is None:
                        if self._stop_event.is_set():
                            proc.terminate()
                            break
                        self._stop_event.wait(timeout=0.1)

                    if not self._loop:
                        break

            elif system == "Windows":
                import winsound
                flags = winsound.SND_FILENAME
                if self._loop:
                    flags |= winsound.SND_LOOP | winsound.SND_ASYNC

                while True:
                    if self._stop_event.is_set():
                        break
                    winsound.PlaySound(str(path), flags)
                    if not self._loop:
                        break

            else:  # Linux
                while True:
                    if self._stop_event.is_set():
                        break
                    proc = subprocess.Popen(
                        ["aplay", str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    while proc.poll() is None:
                        if self._stop_event.is_set():
                            proc.terminate()
                            break
                        self._stop_event.wait(timeout=0.1)

                    if not self._loop:
                        break

        except Exception:
            pass  # Silently fail if system audio unavailable

    def play_alarm(self, label: Optional[str] = None) -> None:
        """Play alarm sound (loops until stopped)."""
        self.play_sound(SoundType.ALARM, label)

    def play_timer(self, label: Optional[str] = None) -> None:
        """Play timer completion sound."""
        self.play_sound(SoundType.TIMER, label)

    def play_notification(self) -> None:
        """Play notification chime."""
        self.play_sound(SoundType.NOTIFICATION)

    def stop(self) -> None:
        """Stop all playback."""
        self._stop_event.set()
        self._loop = False

        if self._pygame_available:
            try:
                import pygame
                if self._mixer_initialized:
                    pygame.mixer.music.stop()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._playing = False
        self._current_sound_type = None
        self._current_label = None

    def is_alarm_or_timer_sounding(self) -> bool:
        """Check if an alarm or timer sound is currently playing."""
        if not self._playing:
            return False
        return self._current_sound_type in (SoundType.ALARM, SoundType.TIMER)

    def get_current_sound_info(self) -> Optional[tuple]:
        """Get info about current sound (type, label) or None if not playing."""
        if not self._playing:
            return None
        return (self._current_sound_type, self._current_label)

    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)."""
        volume = max(0.0, min(1.0, volume))

        if self._pygame_available:
            try:
                import pygame
                if self._mixer_initialized:
                    pygame.mixer.music.set_volume(volume)
            except Exception:
                pass

        if self._config:
            self._config.volume = volume

    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._playing


class NullAudioPlayer:
    """
    No-op audio player for testing or headless mode.

    Implements the same interface but produces no sound.
    """

    def __init__(self):
        self._config: Optional[SoundConfig] = None
        self._playing = False

    def configure(self, config: SoundConfig) -> None:
        self._config = config

    def play_sound(self, sound_type: SoundType, label: Optional[str] = None) -> None:
        pass

    def play_file(self, path: Path, loop: bool = False) -> None:
        pass

    def play_alarm(self, label: Optional[str] = None) -> None:
        pass

    def play_timer(self, label: Optional[str] = None) -> None:
        pass

    def play_notification(self) -> None:
        pass

    def stop(self) -> None:
        self._playing = False

    def set_volume(self, volume: float) -> None:
        pass

    def is_playing(self) -> bool:
        return self._playing

    def is_alarm_or_timer_sounding(self) -> bool:
        return False

    def get_current_sound_info(self) -> Optional[tuple]:
        return None


def get_default_sound_dir() -> Path:
    """Get the default sound directory."""
    # Check for user sounds first (must contain actual WAV files)
    user_sounds = Path.home() / ".episodic" / "sounds"
    if user_sounds.exists() and any(user_sounds.glob("*.wav")):
        return user_sounds

    # Fall back to bundled sounds
    bundled = Path(__file__).parent.parent / "sounds"
    if bundled.exists():
        return bundled

    # Create user sounds dir
    user_sounds.mkdir(parents=True, exist_ok=True)
    return user_sounds


def create_audio_player(headless: bool = False) -> AudioPlayerImpl:
    """
    Create and configure an audio player.

    Args:
        headless: If True, returns a NullAudioPlayer for testing.
    """
    if headless:
        return NullAudioPlayer()

    player = AudioPlayerImpl()

    sound_dir = get_default_sound_dir()
    config = SoundConfig(sound_dir=sound_dir)
    player.configure(config)

    return player
