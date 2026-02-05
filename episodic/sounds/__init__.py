"""Bundled sound files for alarms, timers, and notifications."""

from pathlib import Path

SOUNDS_DIR = Path(__file__).parent


def get_sound_path(name: str) -> Path:
    """Get the path to a bundled sound file."""
    return SOUNDS_DIR / name
