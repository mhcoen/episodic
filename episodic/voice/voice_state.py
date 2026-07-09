"""Voice mode state enum (leaf module)."""

from enum import Enum, auto


class VoiceState(Enum):
    """Voice mode states."""
    OFF = auto()        # Voice mode disabled
    LISTENING = auto()  # Waiting for speech input
    PROCESSING = auto() # STT in progress
    SPEAKING = auto()   # TTS playback in progress
    IDLE = auto()       # Waiting for wake word (uses local STT only)
