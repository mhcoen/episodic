"""
Voice mode for Episodic.

Provides speech-to-text (STT) and text-to-speech (TTS) capabilities
with support for both local and cloud providers.

Usage:
    from episodic.voice import get_voice_manager

    manager = get_voice_manager()
    manager.start()  # Enter voice mode

    text = manager.listen()  # Get speech input
    manager.speak("Hello!")  # Speak response

    manager.stop()  # Exit voice mode
"""

from episodic.voice.voice_mode import (
    VoiceModeManager,
    VoiceState,
    get_voice_manager,
)
from episodic.voice.stt_providers import (
    BaseSTTProvider,
    get_stt_provider,
)
from episodic.voice.tts_providers import (
    BaseTTSProvider,
    get_tts_provider,
)

__all__ = [
    "VoiceModeManager",
    "VoiceState",
    "get_voice_manager",
    "BaseSTTProvider",
    "get_stt_provider",
    "BaseTTSProvider",
    "get_tts_provider",
]
