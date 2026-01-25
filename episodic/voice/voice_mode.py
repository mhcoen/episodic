"""
Voice mode manager for Episodic.

Coordinates STT, TTS, audio capture, and playback into a unified voice interface.
"""

import threading
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np
import typer

from episodic.config import config


class VoiceState(Enum):
    """Voice mode states."""
    OFF = auto()        # Voice mode disabled
    LISTENING = auto()  # Waiting for speech input
    PROCESSING = auto() # STT in progress
    SPEAKING = auto()   # TTS playback in progress
    IDLE = auto()       # Waiting for wake word (uses local STT only)


# Singleton instance
_voice_manager: Optional["VoiceModeManager"] = None


class VoiceModeManager:
    """
    Manages voice mode for Episodic.

    Coordinates:
    - Audio capture with VAD
    - Speech-to-text conversion
    - Text-to-speech output
    - State transitions and callbacks
    """

    def __init__(self):
        self._state = VoiceState.OFF
        self._lock = threading.Lock()

        # Components (lazy loaded)
        self._audio_capture = None
        self._audio_playback = None
        self._stt_provider = None
        self._tts_provider = None

        # Wake word detection
        self._wake_word_detector = None
        self._idle_timer: Optional[threading.Timer] = None

        # Callbacks
        self._on_transcription: Optional[Callable[[str], None]] = None
        self._on_state_change: Optional[Callable[[VoiceState], None]] = None

        # Result storage for blocking listen()
        self._transcription_result: Optional[str] = None
        self._transcription_event = threading.Event()

    def _get_stt_provider(self):
        """Get or create STT provider based on config."""
        if self._stt_provider is None:
            from episodic.voice.stt_providers import get_stt_provider

            provider_name = config.get("voice_stt_provider", "openai_whisper")
            kwargs = {}

            if provider_name == "local_whisper":
                kwargs["model_size"] = config.get("voice_local_whisper_model", "base")

            self._stt_provider = get_stt_provider(provider_name, **kwargs)

        return self._stt_provider

    def _get_tts_provider(self):
        """Get or create TTS provider based on config."""
        if self._tts_provider is None:
            from episodic.voice.tts_providers import get_tts_provider

            provider_name = config.get("voice_tts_provider", "openai_tts")
            kwargs = {}

            # Speed setting applies to all providers
            kwargs["speed"] = config.get("voice_tts_speed", 1.0)

            if provider_name == "local_piper":
                kwargs["voice"] = config.get("voice_local_piper_voice", "en_US-lessac-medium")
            elif provider_name == "local_xtts":
                kwargs["speaker"] = config.get("voice_local_xtts_speaker", "Claribel Dervla")
            elif provider_name == "openai_tts":
                kwargs["voice"] = config.get("voice_openai_tts_voice", "alloy")

            self._tts_provider = get_tts_provider(provider_name, **kwargs)

        return self._tts_provider

    def _get_audio_capture(self):
        """Get or create audio capture."""
        if self._audio_capture is None:
            from episodic.voice.audio_capture import AudioCapture

            self._audio_capture = AudioCapture(
                target_sample_rate=16000,
                vad_aggressiveness=config.get("voice_vad_aggressiveness", 2),
                silence_threshold_ms=config.get("voice_silence_threshold_ms", 1000),
            )

        return self._audio_capture

    def _get_audio_playback(self):
        """Get or create audio playback."""
        if self._audio_playback is None:
            from episodic.voice.audio_playback import AudioPlayback

            self._audio_playback = AudioPlayback(
                on_start=self._on_playback_start,
                on_stop=self._on_playback_stop,
            )

        return self._audio_playback

    def _get_wake_word_detector(self):
        """Get or create Porcupine wake word detector."""
        if self._wake_word_detector is None:
            from episodic.voice.wake_word import PorcupineWakeWordDetector

            keyword = config.get("voice_wake_word", "computer").lower()
            sensitivity = config.get("voice_wake_word_sensitivity", 0.5)

            self._wake_word_detector = PorcupineWakeWordDetector(
                keyword=keyword,
                sensitivity=sensitivity,
                on_wake_word=self._activate_from_idle,
            )

        return self._wake_word_detector

    def _start_idle_timer(self):
        """Start timer to transition to IDLE state after inactivity."""
        self._cancel_idle_timer()

        timeout = config.get("voice_idle_timeout", 60)
        if timeout > 0 and config.get("voice_wake_word_enabled", True):
            self._idle_timer = threading.Timer(timeout, self._enter_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _cancel_idle_timer(self):
        """Cancel pending idle timer."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    def pause_idle_timer(self):
        """Pause the idle timer (call when processing LLM request)."""
        self._cancel_idle_timer()

    def resume_idle_timer(self):
        """Resume the idle timer (call after LLM response completes)."""
        if self._state not in (VoiceState.OFF, VoiceState.IDLE):
            self._start_idle_timer()

    def _enter_idle(self):
        """Transition to IDLE state (wake word listening only)."""
        # Only transition from LISTENING state
        # If in PROCESSING or SPEAKING, restart timer to try again later
        if self._state == VoiceState.LISTENING:
            wake_word = config.get("voice_wake_word", "computer")
            if config.get("voice_show_transcription", True):
                # Flush output since we're in a timer thread
                import sys
                typer.secho(f"💤 Idle - say \"{wake_word}\" to wake", fg="yellow")
                sys.stdout.flush()
            self._set_state(VoiceState.IDLE)
        elif self._state in (VoiceState.PROCESSING, VoiceState.SPEAKING):
            # Busy right now, try again in a few seconds
            self._start_idle_timer()

    def _check_wake_word(self, audio: np.ndarray):
        """Check if audio contains the wake word using Porcupine."""
        try:
            detector = self._get_wake_word_detector()
            # Porcupine processes and calls _activate_from_idle via callback if detected
            detector.process_audio(audio, sample_rate=16000)
        except Exception as e:
            typer.secho(f"Wake word detection error: {e}", fg="red", err=True)

    def _activate_from_idle(self):
        """Wake up from IDLE state."""
        import sys

        # Play ready chime
        if config.get("voice_audio_cues", True):
            from episodic.voice.audio_playback import play_chime
            play_chime()

        if config.get("voice_show_transcription", True):
            typer.secho("🎤 Listening...", fg="green")
            sys.stdout.flush()

        self._set_state(VoiceState.LISTENING)
        self._start_idle_timer()

    def force_idle(self):
        """Force transition to IDLE state (for voice commands like 'go to sleep')."""
        import sys

        if self._state == VoiceState.OFF:
            return

        self._cancel_idle_timer()

        wake_word = config.get("voice_wake_word", "computer")
        if config.get("voice_show_transcription", True):
            typer.secho(f"😴 Going to sleep - say \"{wake_word}\" to wake", fg="yellow")
            sys.stdout.flush()

        self._set_state(VoiceState.IDLE)

    def is_sleep_command(self, text: str) -> bool:
        """Check if text is a command to go to sleep/idle mode."""
        sleep_phrases = [
            "go to sleep",
            "stop listening",
            "go idle",
            "sleep mode",
            "standby",
            "go to standby",
        ]
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in sleep_phrases)

    def _set_state(self, state: VoiceState):
        """Update state and trigger callback."""
        with self._lock:
            old_state = self._state
            self._state = state

        if self._on_state_change and old_state != state:
            self._on_state_change(state)

    def _on_speech_start(self):
        """Called when speech is detected."""
        # Don't cancel idle timer here - only reset it after successful transcription
        # This prevents background noise from resetting the idle timeout

        # Don't change state if in IDLE mode (let _on_speech_end handle wake word)
        if self._state != VoiceState.IDLE:
            self._set_state(VoiceState.PROCESSING)

    def _on_speech_end(self, audio: np.ndarray):
        """Called when speech ends, process with STT."""
        # Handle IDLE state - check for wake word only
        if self._state == VoiceState.IDLE:
            self._check_wake_word(audio)
            return

        try:
            stt = self._get_stt_provider()
            text = stt.transcribe(audio, 16000)

            if text:
                self._transcription_result = text
                if self._on_transcription:
                    self._on_transcription(text)

                # Only reset idle timer after successful transcription with actual text
                # This prevents background noise from resetting the timeout
                self._cancel_idle_timer()
                self._start_idle_timer()

            self._transcription_event.set()

            # Return to listening
            self._set_state(VoiceState.LISTENING)

        except Exception as e:
            typer.secho(f"STT error: {e}", fg="red", err=True)
            self._transcription_event.set()
            self._set_state(VoiceState.LISTENING)

    def _on_playback_start(self):
        """Called when TTS playback starts."""
        # Mute microphone during playback to prevent feedback
        if self._audio_capture:
            self._audio_capture.mute()

    def _on_playback_stop(self):
        """Called when TTS playback queue is empty."""
        # Unmute microphone
        if self._audio_capture:
            self._audio_capture.unmute()

        # Restart idle timer after speaking completes
        self._start_idle_timer()

        # Note: Don't play beep here - this fires between sentences during streaming.
        # The ready chime is played only at voice mode start.

    def start(
        self,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[VoiceState], None]] = None,
    ):
        """
        Start voice mode.

        Args:
            on_transcription: Called with transcribed text
            on_state_change: Called on state transitions
        """
        if self._state != VoiceState.OFF:
            return

        self._on_transcription = on_transcription
        self._on_state_change = on_state_change

        # Initialize components
        capture = self._get_audio_capture()
        playback = self._get_audio_playback()

        # Pre-load providers if possible
        try:
            self._get_stt_provider()
        except Exception as e:
            typer.secho(f"Warning: STT provider init failed: {e}", fg="yellow", err=True)

        try:
            self._get_tts_provider()
        except Exception as e:
            typer.secho(f"Warning: TTS provider init failed: {e}", fg="yellow", err=True)

        # Play ready sound BEFORE starting audio capture to avoid capturing the chime
        if config.get("voice_audio_cues", True):
            from episodic.voice.audio_playback import play_chime
            play_chime()

        # Start audio capture (after chime finishes)
        capture.start(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )

        # Start playback thread
        playback.start()

        self._set_state(VoiceState.LISTENING)

        # Start idle timer for wake word mode
        self._start_idle_timer()

    def stop(self):
        """Stop voice mode."""
        if self._state == VoiceState.OFF:
            return

        # Cancel idle timer
        self._cancel_idle_timer()

        # Stop audio capture
        if self._audio_capture:
            self._audio_capture.stop()

        # Stop playback
        if self._audio_playback:
            self._audio_playback.stop()

        self._set_state(VoiceState.OFF)

        # Clear all cached components so they'll be recreated with current config
        # next time voice mode is enabled (allows /mode switch and config changes to take effect)
        self._stt_provider = None
        self._tts_provider = None
        self._audio_capture = None
        self._audio_playback = None

        # Cleanup wake word detector (releases Porcupine resources)
        if self._wake_word_detector is not None:
            self._wake_word_detector.cleanup()
            self._wake_word_detector = None

        # Release any waiting listen() calls
        self._transcription_event.set()

    def listen(self, timeout: float = 30.0) -> Optional[str]:
        """
        Wait for and return transcribed speech.

        Blocks until speech is detected and transcribed.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Transcribed text, or None on timeout/error
        """
        if self._state == VoiceState.OFF:
            return None

        self._transcription_result = None
        self._transcription_event.clear()

        # Wait for transcription
        if self._transcription_event.wait(timeout):
            return self._transcription_result

        return None

    def speak(self, text: str, immediate: bool = False):
        """
        Speak text using TTS.

        Args:
            text: Text to speak
            immediate: If True, interrupt any current playback
        """
        if not config.get("voice_tts_enabled", True):
            return

        if not text.strip():
            return

        try:
            tts = self._get_tts_provider()
            audio, sample_rate = tts.synthesize(text)

            if len(audio) == 0:
                return

            playback = self._get_audio_playback()

            self._set_state(VoiceState.SPEAKING)

            if immediate:
                playback.play_immediate(audio, sample_rate)
            else:
                playback.enqueue(audio, sample_rate)

        except Exception as e:
            typer.secho(f"TTS error: {e}", fg="red", err=True)

    def speak_sentence(self, sentence: str):
        """
        Queue a sentence for TTS (non-blocking).

        Used by the sentence buffer during streaming.
        """
        self.speak(sentence, immediate=False)

    def interrupt_speech(self):
        """
        Interrupt current TTS playback.

        Clears the audio queue with a fade-out to prevent clicks.
        Used when user presses a key to stop speech output.
        """
        if self._audio_playback:
            self._audio_playback.clear_queue()

    def wait_for_speech_complete(self):
        """Wait for all queued TTS to finish playing."""
        if self._audio_playback:
            self._audio_playback.finish()  # Apply fade-out to prevent click
            self._audio_playback.wait()

    def mute(self):
        """Mute microphone input."""
        if self._audio_capture:
            self._audio_capture.mute()

    def unmute(self):
        """Unmute microphone input."""
        if self._audio_capture:
            self._audio_capture.unmute()

    @property
    def state(self) -> VoiceState:
        """Get current voice mode state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if voice mode is active."""
        return self._state != VoiceState.OFF

    @property
    def is_listening(self) -> bool:
        """Check if currently listening for speech."""
        return self._state == VoiceState.LISTENING

    @property
    def is_idle(self) -> bool:
        """Check if in idle/wake word mode."""
        return self._state == VoiceState.IDLE

    @property
    def is_speaking(self) -> bool:
        """Check if currently playing TTS."""
        return self._state == VoiceState.SPEAKING


def get_voice_manager() -> VoiceModeManager:
    """Get the singleton voice manager instance."""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceModeManager()
    return _voice_manager


def cleanup_voice_mode():
    """Clean up voice mode resources."""
    global _voice_manager

    if _voice_manager is not None:
        _voice_manager.stop()
        _voice_manager = None

    # Clean up providers
    from episodic.voice.stt_providers import cleanup_stt_providers
    from episodic.voice.tts_providers import cleanup_tts_providers

    cleanup_stt_providers()
    cleanup_tts_providers()
